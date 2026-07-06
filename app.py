import os
# Suppress TensorFlow warnings and optimize memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure TensorFlow to use CPU only (reduces memory overhead)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU, use CPU only

# Set memory limits for TensorFlow
import tensorflow as tf
tf.config.set_soft_device_placement(True)

# Limit CPU threads to reduce memory usage
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Now import the rest of your modules
import faiss
import numpy as np
import json
import logging
import time
from typing import List, Dict, Optional
from threading import Lock
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import cv2
from deepface import DeepFace


from database import SessionLocal, engine, Base, get_db
from db.employees import CompanyEmployee
from db.company import Company

# --- Configuration ---
# EMBEDDING_DIM = 512 # ArcFace default is 512
EMBEDDING_DIM = 128  # Smaller embedding size for lighter models
FACE_MODEL = "Facenet"  # or "OpenFace", "VGG-Face", "Facenet512"
# Adjust this value based on your testing and observed L2 distances for same faces.
# Common range for L2 normalized ArcFace embeddings is 0.8 to 1.2 for same person.
FACE_MATCH_THRESHOLD = 0.8 # Adjusted threshold for better accuracy with L2 normalized ArcFace
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "face_index.bin")
ID_MAP_FILE = os.path.join(FAISS_INDEX_DIR, "id_map.json")

# --- Global FAISS Index and ID Map ---
GLOBAL_FAISS_INDEX: Optional[faiss.Index] = None

# This map will store {FAISS_internal_id: DB_user_id}
GLOBAL_ID_MAP: Dict[int, int] = {}
# Lock for thread-safe access to FAISS index and ID map during modifications
FAISS_LOCK = Lock()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Face Recognition API", description="API for face recognition login and registration.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ✅ Allow all origins
    allow_credentials=True,    # ✅ Allow cookies/auth headers
    allow_methods=["*"],       # ✅ Allow all HTTP methods
    allow_headers=["*"],       # ✅ Allow all headers
)

# --- Helper Functions ---
def validate_image(image_bytes: bytes):
    """
    Validates if the provided bytes represent a valid image and decodes it into a NumPy array.
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. It might be corrupted or an unsupported format.")
        return img
    except Exception as e:
        logging.error(f"Image validation error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    

def get_embedding(image_np: np.ndarray) -> List[float]:
    """
    Generates a 512-dimensional face embedding using DeepFace's ArcFace model.
    Returns a normalized embedding list. Raises HTTPException on failure.
    """
    try:
        # Optimize image size before processing
        # Resize to reduce memory usage (ArcFace works well with 112x112)
        height, width = image_np.shape[:2]
        if height > 640 or width > 640:
            # Maintain aspect ratio
            scale = 640 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_np = cv2.resize(image_np, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        
          # Use Facenet which is lighter than ArcFace
        embedding_objs = DeepFace.represent(
            img_path=image_np,
            model_name=FACE_MODEL,  # Use configured model
            enforce_detection=True,
            detector_backend='opencv',  # Lightweight detector
            align=False
        )

        if not embedding_objs:
            raise ValueError("No face embedding returned. Possibly no face detected.")

        # Extract and normalize the first detected face's embedding
        embedding = np.array(embedding_objs[0]["embedding"], dtype="float32")
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Zero-norm embedding vector detected. Cannot normalize.")

        normalized_embedding = embedding / norm
        return normalized_embedding.tolist()

    except ValueError as ve:
        logging.error(f"Face detection error: {ve}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="No face detected in the provided image. Please upload a clear face photo with a visible frontal face."
        )
    except Exception as e:
        logging.error(f"Embedding generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating face embedding: {str(e)}"
        )

# --- FAISS Index Management ---
def initialize_faiss_index(db: Session):
    """
    Initializes or reloads the FAISS index and its associated ID map.
    Attempts to load from disk; if unsuccessful, rebuilds from the database.
    """
    global GLOBAL_FAISS_INDEX, GLOBAL_ID_MAP
    
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

    with FAISS_LOCK:
        logging.info("Acquired FAISS_LOCK for initialization.")
        loaded_successfully = False
        
        # FIRST: Check if the index file exists
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(ID_MAP_FILE):
            try:
                logging.info(f"Attempting to load FAISS index from {FAISS_INDEX_FILE} and ID map from {ID_MAP_FILE}...")
                
                # Try to load the index
                GLOBAL_FAISS_INDEX = faiss.read_index(FAISS_INDEX_FILE)
                assert GLOBAL_FAISS_INDEX is not None, "faiss.read_index returned None unexpectedly."
                
                # Check if loaded index has the right dimension
                if GLOBAL_FAISS_INDEX.d != EMBEDDING_DIM:
                    logging.warning(f"FAISS index dimension mismatch! Loaded: {GLOBAL_FAISS_INDEX.d}, Expected: {EMBEDDING_DIM}. Creating new index.")
                    
                    # Close the old index and create new one
                    GLOBAL_FAISS_INDEX = None
                    GLOBAL_ID_MAP = {}
                    
                    # Remove old files
                    try:
                        os.remove(FAISS_INDEX_FILE)
                        os.remove(ID_MAP_FILE)
                        logging.info("Removed old dimension-mismatched FAISS index files.")
                    except:
                        pass
                    
                    # Set flag to create new index
                    loaded_successfully = False
                else:
                    # Dimensions match, load ID map
                    with open(ID_MAP_FILE, 'r') as f:
                        temp_map = json.load(f)
                        GLOBAL_ID_MAP = {int(k): v for k, v in temp_map.items()}
                    
                    logging.info(f"Loaded FAISS index with {GLOBAL_FAISS_INDEX.ntotal} vectors and ID map with {len(GLOBAL_ID_MAP)} entries.")
                    loaded_successfully = True
                    
            except Exception as e:
                logging.error(f"Error loading FAISS index or ID map: {e}. Will attempt to rebuild from database.", exc_info=True)
                GLOBAL_FAISS_INDEX = None
                GLOBAL_ID_MAP = {}
        
        # If not loaded successfully, create new index
        if not loaded_successfully:
            logging.info("Initializing new FAISS index and populating from database...")
            
            # Initialize a new HNSW index with the NEW embedding dimension
            GLOBAL_FAISS_INDEX = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_L2)
            GLOBAL_FAISS_INDEX.hnsw.efConstruction = 100
            GLOBAL_FAISS_INDEX.hnsw.efSearch = 50

            # Populate index from database - but ONLY if embeddings have the right dimension
            users = db.query(CompanyEmployee).all()
            if users:
                embeddings_to_add = []
                user_ids_to_map = [] 
                for user in users:
                    if user.embedding is not None:
                        # Check if embedding has the right dimension
                        if len(user.embedding) == EMBEDDING_DIM:
                            embeddings_to_add.append(np.array(user.embedding, dtype='float32'))
                            user_ids_to_map.append(user.id)
                        else:
                            logging.warning(f"User {user.id} has embedding with wrong dimension {len(user.embedding)}. Skipping.")
                            # Optionally: Clear the old embedding
                            # setattr(user, "embedding", None)
                            # db.commit()
                
                if embeddings_to_add:
                    embeddings_matrix = np.array(embeddings_to_add).astype('float32')
                    GLOBAL_FAISS_INDEX.add(embeddings_matrix)
                    
                    # Populate GLOBAL_ID_MAP after adding to FAISS
                    for i, db_id in enumerate(user_ids_to_map):
                        GLOBAL_ID_MAP[i] = db_id
                    
                    logging.info(f"Rebuilt FAISS index with {GLOBAL_FAISS_INDEX.ntotal} vectors from DB.")
                else:
                    logging.info("No valid embeddings in DB (all have wrong dimension). FAISS index initialized empty.")
            else:
                logging.info("No users in DB. FAISS index initialized empty.")
        
        logging.info("Released FAISS_LOCK after initialization.")
        
def save_faiss_index():
    """
    Saves the current FAISS index and ID map to disk.
    This operation is performed under a lock and ideally in a background task.
    """
    global GLOBAL_FAISS_INDEX, GLOBAL_ID_MAP

    logging.info("Attempting to save FAISS index...")
    start_time = time.time()

    with FAISS_LOCK:
        try:
            logging.info("Acquired FAISS_LOCK for saving index")

            # Ensure FAISS index directory exists before saving
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

            if GLOBAL_FAISS_INDEX is None:
                logging.warning("FAISS index is None, creating empty index for saving")
                GLOBAL_FAISS_INDEX = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_L2)
                GLOBAL_FAISS_INDEX.hnsw.efConstruction = 100
                GLOBAL_FAISS_INDEX.hnsw.efSearch = 50
                GLOBAL_ID_MAP = {}

            # 🚨 SKIP saving if index is empty (this prevents freeze!)
            if GLOBAL_FAISS_INDEX.ntotal == 0:
                logging.warning("FAISS index is empty, skipping save to avoid freeze.")
                return

            # Save FAISS index
            logging.info(f"Saving FAISS index with {GLOBAL_FAISS_INDEX.ntotal} vectors...")
            faiss.write_index(GLOBAL_FAISS_INDEX, FAISS_INDEX_FILE)

            # Save ID map
            logging.info(f"Saving ID map with {len(GLOBAL_ID_MAP)} entries...")
            with open(ID_MAP_FILE, 'w') as f:
                json.dump({str(k): v for k, v in GLOBAL_ID_MAP.items()}, f)

            logging.info(f"FAISS index and ID map saved successfully in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"Failed to save FAISS index: {str(e)}", exc_info=True)
        finally:
            logging.info("Released FAISS_LOCK after saving attempt")



# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler. Initializes database and FAISS index.
    """
    logging.info("Application startup event triggered.")
    db = SessionLocal() # Get a database session
    try:
        Base.metadata.create_all(bind=engine) # Create database tables if they don't exist
        initialize_faiss_index(db) # Initialize FAISS
    except Exception as e:
        logging.critical(f"Failed to initialize database or FAISS index on startup: {e}", exc_info=True)
        # In a production environment, you might want to sys.exit(1) here
        # or implement a more robust health check and recovery mechanism.
    finally:
        db.close() # Ensure the session is closed

@app.on_event("shutdown")
async def shutdown_event():
    """
    FastAPI shutdown event handler. Saves the FAISS index to disk.
    """
    logging.info("Application shutdown event triggered. Saving FAISS index...")
    save_faiss_index()

# --- API Endpoints ---

@app.post("/register")
async def register(
    background_tasks: BackgroundTasks,
    employeeId: int = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Registers a face for an existing employee (identified by employeeId).
    Checks for duplicate faces before registration.
    """
    try:
        # Validate image and get embedding
        image_bytes = await image.read()
        image_np = validate_image(image_bytes)
        embedding = get_embedding(image_np)

        # Ensure employee exists
        employee = db.query(CompanyEmployee).filter(CompanyEmployee.id == employeeId).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee ID not found.")

        # Check if this employee already has a registered face
        if employee.embedding is not None:
            raise HTTPException(
                status_code=400,
                detail="This employee already has a registered face. Please clear the existing face first."
            )

        # Check for duplicate face in the system
        with FAISS_LOCK:
            assert GLOBAL_FAISS_INDEX is not None, "FAISS index not initialized."

            if GLOBAL_FAISS_INDEX.ntotal > 0:
                query_embedding_np = np.array([embedding], dtype='float32')
                # Search for the closest match
                distances, labels = GLOBAL_FAISS_INDEX.search(query_embedding_np, k=1)  # type: ignore
                best_distance = distances[0][0]
                best_faiss_id = labels[0][0]

                # If a close match is found below threshold
                if best_faiss_id != -1 and best_distance < FACE_MATCH_THRESHOLD:
                    matched_db_id = GLOBAL_ID_MAP.get(best_faiss_id)
                    if matched_db_id:
                        existing_user = db.query(CompanyEmployee).filter(
                            CompanyEmployee.id == matched_db_id
                        ).first()
                        if existing_user:
                            raise HTTPException(
                                status_code=400,
                                detail=f"This face is already registered."
                            )

        # Save embedding to DB
        setattr(employee, "embedding", list(map(float, embedding)))
        db.commit()

        # Update FAISS index
        with FAISS_LOCK:
            embedding_np = np.array([embedding], dtype='float32')
            new_faiss_id = GLOBAL_FAISS_INDEX.ntotal
            GLOBAL_FAISS_INDEX.add(embedding_np)  # type: ignore
            GLOBAL_ID_MAP[new_faiss_id] = int(getattr(employee, "id"))
            background_tasks.add_task(save_faiss_index)

        return {"success": True, "message": "Face registered successfully."}

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error during registration: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login")
async def login(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Authenticates a user by comparing a provided face image with registered faces.
    """
    try:
        # Acquire lock for FAISS operations
        with FAISS_LOCK:
            assert GLOBAL_FAISS_INDEX is not None, "FAISS index is not initialized during login."

            if GLOBAL_FAISS_INDEX.ntotal == 0:
                logging.warning("Login attempt when FAISS index is empty. No users registered.")
                raise HTTPException(status_code=404, detail="No registered users found with this face. Please register first.")

            # 1. Image Processing and Embedding Generation
            image_bytes = await file.read()
            image_np = validate_image(image_bytes)
            query_embedding = get_embedding(image_np) # This includes L2 normalization
            query_embedding_np = np.array([query_embedding], dtype='float32')

            # 2. Search FAISS index for the closest match
            distances, faiss_indices = GLOBAL_FAISS_INDEX.search(query_embedding_np, k=1)  # type: ignore

            best_distance = distances[0][0]
            best_faiss_id = faiss_indices[0][0]

            # 3. Evaluate Match
            if best_faiss_id != -1 and best_distance < FACE_MATCH_THRESHOLD:
                matched_db_id = GLOBAL_ID_MAP.get(best_faiss_id)
                
                if matched_db_id is not None:
                    matched_user = db.query(CompanyEmployee).filter(CompanyEmployee.id == matched_db_id).first()
                    if matched_user:
                        company_details = db.query(Company).filter(Company.id == matched_user.company_id).first()
                        logging.info(f"Login successful for user '{matched_user.user_name}' (DB ID: {matched_user.id}). Distance: {best_distance:.4f} < Threshold: {FACE_MATCH_THRESHOLD:.4f}")
                        return {
                            "success": True,
                            "userName": matched_user.user_name,
                            "companyId": company_details.company_no if company_details is not None else None,
                            "password": matched_user.password,  # Include password if needed for further processing
                            "distance": float(best_distance),
                            # Confidence can be derived from distance. 
                            # If distance ranges 0-2, 1 - (dist/2) gives 1 to 0 confidence.
                            "confidence": float(1 - (best_distance / 2.0)), 
                            "message": "Login successful!"
                        }
                    else:
                        logging.warning(f"FAISS matched ID {best_faiss_id} mapped to DB ID {matched_db_id}, but user not found in DB. Index/map might be out of sync. Rebuild recommended.")
                else:
                    logging.warning(f"FAISS matched ID {best_faiss_id}, but no mapping found in GLOBAL_ID_MAP. Index/map might be corrupted. Rebuild recommended.")

        # If we reach here, no sufficient match was found or data was inconsistent
        detail_message = "Face not clear. Adjust position and retry"
        # if best_faiss_id != -1:
        #     detail_message += f" (Closest match L2 distance: {best_distance:.4f} which is >= Threshold: {FACE_MATCH_THRESHOLD:.4f})"
        #     detail_message
        # else:
        #     detail_message += " No close match found."

        if best_faiss_id != -1:
            logging.info(f"Login failed. Closest match L2 distance: {best_distance:.4f} (>= {FACE_MATCH_THRESHOLD:.4f})")
            detail_message = "Face not clear. Adjust position and retry."
        else:
            logging.info("Login failed. No close match found in database.")
            detail_message = "No match found. Please register first."

        logging.info(f"Login failed. {detail_message}")
        raise HTTPException(status_code=401, detail=detail_message)

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error during login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unhandled error during login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{str(e)}")
    

@app.delete("/clear-all-data", include_in_schema=False)
async def clear_all_data(db: Session = Depends(get_db)):
    """
    DEBUG/ADMIN ONLY: Clears all face embedding data from the database and resets the FAISS index.
    User records are NOT deleted, only their embeddings ar  e cleared.
    NOT FOR PRODUCTION USE WITHOUT CAREFUL SECURITY CONSIDERATIONS.
    """
    global GLOBAL_FAISS_INDEX, GLOBAL_ID_MAP
    with FAISS_LOCK:
        logging.warning("Acquired FAISS_LOCK. Initiating embedding data clear!")
        try:
            # Set embedding to None for all users (do not delete users)
            db.query(CompanyEmployee).filter(CompanyEmployee.embedding != None).update(
                {CompanyEmployee.embedding: None}, synchronize_session=False
            )
            db.commit()
            logging.info("All embeddings cleared from company_employees table.")

            # Reinitialize FAISS index to empty
            GLOBAL_FAISS_INDEX = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_L2)
            GLOBAL_FAISS_INDEX.hnsw.efConstruction = 100
            GLOBAL_FAISS_INDEX.hnsw.efSearch = 50
            GLOBAL_ID_MAP = {}
            logging.info("FAISS index reinitialized to empty.")

            # Clean up disk files
            if os.path.exists(FAISS_INDEX_FILE):
                os.remove(FAISS_INDEX_FILE)
                logging.info(f"Deleted FAISS index file: {FAISS_INDEX_FILE}")
            if os.path.exists(ID_MAP_FILE):
                os.remove(ID_MAP_FILE)
                logging.info(f"Deleted ID map file: {ID_MAP_FILE}")
            
            logging.info("All face embeddings and FAISS index data cleared successfully.")
            return {"message": "All face embeddings and FAISS index data cleared. User records remain."}
        except Exception as e:
            db.rollback()
            logging.error(f"Error clearing all embedding data: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to clear all embedding data: {e}")
        finally:
            logging.warning("Released FAISS_LOCK after embedding data clear attempt.")


@app.delete("/clear-embedding/{user_id}", include_in_schema=True)
async def clear_single_embedding( background_tasks: BackgroundTasks,user_id: int, db: Session = Depends(get_db)):
    """
    Clears the face embedding for a single user (by user_id) without deleting the user record.
    Also removes the embedding from the FAISS index and updates the index/map.
    """
    global GLOBAL_FAISS_INDEX, GLOBAL_ID_MAP
    with FAISS_LOCK:
        try:
            user = db.query(CompanyEmployee).filter(CompanyEmployee.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found.")

            if user.embedding is None:
                return {"message": f"User already has no embedding."}

            # Remove embedding from DB
            setattr(user, "embedding", None)
            db.commit()
            logging.info(f"Embedding cleared for user {user_id} in DB.")

            # Rebuild FAISS index and ID map from all valid users
            GLOBAL_FAISS_INDEX = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32, faiss.METRIC_L2)
            GLOBAL_FAISS_INDEX.hnsw.efConstruction = 100
            GLOBAL_FAISS_INDEX.hnsw.efSearch = 50
            GLOBAL_ID_MAP = {}

            users_with_embeddings = db.query(CompanyEmployee).filter(CompanyEmployee.embedding != None).all()

            embeddings_to_add = []
            user_ids_to_map = []

            for u in users_with_embeddings:
                embedding = np.array(u.embedding, dtype='float32')
                if embedding.shape != (EMBEDDING_DIM,):
                    logging.warning(f"Skipping user {u.id} due to invalid embedding shape: {embedding.shape}")
                    continue
                embeddings_to_add.append(embedding)
                user_ids_to_map.append(u.id)

            if embeddings_to_add:
                embeddings_matrix = np.vstack(embeddings_to_add).astype('float32')
                GLOBAL_FAISS_INDEX.add(embeddings_matrix)  # type: ignore

                for i, db_id in enumerate(user_ids_to_map):
                    GLOBAL_ID_MAP[i] = db_id

                logging.info(f"FAISS index rebuilt with {GLOBAL_FAISS_INDEX.ntotal} vectors after clearing embedding for user {user_id}.")
            else:
                logging.info("No valid embeddings left in DB. FAISS index is now empty.")

            # Clean up and save FAISS index
            try:
                logging.info("Cleaning up and saving FAISS index...")
                if os.path.exists(FAISS_INDEX_FILE):
                    logging.info(f"Removing existing FAISS index file: {FAISS_INDEX_FILE}")
                    os.remove(FAISS_INDEX_FILE)
                if os.path.exists(ID_MAP_FILE):
                    logging.info(f"Removing existing ID map file: {ID_MAP_FILE}")
                    os.remove(ID_MAP_FILE)
                logging.info("Saving updated FAISS index and ID map to disk...")
                background_tasks.add_task(save_faiss_index)
            except Exception as e:
                logging.error(f"Error while saving FAISS index or ID map: {e}", exc_info=True)

            logging.info("Returning success response to client.")
            return {"message": f"User face recognition data cleared successfully","status": "success"}

        except Exception as e:
            db.rollback()
            logging.error(f"Error clearing embedding for user: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"{e}")
        finally:
            logging.warning("Released FAISS_LOCK after single embedding clear attempt.")

@app.get("/test", include_in_schema=False)
async def health_check():
    """
    Health check endpoint to verify if the service is running.
    """
    return {"status": "ok", "message": "Face Recognition API is running."}

# To run the application:
# python -m venv venv
# face_env\Scripts\activate

# # Reinstall all dependencies cleanly
# pip install --upgrade pip
# pip install fastapi uvicorn[standard] sqlalchemy python-multipart opencv-python faiss-cpu pydantic python-dotenv deepface tensorflow==2.13.0 numpy==1.26.4 typing-extensions==4.12.2
# uvicorn app:app --reload --port 8000
# uvicorn app:app --reload --host 0.0.0.0 --port 8000