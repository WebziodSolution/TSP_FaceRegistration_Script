import os
import io
import time
import base64
import json
import logging
import sys
from typing import Optional, List, Union

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import get_db
from db.employees import CompanyEmployee
from db.company import Company

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FaceRecognitionAPI")

# --- Threshold Constants ---
# InsightFace ArcFace (buffalo_s) Cosine Similarity Thresholds:
# Normal matching for Login: 0.50 - 0.55
LOGIN_COSINE_SIMILARITY_THRESHOLD = 0.55
# Duplicate check threshold for Registration:
REGISTER_COSINE_SIMILARITY_THRESHOLD = 0.44
# Fallback for legacy 128D FaceNet / face-api.js embeddings:
EUCLIDEAN_MATCH_THRESHOLD = 0.45

# --- FastAPI Initialization ---
app = FastAPI(
    title="Face Recognition & InsightFace API",
    description="Unified API for Face Recognition Login, Registration, Embedding Management, and InsightFace 512D ArcFace Extraction.",
    version="2.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"-> {request.method} {request.url.path} from {client_ip}")
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    logger.info(f"<- {request.method} {request.url.path} status={response.status_code} in {duration:.2f}ms")
    return response


# --- InsightFace Model Loader ---
face_app = None

def get_face_app():
    global face_app
    if face_app is None:
        logger.info("Initializing InsightFace models (buffalo_s / CPUExecutionProvider)...")
        try:
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace models (buffalo_s) loaded and initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FaceAnalysis models: {e}", exc_info=True)
            raise
    return face_app

@app.on_event("startup")
def startup_event():
    try:
        get_face_app()
    except Exception as e:
        logger.warning(f"Warning during startup model pre-load: {e}")


# --- Image Decoding Helpers ---
def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return decode_image_bytes(image_bytes)

def extract_face_embedding(img: np.ndarray) -> dict:
    """
    Extracts L2-normalized 512D ArcFace embedding from a BGR image array.
    """
    if img is None:
        return {"success": False, "detail": "Failed to decode image data."}

    engine = get_face_app()
    faces = engine.get(img)

    if len(faces) == 0:
        logger.warning("No face detected in image.")
        return {"success": False, "detail": "No face detected. Please ensure your face is clearly visible."}

    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        main_area = (faces[0].bbox[2] - faces[0].bbox[0]) * (faces[0].bbox[3] - faces[0].bbox[1])
        second_area = (faces[1].bbox[2] - faces[1].bbox[0]) * (faces[1].bbox[3] - faces[1].bbox[1])
        if second_area > 0.4 * main_area:
            logger.warning(f"Multiple faces detected ({len(faces)} faces).")
            return {"success": False, "detail": "Multiple faces detected. Please show only one face in frame."}

    face = faces[0]
    raw_emb = face.embedding
    norm = np.linalg.norm(raw_emb)
    normalized_emb = (raw_emb / norm).tolist() if norm > 0 else raw_emb.tolist()

    det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
    gender = int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None
    age = int(face.age) if hasattr(face, 'age') and face.age is not None else None

    return {
        "success": True,
        "embedding": normalized_emb,
        "embedding_size": len(normalized_emb),
        "det_score": det_score,
        "bbox": [float(x) for x in face.bbox],
        "gender": gender,
        "age": age
    }


# --- Mathematical Comparison Helpers ---
def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return -1.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a <= 0 or norm_b <= 0:
        return -1.0
    return float(np.dot(va, vb) / (norm_a * norm_b))

def euclidean_distance(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 999.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(va - vb))

def parse_db_embedding(raw: Union[str, list, None]) -> Optional[List[float]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [float(x) for x in val]
        except Exception:
            return None
    return None


# --- Flexible Request Parser ---
async def extract_request_data(request: Request):
    """
    Extracts text/json fields and files from JSON, form-data, or query parameters.
    """
    content_type = request.headers.get("content-type", "")
    data = {}
    files = {}

    for k, v in request.query_params.items():
        data[k] = v

    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                data.update(body)
        except Exception:
            pass
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            for key, val in form.items():
                if hasattr(val, "filename") and val.filename:
                    files[key] = val
                else:
                    data[key] = val
        except Exception:
            pass

    return data, files

async def resolve_face_descriptor(data: dict, files: dict) -> Optional[List[float]]:
    """
    Extracts face descriptor from uploaded file, base64 image, or pre-computed descriptor.
    """
    # 1. Uploaded file (multipart)
    uploaded = files.get("image") or files.get("file")
    if uploaded is not None:
        contents = await uploaded.read()
        img = decode_image_bytes(contents)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image file.")
        res = extract_face_embedding(img)
        if not res["success"]:
            raise HTTPException(status_code=400, detail=res["detail"])
        return res["embedding"]

    # 2. Base64 image
    img_b64 = data.get("image") or data.get("image_base64")
    if isinstance(img_b64, str) and (img_b64.startswith("data:image") or len(img_b64) > 100):
        img = decode_base64_image(img_b64)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode base64 image.")
        res = extract_face_embedding(img)
        if not res["success"]:
            raise HTTPException(status_code=400, detail=res["detail"])
        return res["embedding"]

    # 3. Pre-computed faceDescriptor
    raw_desc = data.get("faceDescriptor")
    if raw_desc is not None:
        if isinstance(raw_desc, str):
            try:
                raw_desc = json.loads(raw_desc)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid faceDescriptor format. Must be a JSON array.")
        if isinstance(raw_desc, list) and len(raw_desc) > 0:
            return [float(x) for x in raw_desc]
        raise HTTPException(status_code=400, detail="Invalid faceDescriptor format. Must be a list of numbers.")

    return None


# ==============================================================================
# 1. LOGIN ROUTE (/login and /login.php)
# ==============================================================================
@app.post("/login")
async def login(request: Request, db: Session = Depends(get_db)):
    """
    Matches uploaded face image / InsightFace 512D ArcFace embedding against database embeddings.
    Supports file upload ('image'), base64 image ('image' / 'image_base64'), or 'faceDescriptor'.
    """
    data, files = await extract_request_data(request)
    login_descriptor = await resolve_face_descriptor(data, files)

    if not login_descriptor or not isinstance(login_descriptor, list):
        raise HTTPException(status_code=400, detail="No valid face image or descriptor provided for login.")

    login_dim = len(login_descriptor)
    logger.info(f"Login request with descriptor dimension: {login_dim}")

    try:
        # Retrieve all employees with registered faces
        employees = db.query(CompanyEmployee).filter(CompanyEmployee.embedding.isnot(None)).all()

        best_match = None
        highest_similarity = -1.0
        best_distance = 999.0

        for employee in employees:
            saved_embedding = parse_db_embedding(employee.embedding)
            if not saved_embedding:
                continue

            saved_dim = len(saved_embedding)

            # InsightFace 512D ArcFace Cosine Matching
            if login_dim == 512 and saved_dim == 512:
                sim = cosine_similarity(login_descriptor, saved_embedding)
                if sim > highest_similarity:
                    highest_similarity = sim
                    if sim >= LOGIN_COSINE_SIMILARITY_THRESHOLD:
                        best_match = employee

            # Legacy 128D Euclidean Matching
            elif login_dim == 128 and saved_dim == 128:
                dist = euclidean_distance(login_descriptor, saved_embedding)
                if dist < best_distance:
                    best_distance = dist
                    if dist < EUCLIDEAN_MATCH_THRESHOLD:
                        best_match = employee

        if best_match is not None:
            company_details = None
            if best_match.company_id is not None:
                company_details = db.query(Company).filter(Company.id == best_match.company_id).first()

            confidence = (
                f"{round(highest_similarity * 100, 2):.2f}%"
                if login_dim == 512
                else f"{round((1.0 - (best_distance / 2.0)) * 100, 2):.2f}%"
            )

            logger.info(f"Login successful for user '{best_match.user_name}' (ID: {best_match.id}). Similarity/Distance: {highest_similarity if login_dim == 512 else best_distance}")

            return {
                "success": True,
                "userName": best_match.user_name,
                "companyId": company_details.company_no if company_details else None,
                "password": best_match.password,
                "similarity": float(highest_similarity) if login_dim == 512 else None,
                "distance": float(best_distance) if login_dim == 128 else None,
                "confidence": confidence,
                "message": "Login successful!"
            }
        else:
            logger.warning("Login failed: Face not recognized.")
            raise HTTPException(
                status_code=401,
                detail="Face not recognized. Please login using Username & Password."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


# ==============================================================================
# 2. REGISTER ROUTE (/register and /register.php)
# ==============================================================================
@app.post("/register")
async def register(request: Request, db: Session = Depends(get_db)):
    """
    Registers a new face descriptor / InsightFace 512D ArcFace embedding for an employee.
    Validates employeeId, ensures no existing embedding, and prevents duplicate faces.
    """
    data, files = await extract_request_data(request)

    raw_emp_id = data.get("employeeId")
    if raw_emp_id is None:
        raise HTTPException(status_code=400, detail="employeeId is a required parameter.")

    try:
        employee_id = int(raw_emp_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="employeeId must be an integer.")

    new_descriptor = await resolve_face_descriptor(data, files)
    if not new_descriptor or not isinstance(new_descriptor, list):
        raise HTTPException(status_code=400, detail="No valid face image or descriptor provided for registration.")

    descriptor_dim = len(new_descriptor)
    logger.info(f"Registering face for employee ID {employee_id} (dimension: {descriptor_dim})")

    try:
        # Ensure employee exists
        employee = db.query(CompanyEmployee).filter(CompanyEmployee.id == employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="Employee ID not found.")

        # Check if this employee already has a registered face
        if employee.embedding is not None and str(employee.embedding).strip() != "":
            raise HTTPException(
                status_code=400,
                detail="This employee already has a registered face. Please clear the existing face first."
            )

        # Check if the face is already registered with another employee
        other_employees = db.query(CompanyEmployee).filter(
            CompanyEmployee.embedding.isnot(None),
            CompanyEmployee.id != employee_id
        ).all()

        for other in other_employees:
            saved_embedding = parse_db_embedding(other.embedding)
            if not saved_embedding:
                continue

            saved_dim = len(saved_embedding)

            # Compare 512D ArcFace descriptors via Cosine Similarity
            if descriptor_dim == 512 and saved_dim == 512:
                sim = cosine_similarity(new_descriptor, saved_embedding)
                if sim >= REGISTER_COSINE_SIMILARITY_THRESHOLD:
                    logger.warning(f"Registration duplicate: matches {other.user_name} with similarity {sim:.4f}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"This face is already registered with employee: {other.user_name}"
                    )

            # Compare legacy 128D descriptors via Euclidean distance
            elif descriptor_dim == 128 and saved_dim == 128:
                dist = euclidean_distance(new_descriptor, saved_embedding)
                if dist < EUCLIDEAN_MATCH_THRESHOLD:
                    logger.warning(f"Registration duplicate: matches {other.user_name} with distance {dist:.4f}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"This face is already registered with employee: {other.user_name}"
                    )

        # Save new descriptor
        employee.embedding = json.dumps(new_descriptor)
        db.commit()

        logger.info(f"Face registered successfully for employee ID {employee_id}.")
        return {
            "success": True,
            "message": "Face registered successfully.",
            "dimensions": descriptor_dim
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


# ==============================================================================
# 3. CLEAR EMBEDDING ROUTE (/clear-embedding and aliases)
# ==============================================================================
async def execute_clear_embedding(user_id: Optional[int], db: Session):
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is a required parameter.")

    try:
        user = db.query(CompanyEmployee).filter(CompanyEmployee.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        if user.embedding is not None:
            user.embedding = None
            db.commit()
            logger.info(f"User face recognition data cleared for user ID {user_id}.")

        return {
            "message": "User face recognition data cleared successfully",
            "status": "success"
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear embedding: {str(e)}")

@app.delete("/clear-embedding/{user_id}")
async def clear_embedding_path(user_id: int, db: Session = Depends(get_db)):
    return await execute_clear_embedding(user_id, db)

@app.delete("/clear-embedding")
async def clear_embedding_body_or_query(request: Request, db: Session = Depends(get_db)):
    data, _ = await extract_request_data(request)
    raw_id = data.get("user_id") or data.get("id")
    user_id = None
    if raw_id is not None:
        try:
            user_id = int(raw_id)
        except (ValueError, TypeError):
            pass
    return await execute_clear_embedding(user_id, db)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Face Recognition & InsightFace API with model buffalo_s",
        "model": "buffalo_s"
    }

# sudo systemctl daemon-reload
# sudo systemctl enable faceRegistration_service.service
# sudo systemctl start faceRegistration_service.service
# sudo systemctl status faceRegistration_service.service
# sudo systemctl restart faceRegistration_service.service
# sudo systemctl stop faceRegistration_service.service