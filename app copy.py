import cv2
import numpy as np
from typing import List
import numpy as np
import json
import logging
import time
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Form, Depends, BackgroundTasks, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from database import get_db
from db.employees import CompanyEmployee
from db.company import Company

# --- Configuration ---
# L2 distance threshold of 0.6 is standard for face-api.js.
# 0.5 to 0.55 is a good balance for security and user convenience.
FACE_MATCH_THRESHOLD = 0.52

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




def euclidean_distance(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 999.0
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# --- API Endpoints ---

@app.post("/register")
async def register(
    background_tasks: BackgroundTasks,
    employeeId: int = Form(...),
    faceDescriptor: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Registers a face for an existing employee (identified by employeeId).
    Checks for duplicate faces before registration.
    """
    try:                
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

        # Parse faceDescriptor
        try:
            new_descriptor = json.loads(faceDescriptor)
        except Exception as json_err:
            logging.error(f"Failed to parse faceDescriptor JSON: {json_err}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail="Invalid faceDescriptor format. Must be a JSON-encoded array of numbers."
            )
        
        if not isinstance(new_descriptor, list):
            raise HTTPException(
                status_code=400,
                detail="Invalid faceDescriptor format. Must be a list."
            )

        # Check if the face is already registered with another employee
        other_employees = db.query(CompanyEmployee).filter(
            CompanyEmployee.embedding.isnot(None),
            CompanyEmployee.id != employeeId
        ).all()

        logging.info(f"[Register Check] Found {len(other_employees)} other employees with registered embeddings.")

        for other in other_employees:
            saved_embedding = other.embedding
            if isinstance(saved_embedding, str):
                try:
                    saved_embedding = json.loads(saved_embedding)
                except Exception as parse_err:
                    logging.warning(f"[Register Check] Failed to parse embedding string for employee ID {other.id}: {parse_err}")
                    continue
            
            if isinstance(saved_embedding, list):
                dist = euclidean_distance(new_descriptor, saved_embedding)
                logging.info(f"[Register Check] Comparing new descriptor (len: {len(new_descriptor)}) with employee ID {other.id} ({other.user_name}, len: {len(saved_embedding)}). Distance: {dist:.6f}")
                if dist < FACE_MATCH_THRESHOLD:  # Threshold matching login
                    logging.warning(f"[Register Check] Face match found! Employee ID {other.id} ({other.user_name}) has distance {dist:.6f} < {FACE_MATCH_THRESHOLD}")
                    raise HTTPException(
                        status_code=400,
                        detail="This face is already registered."
                    )

        employee.embedding = new_descriptor
        db.commit()      
        return {"success": True, "message": "Face registered successfully."}

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error during registration: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login")
async def login(
    faceDescriptor: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Authenticates a user by comparing a provided face image with registered faces.
    """
    try:
        # Parse faceDescriptor
        try:
            login_descriptor = json.loads(faceDescriptor)
        except Exception as json_err:
            logging.error(f"Failed to parse faceDescriptor JSON: {json_err}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail="Invalid faceDescriptor format. Must be a JSON-encoded array of numbers."
            )
        
        if not isinstance(login_descriptor, list):
            raise HTTPException(
                status_code=400,
                detail="Invalid faceDescriptor format. Must be a list."
            )

        # Retrieve all employees with registered faces
        employees = db.query(CompanyEmployee).filter(CompanyEmployee.embedding.isnot(None)).all()

        best_match = None
        best_distance = FACE_MATCH_THRESHOLD

        for employee in employees:
            saved_embedding = employee.embedding
            # If stored as JSON string (due to previous bug / legacy format), parse it
            if isinstance(saved_embedding, str):
                try:
                    saved_embedding = json.loads(saved_embedding)
                except Exception:
                    continue
            
            if isinstance(saved_embedding, list):
                dist = euclidean_distance(login_descriptor, saved_embedding)
                if dist < best_distance:
                    best_distance = dist
                    best_match = employee

        if best_match is not None:
            company_details = db.query(Company).filter(Company.id == best_match.company_id).first()
            logging.info(f"Login successful for user '{best_match.user_name}' (DB ID: {best_match.id}). Distance: {best_distance:.4f} < Threshold: 0.45")
            
            return {
                "success": True,
                "userName": best_match.user_name,
                "companyId": company_details.company_no if company_details is not None else None,
                "password": best_match.password,
                "distance": float(best_distance),
                "confidence": float(1.0 - (best_distance / 2.0)),
                "message": "Login successful!"
            }
        else:
            raise HTTPException(
                status_code=401,
                detail="Face not recognized. Please login using Username & Password."
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error during login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


@app.delete("/clear-embedding/{user_id}", include_in_schema=True)
async def clear_single_embedding(user_id: int, db: Session = Depends(get_db)):
    """
    Clears the face embedding for a single user (by user_id) without deleting the user record.
    Also removes the embedding from the FAISS index and updates the index/map.
    """
    try:
        user = db.query(CompanyEmployee).filter(CompanyEmployee.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        if user.embedding is not None:
            # Remove embedding from DB
            setattr(user, "embedding", None)
            db.commit()
            logging.info(f"Embedding cleared for user {user_id} in DB.")
            return {"message": f"User face recognition data cleared successfully","status": "success"}      
    except Exception as e:
        db.rollback()
        logging.error(f"Error clearing embedding for user: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{e}") 

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