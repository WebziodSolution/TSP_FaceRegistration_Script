import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv
load_dotenv()

# Use DATABASE_URL_PROD if ENV=prod, else DATABASE_URL (local/dev)
ENV = os.getenv("ENV", "local").lower()
print("============ ENV ================"+ENV)
if ENV == "prod":
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL_PROD")
else:
    SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if not SQLALCHEMY_DATABASE_URL:
    raise RuntimeError("No database URL found in environment variables.")

# Log DB connection attempt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    # Try connecting to DB
    with engine.connect() as conn:        
        logging.info("-------------- Database connection successful. --------------")
except Exception as e:
    logging.error(f"Database connection failed: {e}", exc_info=True)
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()