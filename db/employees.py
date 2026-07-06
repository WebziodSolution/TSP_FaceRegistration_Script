from sqlalchemy import Column, Integer, String, Float, Boolean, Date, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CompanyEmployee(Base):
    __tablename__ = "company_employees"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=True)
    user_name = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=True)
    embedding = Column(JSON, nullable=True)  # Set to nullable if not all employees have embeddings