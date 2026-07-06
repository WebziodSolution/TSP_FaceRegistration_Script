from sqlalchemy import Column, Integer, String, Float, Boolean, Date, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Company(Base):
    __tablename__ = "company_details"

    id = Column(Integer, primary_key=True, index=True)
    company_no = Column(String(255), nullable=True)  