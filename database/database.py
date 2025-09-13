from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum
from typing import Generator
import os
from sqlalchemy.orm import sessionmaker, Session

# Database URL - using SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./policy_management.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

# Policy Type Enum
class PolicyTypeEnum(str, enum.Enum):  # Inherit from str for better compatibility with Pydantic
    LEAVE = "leave"
    HR = "hr"
    IT = "it"
    CUSTOMER = "customer"

# Database Model
class Policy(Base):
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(Enum(PolicyTypeEnum), nullable=False, index=True)
    description = Column(Text, nullable=False)
    effective_date = Column(DateTime, nullable=False)
    document_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Dependency to get DB session
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)

# Export the Policy model to avoid circular imports
__all__ = ["Base", "engine", "SessionLocal", "get_db", "init_db", "Policy"]