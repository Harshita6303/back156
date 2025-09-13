from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum

class PolicyType(str, Enum):
    LEAVE = "leave"
    HR = "hr"
    IT = "it"
    CUSTOMER = "customer"

class PolicyBase(BaseModel):
    name: str
    type: PolicyType
    description: str
    effective_date: datetime

class PolicyCreate(PolicyBase):
    pass

class PolicyUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[PolicyType] = None
    description: Optional[str] = None
    effective_date: Optional[datetime] = None
    document_path: Optional[str] = None

class Policy(PolicyBase):
    id: int
    document_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True  # ✅ Updated for Pydantic v2



class PolicyResponse(BaseModel):
    id: int
    name: str
    type: str
    description: str
    effective_date: datetime
    document_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    download_url: Optional[str] = None

    class Config:
        from_attributes = True  # ✅ Required for SQLAlchemy compatibility


class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    relevant_policies: Optional[List[PolicyResponse]] = None
    context_used: Optional[str] = None