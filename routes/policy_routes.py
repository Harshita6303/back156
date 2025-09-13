from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from models.policy_models import PolicyCreate, PolicyUpdate, PolicyResponse, PolicyType
from services.policy_service import PolicyService
from database.database import get_db
from fastapi.responses import FileResponse
import os

router = APIRouter()
policy_service = PolicyService()

@router.post("/", response_model=PolicyResponse, status_code=201)
async def create_policy(
    name: str = Form(...),
    type: str = Form(...),
    description: str = Form(...),
    effective_date: str = Form(...),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Create a new policy with optional document upload"""
    try:
        policy_type = PolicyType(type.lower())
        effective_date_obj = datetime.fromisoformat(effective_date.replace('Z', '+00:00'))

        policy_data = PolicyCreate(
            name=name,
            type=policy_type,
            description=description,
            effective_date=effective_date_obj
        )

        policy = await policy_service.create_policy(db, policy_data, file)


        return PolicyResponse(
            id=policy.id,
            name=policy.name,
            type=policy.type.value,
            description=policy.description,
            effective_date=policy.effective_date,
            document_path=policy.document_path,
            created_at=policy.created_at,
            updated_at=policy.updated_at,
            download_url=policy_service.get_policy_download_url(policy)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating policy: {str(e)}")

@router.get("/", response_model=List[PolicyResponse])
async def get_policies(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    policy_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get all policies with optional filtering"""
    try:
        policies = policy_service.get_policies(db, skip, limit, policy_type, search)
        return [
            PolicyResponse(
                id=policy.id,
                name=policy.name,
                type=policy.type.value,
                description=policy.description,
                effective_date=policy.effective_date,
                document_path=policy.document_path,
                created_at=policy.created_at,
                updated_at=policy.updated_at,
                download_url=policy_service.get_policy_download_url(policy)
            )
            for policy in policies
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(policy_id: int, db: Session = Depends(get_db)):
    """Get a specific policy by ID"""
    policy = policy_service.get_policy(db, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return PolicyResponse(
        id=policy.id,
        name=policy.name,
        type=policy.type.value,
        description=policy.description,
        effective_date=policy.effective_date,
        document_path=policy.document_path,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
        download_url=policy_service.get_policy_download_url(policy)
    )

@router.put("/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: int,
    name: Optional[str] = Form(None),
    type: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    effective_date: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Update an existing policy"""
    try:
        update_data = {}
        if name:
            update_data["name"] = name
        if type:
            update_data["type"] = PolicyType(type.lower())
        if description:
            update_data["description"] = description
        if effective_date:
            update_data["effective_date"] = datetime.fromisoformat(effective_date.replace('Z', '+00:00'))

        policy_update = PolicyUpdate(**update_data)
        policy = await policy_service.update_policy(db, policy_id, policy_update, file)
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")

        return PolicyResponse(
            id=policy.id,
            name=policy.name,
            type=policy.type.value,
            description=policy.description,
            effective_date=policy.effective_date,
            document_path=policy.document_path,
            created_at=policy.created_at,
            updated_at=policy.updated_at,
            download_url=policy_service.get_policy_download_url(policy)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating policy: {str(e)}")

@router.delete("/{policy_id}")
async def delete_policy(policy_id: int, db: Session = Depends(get_db)):
    """Delete a policy"""
    success = await policy_service.delete_policy(db, policy_id)
    if not success:
        raise HTTPException(status_code=404, detail="Policy not found")
    return {"message": "Policy deleted successfully"}

@router.get("/{policy_id}/download")
async def download_policy_document(policy_id: int, db: Session = Depends(get_db)):
    """Download policy document"""
    policy = policy_service.get_policy(db, policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    if not policy.document_path or not os.path.exists(policy.document_path):
        raise HTTPException(status_code=404, detail="Policy document not found")

    filename = os.path.basename(policy.document_path)
    return FileResponse(
        path=policy.document_path,
        filename=filename,
        media_type='application/octet-stream'
    )

# ✅ Export router for FastAPI main app
policy_router = router