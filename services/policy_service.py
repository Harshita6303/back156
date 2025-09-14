from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
import os
import logging
from datetime import datetime
from models.policy_models import PolicyType


from models.policy_models import PolicyCreate, PolicyUpdate
from database.database import SessionLocal, Policy as DBPolicy  # ✅ Use SQLAlchemy model
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

class PolicyService:
    def __init__(self):
        self.upload_dir = "uploads/policies"
        self.vector_db = VectorDBService()
        self.llm_service = LLMService()
        os.makedirs(self.upload_dir, exist_ok=True)

    def get_db(self):
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def create_policy(
        self,
        db: Session,
        policy_data: PolicyCreate,
        file: Optional[UploadFile] = None
    ) -> DBPolicy:
        """Create a new policy with enhanced vector DB integration"""
        try:
            db_policy = DBPolicy(  # ✅ Use SQLAlchemy model
                name=policy_data.name,
                type=policy_data.type,
                description=policy_data.description,
                effective_date=policy_data.effective_date,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(db_policy)
            db.commit()
            db.refresh(db_policy)

            if file:
                await self._process_policy_document(db, db_policy, file)

            logger.info(f"Successfully created policy: {db_policy.name}")
            return db_policy

        except Exception as e:
            db.rollback()
            logger.error(f"Error creating policy: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating policy: {str(e)}")

    async def _process_policy_document(self, db: Session, policy: DBPolicy, file: UploadFile):
        """Process uploaded policy document: save file, chunk content, generate embeddings, and store in vector DB"""
        try:
            file_path = os.path.join(self.upload_dir, f"policy_{policy.id}_{file.filename}")
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)

            policy.document_path = file_path
            db.commit()

            if file.filename.lower().endswith('.pdf'):
                chunks = self.llm_service.chunk_pdf_content(content)
                if not chunks:
                    logger.warning(f"No content extracted from PDF for policy {policy.id}")
                    return

                embeddings = []
                successful_chunks = []
                for i, chunk in enumerate(chunks):
                    embedding_result = await self.llm_service.generate_embedding(chunk)
                    if embedding_result.success:
                        embeddings.append(embedding_result.embedding)
                        successful_chunks.append(chunk)
                    else:
                        logger.warning(f"Failed to generate embedding for chunk {i} of policy {policy.id}")

                metadata = {
                    "policy_id": policy.id,
                    "policy_name": policy.name,
                    "policy_type": policy.type.value,
                    "effective_date": policy.effective_date.isoformat(),
                    "document_path": file_path,
                    "created_at": policy.created_at.isoformat()
                }

                if successful_chunks and embeddings:
                    success = await self.vector_db.add_document(
                        document_chunks=successful_chunks,
                        embeddings=embeddings,
                        policy_id=policy.id,
                        metadata=metadata
                    )
                    if success:
                        logger.info(f"Successfully added {len(successful_chunks)} chunks to vector DB for policy {policy.id}")
                    else:
                        logger.error(f"Failed to add policy {policy.id} to vector DB")

        except Exception as e:
            logger.error(f"Error processing policy document: {str(e)}")

    def get_policies(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        policy_type: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[DBPolicy]:
        """Get policies with optional filtering"""
        try:
            query = db.query(DBPolicy)
            if policy_type:
    # normalize incoming filter to the enum value (e.g., "hr")
             normalized = PolicyType(policy_type.lower()).value if isinstance(policy_type, str) else policy_type.value
             query = query.filter(DBPolicy.type == normalized)

            
            if search:
                query = query.filter(
                    DBPolicy.name.contains(search) | DBPolicy.description.contains(search)
                )
            policies = query.offset(skip).limit(limit).all()
            logger.info(f"Retrieved {len(policies)} policies")
            return policies

        except Exception as e:
            logger.error(f"Error retrieving policies: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving policies: {str(e)}")

    def get_policy(self, db: Session, policy_id: int) -> Optional[DBPolicy]:
        """Get a specific policy by ID"""
        try:
            policy = db.query(DBPolicy).filter(DBPolicy.id == policy_id).first()
            if policy:
                logger.info(f"Retrieved policy: {policy.name}")
            return policy

        except Exception as e:
            logger.error(f"Error retrieving policy {policy_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving policy: {str(e)}")

    async def update_policy(
        self,
        db: Session,
        policy_id: int,
        policy_update: PolicyUpdate,
        file: Optional[UploadFile] = None
    ) -> Optional[DBPolicy]:
        """Update an existing policy"""
        try:
            policy = db.query(DBPolicy).filter(DBPolicy.id == policy_id).first()
            if not policy:
                return None

            update_data = policy_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if value is not None:
                    setattr(policy, field, value)
            policy.updated_at = datetime.utcnow()

            if file:
                if policy.document_path and os.path.exists(policy.document_path):
                    os.remove(policy.document_path)
                await self.vector_db.delete_document(policy_id)
                await self._process_policy_document(db, policy, file)

            db.commit()
            logger.info(f"Successfully updated policy: {policy.name}")
            return policy

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating policy {policy_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating policy: {str(e)}")

    async def delete_policy(self, db: Session, policy_id: int) -> bool:
        """Delete a policy"""
        try:
            policy = db.query(DBPolicy).filter(DBPolicy.id == policy_id).first()
            if not policy:
                return False

            if policy.document_path and os.path.exists(policy.document_path):
                os.remove(policy.document_path)

            await self.vector_db.delete_document(policy_id)
            db.delete(policy)
            db.commit()
            logger.info(f"Successfully deleted policy: {policy.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting policy {policy_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting policy: {str(e)}")

    def get_policy_download_url(self, policy: DBPolicy) -> Optional[str]:
        """Generate download URL for policy document"""
        if policy.document_path and os.path.exists(policy.document_path):
            return f"/api/v1/policies/{policy.id}/download"
        return None
