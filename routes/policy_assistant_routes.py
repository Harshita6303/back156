from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from models.policy_models import ChatMessage, ChatResponse
from services.policy_assistant_service import PolicyAssistantService
from database.database import get_db

router = APIRouter()
assistant_service = PolicyAssistantService()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_policy_assistant(
    message: ChatMessage,
    policy_type: Optional[str] = Query(None, description="Filter by policy type (leave, hr, it, customer"),
    db: Session = Depends(get_db)
):
    """
    Chat with the policy assistant to get information about policies.
    This endpoint allows users to ask questions about policies and get AI-powered responses
    based on the policy documents stored in the system.
    """
    valid_types = {"leave", "hr", "it", "customer"}

    # Validate policy type
    if policy_type:
        policy_type = policy_type.lower()
        if policy_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid policy type. Must be one of: {', '.join(valid_types)}"
            )

    # Validate message
    if not message.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    try:
        result = await assistant_service.get_policy_chat_response(
            db=db,
            user_prompt=message.message,
            policy_type_filter=policy_type
        )

        # The service now always returns a usable answer (fallback if needed),
        # so don't raise 500 just because LLM/vector step failed.
        return ChatResponse(
            response=result["response"],
            relevant_policies=result.get("relevant_policies", []),
            context_used=result.get("context_used", "")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/policies")
async def get_policy_by_query(
    query: str = Query(..., description="Search query for policy name"),
    db: Session = Depends(get_db)
):
    """
    Search for policies by name or partial match.
    This endpoint helps users find specific policies by searching through policy names.
    """
    try:
        policies = await assistant_service.get_policy_by_name_or_partial(
            db=db,
            policy_name=query
        )

        return {
            "query": query,
            "results": policies,
            "total_found": len(policies)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def assistant_health_check():
    """Health check for the policy assistant service."""
    try:
        stats = assistant_service.vector_db.get_collection_stats()

        return {
            "status": "healthy",
            "vector_db_status": "connected",
            "total_documents": stats.get("total_documents", 0),
            "services": {
                "vector_db": "operational",
                "llm_service": "operational",
                "policy_service": "operational"
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "vector_db": "error",
                "llm_service": "unknown",
                "policy_service": "unknown"
            }
        }

# Export router for main app
policy_assistant_router = router
