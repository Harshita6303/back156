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
    policy_type: str = Query("all", description="Filter by policy type (all, leave, hr, it, customer)"),
    db: Session = Depends(get_db)
):
    """
    Chat with the policy assistant to get information about policies.
    """
    valid_types = {"all", "leave", "hr", "it", "customer"}

    # 🔑 Normalize and validate
    policy_type = policy_type.strip().lower()
    if policy_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid policy type. Must be one of: {', '.join(sorted(valid_types))}"
        )

    if not message.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    try:
        # Pass None if 'all' so DB search isn't over-filtered
        effective_type = None if policy_type == "all" else policy_type

        result = await assistant_service.get_policy_chat_response(
            db=db,
            user_prompt=message.message,
            policy_type_filter=effective_type
        )

        # ✅ Log the policy type and chunks for debugging HR issue
        if policy_type == "hr":
            print("DEBUG-HR:", result.get("relevant_policies"), result.get("context_used"))

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


@router.get("/debug/collections")
async def debug_collections():
    """Debug endpoint to check vector DB collections"""
    try:
        stats = assistant_service.vector_db.get_collection_stats()
        return {
            "status": "success",
            "collection_stats": stats,
            "total_documents": stats.get("total_documents", 0)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/debug/test-embedding")
async def test_embedding(text: str = Query("test")):
    """Test embedding generation"""
    try:
        result = await assistant_service.llm_service.generate_query_embedding(text)
        return {
            "success": result.success,
            "embedding_length": len(result.embedding) if result.embedding else 0,
            "error": result.error
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Export router for main app
policy_assistant_router = router
