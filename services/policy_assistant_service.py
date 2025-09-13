from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import logging
import re

from services.vector_db_service import VectorDBService
from services.llm_service import LLMService
from services.policy_service import PolicyService
from models.policy_models import PolicyResponse

logger = logging.getLogger(__name__)


class PolicyAssistantService:
    def __init__(self):
        self.vector_db = VectorDBService()
        self.llm_service = LLMService()
        self.policy_service = PolicyService()

    async def get_policy_chat_response(
        self,
        db: Session,
        user_prompt: str,
        policy_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main method for handling policy chat assistance"""
        try:
            logger.info(f"Processing chat request: {user_prompt[:100]}...")

            # Step 1: Validate and extract policy information from prompt
            validation_result = await self._validate_and_extract_policy_info(user_prompt)

            # Step 2: Generate embedding for the user prompt
            query_embedding_result = await self.llm_service.generate_query_embedding(user_prompt)
            if not query_embedding_result.success:
                logger.error("Embedding failed: %s", query_embedding_result.error)
                # Fallback to DB-only answer (no vectors/LLM)
                fallback = await self._generate_fallback_response(db, user_prompt, policy_type_filter)
                return {
                    "response": fallback["response"],
                    "relevant_policies": fallback.get("policies", []),
                    "context_used": "Embedding failed; returned fallback results",
                    "success": True,
                    "validation_info": validation_result,
                }

            # Step 3: Search vector database for relevant content
            try:
                search_results = await self.vector_db.search(
                    query_embedding=query_embedding_result.embedding,
                    n_results=10,
                    policy_type_filter=policy_type_filter,
                )
            except Exception as e:
                logger.error("Vector search failed: %s", e)
                fallback = await self._generate_fallback_response(db, user_prompt, policy_type_filter)
                return {
                    "response": fallback["response"],
                    "relevant_policies": fallback.get("policies", []),
                    "context_used": "Vector search failed; returned fallback results",
                    "success": True,
                    "validation_info": validation_result,
                }

            # Step 4: Get relevant policies from database
            relevant_policy_ids = list(
                set(
                    [
                        meta.get("policy_id")
                        for meta in search_results.get("metadatas", [])
                        if meta.get("policy_id")
                    ]
                )
            )

            relevant_policies: List[PolicyResponse] = []
            for policy_id in relevant_policy_ids:
                policy = self.policy_service.get_policy(db, policy_id)
                if policy:
                    relevant_policies.append(
                        PolicyResponse(
                            id=policy.id,
                            name=policy.name,
                            type=policy.type.value,
                            description=policy.description,
                            effective_date=policy.effective_date,
                            document_path=policy.document_path,
                            created_at=policy.created_at,
                            updated_at=policy.updated_at,
                            download_url=self.policy_service.get_policy_download_url(policy),
                        )
                    )

            # Step 5: Generate LLM response using retrieved context
            if search_results.get("documents"):
                llm_response = await self.llm_service.fetch_policy_information(
                    user_prompt=user_prompt,
                    context_data=search_results["documents"],
                    relevant_metadata=search_results.get("metadatas", []),
                )

                return {
                    "response": llm_response["response"],
                    "relevant_policies": relevant_policies,
                    "context_used": f"Found {search_results.get('total_results', 0)} relevant document chunks",
                    "success": llm_response["success"],
                    "validation_info": validation_result,
                }

            # Fallback if no documents found
            fallback_response = await self._generate_fallback_response(db, user_prompt, policy_type_filter)
            return {
                "response": fallback_response["response"],
                "relevant_policies": fallback_response.get("policies", []),
                "context_used": "No specific document content found, using general policy information",
                "success": True,
                "validation_info": validation_result,
            }

        except Exception as e:
            logger.error(f"Error in policy chat response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
                "relevant_policies": [],
                "context_used": None,
                "success": False,
                "error": str(e),
            }

    async def _validate_and_extract_policy_info(self, user_prompt: str) -> Dict[str, Any]:
        """Validate user prompt and extract policy-related information"""
        try:
            validation_result: Dict[str, Any] = {
                "is_valid": True,
                "extracted_policy_names": [],
                "extracted_policy_types": [],
                "intent": "general_query",
                "confidence": "medium",
            }

            # Extract potential policy names
            policy_name_patterns = [
                r'policy\s+([A-Za-z0-9\s]+)',
                r'([A-Za-z0-9\s]+)\s+policy',
                r'"([^"]+)"',
                r"'([^']+)'",
            ]
            extracted_names: List[str] = []
            for pattern in policy_name_patterns:
                matches = re.findall(pattern, user_prompt, re.IGNORECASE)
                extracted_names.extend([m.strip() for m in matches if len(m.strip()) > 2])
            validation_result["extracted_policy_names"] = list(set(extracted_names))

            # Extract policy types
            policy_types = ["leave", "hr", "it", "customer"]
            extracted_types = [pt for pt in policy_types if pt in user_prompt.lower()]
            validation_result["extracted_policy_types"] = extracted_types

            # Determine intent
            lower = user_prompt.lower()
            if any(word in lower for word in ["show", "list", "all", "get"]):
                validation_result["intent"] = "list_policies"
            elif any(word in lower for word in ["details", "about", "what is", "explain"]):
                validation_result["intent"] = "get_policy_details"
            elif any(word in lower for word in ["find", "search", "look for"]):
                validation_result["intent"] = "search_policies"

            # Set confidence
            if extracted_names or extracted_types:
                validation_result["confidence"] = "high"
            elif "policy" in lower:
                validation_result["confidence"] = "medium"
            else:
                validation_result["confidence"] = "low"

            return validation_result

        except Exception as e:
            logger.error(f"Error in prompt validation: {str(e)}")
            return {
                "is_valid": True,
                "extracted_policy_names": [],
                "extracted_policy_types": [],
                "intent": "general_query",
                "confidence": "low",
            }

    async def _generate_fallback_response(
        self,
        db: Session,
        user_prompt: str,
        policy_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a fallback response when no specific documents are found"""
        try:
            policies = self.policy_service.get_policies(
                db=db,
                policy_type=policy_type_filter,
                limit=5,
            )

            policy_responses = [
                PolicyResponse(
                    id=policy.id,
                    name=policy.name,
                    type=policy.type.value,
                    description=policy.description,
                    effective_date=policy.effective_date,
                    document_path=policy.document_path,
                    created_at=policy.created_at,
                    updated_at=policy.updated_at,
                    download_url=self.policy_service.get_policy_download_url(policy),
                )
                for policy in policies
            ]

            if policies:
                response_text = f"I found {len(policies)} policies that might be relevant to your query. "
                if policy_type_filter:
                    response_text += f"These are {policy_type_filter} policies. "
                response_text += "Here are the available policies:\n\n"
                for policy in policies:
                    response_text += f"• **{policy.name}** ({policy.type.value.upper()}): {policy.description}\n"
                response_text += "\nFor more specific information, please ask about a particular policy or provide more details."
            else:
                response_text = f"I couldn't find any {policy_type_filter or ''} policies matching your query. "
                response_text += "Please try rephrasing your question or check if the policy name is correct."

            return {
                "response": response_text,
                "policies": policy_responses,
            }

        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return {
                "response": "I apologize, but I couldn't retrieve policy information at this time. Please try again later.",
                "policies": [],
            }

    async def get_policy_by_name_or_partial(
        self,
        db: Session,
        policy_name: str
    ) -> List[PolicyResponse]:
        """Search for policies by name (exact or partial match)"""
        try:
            policies = self.policy_service.get_policies(
                db=db,
                search=policy_name,
                limit=10,
            )

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
                    download_url=self.policy_service.get_policy_download_url(policy),
                )
                for policy in policies
            ]

        except Exception as e:
            logger.error(f"Error searching policies by name: {str(e)}")
            return []
