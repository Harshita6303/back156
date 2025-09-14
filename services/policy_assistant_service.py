from typing import List, Dict, Any, Optional, Iterable
from sqlalchemy.orm import Session
import logging
import re
from services.vector_db_service import VectorDBService
from services.llm_service import LLMService
from services.policy_service import PolicyService
from models.policy_models import PolicyResponse

logger = logging.getLogger(__name__)

# Patterns that indicate an LLM refused / failed to answer despite having context
_NO_ANSWER_PAT = re.compile(
    r"(?:can't|cannot|could not)\s+find|no (?:answer|information)|"
    r"not\s+(?:available|present)|insufficient\s+(?:context|information)|"
    r"i\s+don't\s+have|outside\s+the\s+provided\s+context",
    re.I,
)


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

            # If user chose "All" but their query clearly names exactly one type,
            # auto-narrow to that single detected type.
            effective_type_filter = policy_type_filter if policy_type_filter and policy_type_filter != "all" else None
            detected_types = validation_result.get("extracted_policy_types") or []
            if effective_type_filter is None and len(detected_types) == 1:
                effective_type_filter = detected_types[0]

            # Step 2: Generate embedding for the user prompt
            query_embedding_result = await self.llm_service.generate_query_embedding(user_prompt)
            if not query_embedding_result.success:
                logger.error("Embedding failed: %s", query_embedding_result.error)
                fallback = await self._generate_fallback_response(db, user_prompt, effective_type_filter)
                return {
                    "response": fallback["response"],
                    "relevant_policies": fallback.get("policies", []),
                    "context_used": "Embedding failed; returned fallback results",
                    "success": True,
                    "validation_info": validation_result,
                }

            # Step 3: Search vector database
            try:
                search_results = await self.vector_db.search(
                    query_embedding=query_embedding_result.embedding,
                    n_results=10,
                    policy_type_filter=effective_type_filter,
                )
            except Exception as e:
                logger.error("Vector search failed: %s", e)
                fallback = await self._generate_fallback_response(db, user_prompt, effective_type_filter)
                return {
                    "response": fallback["response"],
                    "relevant_policies": fallback.get("policies", []),
                    "context_used": "Vector search failed; returned fallback results",
                    "success": True,
                    "validation_info": validation_result,
                }

            # Step 4: Collect relevant policies
            docs = search_results.get("documents") or []
            metas = search_results.get("metadatas") or []

            # Post-filter by type (robust, case/trim-safe) in case vector where-clause didn't apply
            def _norm(x): return (x or "").strip().lower()

            if effective_type_filter:
                keep = [i for i, m in enumerate(metas) if _norm((m or {}).get("policy_type")) == _norm(effective_type_filter)]
                if keep:
                    docs = [docs[i] for i in keep]
                    metas = [metas[i] for i in keep]
                else:
                    docs, metas = [], []

            # Build relevant policy responses (unique ids from metas)
            rel_ids = list({
                (m or {}).get("policy_id")
                for m in metas
                if (m or {}).get("policy_id") is not None
            })

            relevant_policies: List[PolicyResponse] = []
            context_texts: List[str] = []

            for pid in rel_ids:
                pol = self.policy_service.get_policy(db, pid)
                if pol:
                    relevant_policies.append(
                        PolicyResponse(
                            id=pol.id,
                            name=pol.name,
                            type=pol.type if isinstance(pol.type, str) else pol.type.value,
                            description=pol.description,
                            effective_date=pol.effective_date,
                            document_path=pol.document_path,
                            created_at=pol.created_at,     # keep for Pydantic model
                            updated_at=pol.updated_at,     # keep for Pydantic model
                            download_url=self.policy_service.get_policy_download_url(pol),
                        )
                    )
                    if pol.description:
                        context_texts.append(
                            f"{pol.name} ({pol.type if isinstance(pol.type, str) else pol.type.value}): {pol.description}"
                        )

            # Step 5: Generate LLM response with retrieved context
            flat_docs = self._flatten_docs(docs) if docs else []
            total_sections = len(flat_docs)

            # Combine policy descriptions + chunk texts
            context_data = "\n\n".join(context_texts + flat_docs)

            if context_data.strip():
                # Use the method that exists in LLMService
                llm_payload = await self.llm_service.fetch_policy_information(
                    user_prompt=user_prompt,
                    context_data=[context_data],   # pass as list to keep API consistent
                    relevant_metadata=metas,
                )
                answer_text = (llm_payload or {}).get("response", "") or ""
                success_flag = bool((llm_payload or {}).get("success", False))

                # Deterministic synthesis if LLM refused to answer even with chunks
                if self._looks_like_no_answer(answer_text) and flat_docs:
                    synthesized = self._synthesize_from_chunks(user_prompt, flat_docs, metas)
                    if synthesized:
                        answer_text, success_flag = synthesized, True

                return {
                    "response": answer_text or "I couldn’t find an answer in the current policies.",
                    "relevant_policies": relevant_policies,
                    "context_used": f"Found {total_sections} relevant policy sections"
                                    + (f" (type={effective_type_filter})" if effective_type_filter else ""),
                    "success": success_flag,
                    "validation_info": validation_result,
                }

            # Fallback if no usable context
            fallback = await self._generate_fallback_response(db, user_prompt, effective_type_filter)
            return {
                "response": fallback["response"],
                "relevant_policies": fallback.get("policies", []),
                "context_used": "No matching policy sections after filtering; used fallback.",
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

            # Extract potential policy NAMES (use the intended patterns)
            name_patterns = [
                r'policy\s+([A-Za-z0-9\s]+)',
                r'([A-Za-z0-9\s]+)\s+policy',
                r'"([^"]+)"',
                r"'([^']+)'",
            ]
            extracted_names: List[str] = []
            for pat in name_patterns:
                matches = re.findall(pat, user_prompt, flags=re.IGNORECASE)
                extracted_names.extend([m.strip() for m in matches if len(m.strip()) > 2])
            validation_result["extracted_policy_names"] = list(dict.fromkeys(extracted_names))

            # Extract policy TYPES (whole-word, case-insensitive, de-duped)
            type_matches = re.findall(r"\b(hr|it|leave|customer)\b", user_prompt, flags=re.IGNORECASE)
            extracted_types = [t.lower() for t in type_matches]
            validation_result["extracted_policy_types"] = list(dict.fromkeys(extracted_types))

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
                    id=p.id,
                    name=p.name,
                    type=p.type if isinstance(p.type, str) else p.type.value,
                    description=p.description,
                    effective_date=p.effective_date,
                    document_path=p.document_path,
                    created_at=p.created_at,
                    updated_at=p.updated_at,
                    download_url=self.policy_service.get_policy_download_url(p),
                )
                for p in policies
            ]

            if policies:
                response_text = f"I found {len(policies)} policies that might be relevant. "
                if policy_type_filter:
                    response_text += f"These are {policy_type_filter} policies. "
                response_text += "Here are the available policies:\n\n"
                for p in policies:
                    t = p.type if isinstance(p.type, str) else p.type.value
                    response_text += f"• **{p.name}** ({t}): {p.description}\n"
                response_text += "\nFor more details, ask about a specific policy."
            else:
                response_text = f"I couldn't find any {policy_type_filter or ''} policies matching your query."

            return {"response": response_text, "policies": policy_responses}

        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return {
                "response": "I couldn’t retrieve policy information at this time. Please try again later.",
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
                    id=p.id,
                    name=p.name,
                    type=p.type if isinstance(p.type, str) else p.type.value,
                    description=p.description,
                    effective_date=p.effective_date,
                    document_path=p.document_path,
                    created_at=p.created_at,
                    updated_at=p.updated_at,
                    download_url=self.policy_service.get_policy_download_url(p),
                )
                for p in policies
            ]

        except Exception as e:
            logger.error(f"Error searching policies by name: {str(e)}")
            return []

    # ------------------------
    # Helpers
    # ------------------------
    def _looks_like_no_answer(self, text: Optional[str]) -> bool:
        if not text:
            return True
        return bool(_NO_ANSWER_PAT.search(text))

    def _flatten_docs(self, docs: Iterable[Any]) -> List[str]:
        """Flatten nested doc lists into a simple list of strings"""
        flat: List[str] = []
        for d in docs:
            if isinstance(d, (list, tuple)):
                for x in d:
                    if isinstance(x, str):
                        flat.append(x)
            elif isinstance(d, str):
                flat.append(d)
        return flat

    def _synthesize_from_chunks(
        self,
        user_prompt: str,
        documents: List[Any],
        metadatas: List[Dict[str, Any]]
    ) -> str:
        """Synthesize a deterministic summary if LLM fails"""
        flat_docs = self._flatten_docs(documents)
        if not flat_docs:
            return ""

        bullets: List[str] = []
        for idx, txt in enumerate(flat_docs[:3]):
            snippet = txt.strip().replace("\n", " ")
            first_sentence = re.split(r"(?<=[.!?])\s+", snippet, maxsplit=1)[0]
            source = ""
            if idx < len(metadatas):
                meta = metadatas[idx] or {}
                title = meta.get("policy_name") or meta.get("title") or meta.get("policy_id")
                if title:
                    source = f" — *{title}*"
            bullets.append(f"- {first_sentence}{source}")

        if not bullets:
            return ""

        return (
            "Here’s what the policy content indicates based on retrieved sections:\n\n"
            + "\n".join(bullets)
            + "\n\nIf you need a specific clause, ask me the section or a more direct question."
        )
