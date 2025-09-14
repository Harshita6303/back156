# services/llm_service.py
from __future__ import annotations
import os, asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional   
import PyPDF2
from io import BytesIO
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = (
    "You are a policy assistant. Answer ONLY using the provided policy context. "
    "If the answer is not in the context, say you couldn't find it."
)

# Reuse your existing return shape
@dataclass
class EmbeddingResult:
    success: bool
    embedding: Optional[List[float]] = None
    error: Optional[str] = None

class LLMService:
    def __init__(self) -> None:
        # System instruction is the Gemini equivalent of a ‘system’ role
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        self.embedding_model = "text-embedding-004"  # 768-dim vectors

    # --- Embeddings ---
    def _embed_sync(self, text: str) -> EmbeddingResult:
        try:
            resp = genai.embed_content(model=self.embedding_model, content=text)
            vec = resp["embedding"]
            return EmbeddingResult(success=True, embedding=vec)
        except Exception as e:
            return EmbeddingResult(success=False, error=str(e))

    async def generate_query_embedding(self, text: str) -> EmbeddingResult:
        # google-generativeai is sync; wrap in a thread to keep your async interface
        return await asyncio.to_thread(self._embed_sync, text)

    # --- Completions ---
    def _build_prompt(self, user_prompt: str, context_text: str) -> str:
        return (
            f"Question: {user_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer concisely:"
        )

    def _answer_sync(self, user_prompt: str, context_data: List[str]) -> Dict[str, Any]:
        try:
            context_text = "\n\n".join(context_data[:20]) if context_data else ""
            prompt = self._build_prompt(user_prompt, context_text)
            resp = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
            )
            return {"success": True, "response": (resp.text or "").strip()}
        except Exception as e:
            return {"success": False, "response": f"LLM error: {e}"}

    async def fetch_policy_information(
        self,
        user_prompt: str,
        context_data: List[str],
        relevant_metadata: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self._answer_sync, user_prompt, context_data)
    
    def chunk_pdf_content(self, pdf_content: bytes, chunk_size: int = 1000) -> List[str]:
        """Extract and chunk PDF content"""
        try:
            chunks = []
            pdf_file = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Split text into chunks
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"Error chunking PDF: {str(e)}")
            return []
        
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for text (alias for generate_query_embedding)"""
        return await self.generate_query_embedding(text)


