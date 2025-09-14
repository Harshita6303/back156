import os
import logging
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class VectorDBService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "policy_documents"

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        if self.client is not None:
            return  # Already initialized

        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(path="./chroma_db")

            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Policy documents collection"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            logger.info("ChromaDB client initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            print(f"⚠️ ChromaDB initialization failed: {str(e)}")
            print("The system will continue without vector search capabilities")

    async def add_document(
        self,
        document_chunks: List[str],
        embeddings: List[List[float]],
        policy_id: int,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add document chunks with embeddings to the vector database"""
        try:
            if self.client is None:
                self._initialize_client()
            if self.collection is None:
                logger.warning("Vector database not available, skipping document addition")
                return False

            # ✅ Normalize metadata.policy_type once up-front (case/trim safe)
            base_meta = dict(metadata or {})
            base_meta["policy_type"] = str(base_meta.get("policy_type", "")).strip().lower()

            chunk_ids = [f"policy_{policy_id}_chunk_{i}" for i in range(len(document_chunks))]

            chunk_metadata = []
            for i, chunk in enumerate(document_chunks):
                chunk_meta = {
                    **base_meta,  # ✅ normalized policy_type included on every chunk
                    "policy_id": policy_id,
                    "chunk_index": i,
                    "total_chunks": len(document_chunks)
                }
                chunk_metadata.append(chunk_meta)

            self.collection.add(
                embeddings=embeddings,
                documents=document_chunks,
                metadatas=chunk_metadata,
                ids=chunk_ids
            )

            logger.info(f"Successfully added {len(document_chunks)} chunks for policy {policy_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding document to vector DB: {str(e)}")
            print(f"⚠️ Failed to add document to vector database: {str(e)}")
            return False

    async def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        policy_type_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for similar documents using vector similarity"""
        try:
            if self.client is None:
                self._initialize_client()
            if self.collection is None:
                logger.warning("Vector database not available, returning empty results")
                return {
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                    "total_results": 0
                }

            # Validate embedding dimensions
            if not query_embedding or len(query_embedding) != 768:  # Gemini embedding dimension
                logger.warning(f"Invalid embedding dimension: {len(query_embedding) if query_embedding else 'None'}")
                return {
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                    "total_results": 0
                }

            where_clause: Dict[str, Any] = {}
            if policy_type_filter:
                where_clause["policy_type"] = str(policy_type_filter).strip().lower()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = {
                "documents": results["documents"][0] if results.get("documents") else [],
                "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
                "distances": results["distances"][0] if results.get("distances") else [],
                "total_results": len(results["documents"][0]) if results.get("documents") else 0
            }

            logger.info(f"Vector search returned {formatted_results['total_results']} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            print(f"⚠️ Vector search failed: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "total_results": 0
            }

    async def delete_document(self, policy_id: int) -> bool:
        """Delete all chunks for a specific policy"""
        try:
            if self.client is None:
                self._initialize_client()
            if self.collection is None:
                logger.warning("Vector database not available, skipping document deletion")
                return True  # Don't block policy deletion

            results = self.collection.get(
                where={"policy_id": policy_id},
                include=["metadatas"]
            )

            if results.get("ids"):
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for policy {policy_id}")
            else:
                logger.info(f"No chunks found for policy {policy_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting document from vector DB: {str(e)}")
            print(f"⚠️ Failed to delete from vector database: {str(e)}")
            return True  # Don't block policy deletion

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            if self.client is None:
                self._initialize_client()
            if self.collection is None:
                return {"total_documents": 0, "collection_name": self.collection_name}

            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"total_documents": 0, "collection_name": self.collection_name}
