import os
import io
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

# Set mock key for offline execution
os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

from app.main import app, app_state
from app.services.embeddings.base import BaseEmbeddingService
from app.services.storage.faiss_repository import FAISSVectorRepository
from app.services.pipeline_orchestrator import RAGPipelineOrchestrator
import numpy as np


class MockEmbeddings(Embeddings, BaseEmbeddingService):
    """Mock Embedding class compatible with both interfaces for testing."""
    def embed_documents(self, texts):
        np.random.seed(42)
        return [np.random.rand(1536).tolist() for _ in texts]

    def embed_query(self, text):
        np.random.seed(42)
        return np.random.rand(1536).tolist()


def setup_mock_orchestrator():
    mock_emb = MockEmbeddings()
    vector_repo = FAISSVectorRepository(embedding_service=mock_emb)
    app_state.orchestrator = RAGPipelineOrchestrator(
        embedding_service=mock_emb,
        vector_repository=vector_repo
    )


def test_e2e_fastapi_endpoints():
    setup_mock_orchestrator()

    with TestClient(app) as client:
        # 1. Health Check Test
        res_health = client.get("/health")
        assert res_health.status_code == 200
        print("✓ Health Check Endpoint Passed.")

        # 2. Ingest TXT File
        sample_file_content = (
            "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python. "
            "It utilizes standard Python type hints and Pydantic schemas for data validation."
        )
        file_payload = {
            "file": ("fastapi_intro.txt", io.BytesIO(sample_file_content.encode("utf-8")), "text/plain")
        }

        res_ingest = client.post("/api/v1/ingest/file", files=file_payload)
        assert res_ingest.status_code == 201
        ingest_data = res_ingest.json()
        assert ingest_data["status"] == "success"
        assert ingest_data["total_chunks_created"] > 0
        print(f"✓ File Ingestion Endpoint Passed ({ingest_data['total_chunks_created']} chunks indexed).")

        # 3. Vector Search
        search_payload = {
            "query": "What is FastAPI framework?",
            "top_k": 2
        }
        res_search = client.post("/api/v1/search", json=search_payload)
        assert res_search.status_code == 200
        search_data = res_search.json()
        assert search_data["total_results"] > 0
        assert "FastAPI" in search_data["results"][0]["content"]
        print(f"✓ Search Endpoint Passed. Retrieved top matching chunk snippet.")

    print("\n All End-to-End FastAPI Integration Tests Passed Successfully!")


if __name__ == "__main__":
    test_e2e_fastapi_endpoints()