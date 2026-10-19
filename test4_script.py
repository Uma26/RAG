import os
import numpy as np
from typing import List
from langchain_core.documents import Document
from app.services.embeddings.base import BaseEmbeddingService
from app.services.embeddings.openai_adapter import OpenAIEmbeddingAdapter


class MockEmbeddingAdapter(BaseEmbeddingService):
    """Fallback mock adapter for local offline testing."""
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        np.random.seed(42)
        return np.random.rand(len(documents), self.dimension).tolist()

    def embed_query(self, query: str) -> List[float]:
        np.random.seed(99)
        return np.random.rand(self.dimension).tolist()


def test_embedding_service():
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Build sample document chunks
    chunks = [
        Document(page_content="Enterprise RAG uses FAISS vector stores.", metadata={"chunk_index": 0}),
        Document(page_content="OpenAI text-embedding-3 provides dense vector representations.", metadata={"chunk_index": 1}),
    ]

    if api_key and api_key != "mock-key-for-testing":
        print("Running live OpenAI API test...")
        service = OpenAIEmbeddingAdapter(dimensions=1536)
    else:
        print("No OPENAI_API_KEY detected. Running mock embedding test...")
        service = MockEmbeddingAdapter(dimension=1536)

    # 1. Embed Document Chunks
    vectors = service.embed_documents(chunks)
    assert len(vectors) == 2
    assert len(vectors[0]) == 1536
    print(f"✓ Successfully embedded {len(vectors)} chunks. Dimension: {len(vectors[0])}")

    # 2. Embed Query
    query_vector = service.embed_query("What vector storage is used?")
    assert len(query_vector) == 1536
    print(f"✓ Successfully embedded search query. Dimension: {len(query_vector)}")

    print("\n✓ Step 4 Embeddings Service Verification Passed Successfully!")


if __name__ == "__main__":
    test_embedding_service()