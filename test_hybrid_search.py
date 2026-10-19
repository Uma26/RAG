import os
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.storage.hybrid_repository import HybridVectorRepository


class MockLangChainEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Deterministic dummy vectors based on text hash
        return [np.random.RandomState(hash(t) % (2**32)).rand(1536).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return np.random.RandomState(hash(text) % (2**32)).rand(1536).tolist()


def test_hybrid_search_fusion():
    mock_emb = MockLangChainEmbeddings()
    repo = HybridVectorRepository(embedding_service=mock_emb)

    docs = [
        Document(page_content="System error ERR_404_DATABASE_TIMEOUT occurred during migration.", metadata={"source": "logs.txt"}),
        Document(page_content="Relational databases require indexed primary keys for speed.", metadata={"source": "db_guide.txt"}),
        Document(page_content="Vector databases store high dimensional embeddings.", metadata={"source": "rag_guide.txt"}),
    ]

    repo.add_documents(docs)

    # 1. Test Sparse Search (exact keycode hit)
    sparse_matches = repo.sparse_search(query="ERR_404_DATABASE_TIMEOUT", top_k=1)
    assert len(sparse_matches) == 1
    assert "ERR_404_DATABASE_TIMEOUT" in sparse_matches[0][0].page_content
    print(f"✓ BM25 Sparse Search caught exact keyword match: '{sparse_matches[0][0].page_content}'")

    # 2. Test Hybrid Search Fusion (RRF)
    hybrid_results = repo.hybrid_search(query="ERR_404_DATABASE_TIMEOUT database", k=2)
    assert len(hybrid_results) > 0
    top_doc, rrf_score = hybrid_results[0]
    assert rrf_score > 0.0
    print(f"✓ RRF Hybrid Search passed. Top doc: '{top_doc.page_content[:45]}...' | RRF Score: {rrf_score:.5f}")

    print("\n✓ Step 6 Hybrid Search Integration Tests Passed Successfully!")


if __name__ == "__main__":
    test_hybrid_search_fusion()