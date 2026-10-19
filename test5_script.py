import os
import shutil
import tempfile
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.services.storage.faiss_repository import FAISSVectorRepository


class MockLangChainEmbeddings(Embeddings):
    """Mock Embeddings conforming to LangChain Embeddings interface for offline testing."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        np.random.seed(42)
        return [np.random.rand(1536).tolist() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        np.random.seed(42)
        return np.random.rand(1536).tolist()


def test_faiss_repository():
    mock_embeddings = MockLangChainEmbeddings()
    repo = FAISSVectorRepository(embedding_service=mock_embeddings)

    # 1. Create sample test chunks
    doc1 = Document(
        page_content="FastAPI is an async Python web framework.",
        metadata={"file_type": "pdf", "source": "fastapi_manual.pdf"}
    )
    doc2 = Document(
        page_content="FAISS provides fast dense vector similarity search.",
        metadata={"file_type": "txt", "source": "faiss_docs.txt"}
    )

    # 2. Add Documents
    ids = repo.add_documents([doc1, doc2])
    assert len(ids) == 2
    print(f"✓ Added {len(ids)} documents to FAISS vector repository.")

    # 3. Similarity Search with Metadata Filtering
    filtered_results = repo.similarity_search(
        query="Vector search algorithm",
        k=2,
        filter={"file_type": "txt"}
    )
    assert len(filtered_results) == 1
    assert filtered_results[0].metadata["file_type"] == "txt"
    print(f"✓ Metadata filter test passed: Retrieved '{filtered_results[0].page_content}'")

    # 4. Search with Distance Scores
    results_with_scores = repo.similarity_search_with_score(query="Python web framework", k=1)
    assert len(results_with_scores) == 1
    doc, score = results_with_scores[0]
    print(f"✓ Score search test passed. Document: '{doc.page_content[:30]}...' Score: {score:.4f}")

    # 5. Serialization and Disk Loading Test
    temp_dir = tempfile.mkdtemp()
    try:
        repo.save_index(folder_path=temp_dir, index_name="enterprise_index")
        assert os.path.exists(os.path.join(temp_dir, "enterprise_index.faiss"))
        assert os.path.exists(os.path.join(temp_dir, "enterprise_index.pkl"))
        print("✓ Index persistence test passed (saved to disk).")

        # Load into a new repository instance
        new_repo = FAISSVectorRepository(embedding_service=mock_embeddings)
        new_repo.load_index(folder_path=temp_dir, index_name="enterprise_index")
        
        reloaded_results = new_repo.similarity_search(query="Python web framework", k=1)
        assert len(reloaded_results) == 1
        print("✓ Index reload test passed (successfully restored state from disk).")

    finally:
        shutil.rmtree(temp_dir)

    print("\n✓ Step 5 FAISS Vector Repository Verification Passed Successfully!")


if __name__ == "__main__":
    test_faiss_repository()