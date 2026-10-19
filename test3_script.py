from langchain_core.documents import Document
from app.services.chunking.engine import ChunkingEngine
from app.services.chunking.strategies.recursive_strategy import RecursiveChunkingStrategy
from app.services.chunking.strategies.token_strategy import TokenChunkingStrategy

def test_chunking_engine():
    # Long text (> 500 chars) to test chunking
    long_text = (
        "Enterprise RAG systems require high modularity and robust architectural design patterns. "
        "By utilizing the Strategy Pattern for chunking, we can swap algorithms depending on data types. "
        "For instance, dense PDF manuals benefit from recursive character splitting, while code repositories "
        "or technical documentation might require token-bound chunking. "
        "Each chunk must maintain complete metadata lineage, including parent file details, chunk index, "
        "and positional information to allow precise retrieval and accurate context assembly for the LLM."
    )

    doc = Document(
        page_content=long_text,
        metadata={"source": "architecture_guide.pdf", "file_type": "pdf", "author": "Engineering Lead"}
    )

    # 1. Test Default Strategy (Recursive with chunk_size=200, overlap=20)
    engine = ChunkingEngine(strategy=RecursiveChunkingStrategy(chunk_size=200, chunk_overlap=20))
    chunks = engine.execute([doc])

    print(f"Total Chunks Generated: {len(chunks)}")
    assert len(chunks) > 1, "Document should be split into multiple chunks"

    # Verify Chunk 0
    first_chunk = chunks[0]
    print("\n--- CHUNK 0 CONTENT ---")
    print(first_chunk.page_content)
    print("\n--- CHUNK 0 METADATA ---")
    print(first_chunk.metadata)

    # Metadata Assertions
    assert first_chunk.metadata["source"] == "architecture_guide.pdf"
    assert first_chunk.metadata["file_type"] == "pdf"
    assert first_chunk.metadata["chunk_index"] == 0
    assert first_chunk.metadata["total_chunks"] == len(chunks)
    assert first_chunk.metadata["chunk_size_setting"] == 200

    # 2. Test Runtime Strategy Switch to Token Chunking
    engine.set_strategy(TokenChunkingStrategy(chunk_size=50, chunk_overlap=5))
    token_chunks = engine.execute([doc])
    print(f"\nToken Strategy Chunks Generated: {len(token_chunks)}")
    assert len(token_chunks) > 0

    print("\n✓ Step 3 Chunking Engine Verification Passed Successfully!")

if __name__ == "__main__":
    test_chunking_engine()