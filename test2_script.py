from langchain_core.documents import Document
from app.services.cleaning.pipeline import TextCleaningPipeline

def test_text_cleaning_pipeline():
    # Dirty text sample with various issues
    dirty_text = (
        "Enterpriseâ€œ RAG System â€\x00\n\n"
        "This is an enterprise- \nlevel document process-\ning pipeline. │\n\n"
        "This is an enterprise- \nlevel document process-\ning pipeline. │\n\n\n\n\n"
        "It supports high   accuracy   retrieval.   "
    )

    raw_doc = Document(
        page_content=dirty_text,
        metadata={"source": "dirty_sample.pdf", "file_type": "pdf"}
    )

    pipeline = TextCleaningPipeline()
    cleaned_doc = pipeline.clean(raw_doc)

    print("--- RAW TEXT ---")
    print(repr(raw_doc.page_content))
    print("\n--- CLEANED TEXT ---")
    print(cleaned_doc.page_content)
    print("\n--- METADATA AUDIT TRAIL ---")
    print(cleaned_doc.metadata)

    # Assertions
    assert "process- \ning" not in cleaned_doc.page_content
    assert "│" not in cleaned_doc.page_content
    assert "\n\n\n" not in cleaned_doc.page_content
    assert len(cleaned_doc.metadata["cleaned_by"]) == 5
    print("\n✓ Step 2 Cleaning Pipeline Verification Passed Successfully!")

if __name__ == "__main__":
    test_text_cleaning_pipeline()