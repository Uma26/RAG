from langchain_core.documents import Document
from app.services.ingestion.factory import DocumentLoaderFactory

def test_ingestion_factory():
    # 1. Test URL Loader
    web_loader = DocumentLoaderFactory.get_loader("https")
    doc_web = web_loader.load("https://google.com")
    assert isinstance(doc_web, Document)
    assert doc_web.metadata["file_type"] == "web"
    print("✓ Web Loader Test Passed")

    # 2. Test TXT Loader
    txt_loader = DocumentLoaderFactory.get_loader("txt")
    doc_txt = txt_loader.load("Hello Enterprise RAG", filename="test.txt")
    assert isinstance(doc_txt, Document)
    assert doc_txt.metadata["source"] == "test.txt"
    print("✓ TXT Loader Test Passed")

if __name__ == "__main__":
    test_ingestion_factory()
    print("All Step 1 tests executed successfully!")