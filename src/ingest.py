from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import get_pdf_path, get_vector_store


def ingest_pdf():
    pdf_path = get_pdf_path()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF nao encontrado em: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Nenhum chunk foi gerado a partir do PDF informado.")

    vector_store = get_vector_store(pre_delete_collection=True)
    vector_store.add_documents(chunks)

    print(f"PDF processado: {pdf_path.name}")
    print(f"Paginas carregadas: {len(documents)}")
    print(f"Chunks gerados: {len(chunks)}")


if __name__ == "__main__":
    ingest_pdf()
