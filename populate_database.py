import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"

SUPPORTED_DOC_TYPES = ["pdf", "txt"]


def main():

    # Set up CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--doc_type", type=str, default="pdf", choices=SUPPORTED_DOC_TYPES)
    args = parser.parse_args()
    
    # Check if the database should be cleared (using the --clear flag).
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    # Create (or update) the data store.
    if args.doc_type == "pdf":
        documents = load_documents_pdf()
        chunks = split_documents(documents)
        chunks = calculate_chunk_ids_pdf(chunks)
    elif args.doc_type == "txt":
        documents = load_documents_txt()
        chunks = split_documents(documents)
        chunks = calculate_chunk_ids_txt(chunks)
    else:
        raise ValueError(f"Please set the value of `DOC_TYPE` to one of {SUPPORTED_DOC_TYPES}")
    
    # Add records to DB
    add_to_chroma(chunks)


def load_documents_txt():
    document_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    return document_loader.load()


def load_documents_pdf():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    chunks = calculate_chunk_ids_pdf(chunks)
    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids_pdf(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def calculate_chunk_ids_txt(chunks):

    # This will create IDs like "data/eminem_25tolife.txt:2"
    # Document Source : Chunk Index

    for chunk_idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source")
        chunk.metadata["id"] = f"{source}:{chunk_idx}"

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
