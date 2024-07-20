import argparse
from collections import defaultdict
import os
import shutil

from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from config import CHROMA_PATH, DATA_PATH
from get_embedding_function import get_embedding_function


SUPPORTED_DOC_TYPES = ["pdf", "txt"]


def main():

    # Set up CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--doc_type", type=str, default="pdf", choices=SUPPORTED_DOC_TYPES)
    parser.add_argument("--chroma", type=str, default=CHROMA_PATH, help="Path to chroma db folder")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data folder")
    args = parser.parse_args()
    
    # Check if the database should be cleared (using the --clear flag).
    if args.reset:
        print("✨ Clearing Database")
        clear_database(args.chroma)
    
    # Create the right kind of file loader
    if args.doc_type == "pdf":
        loader = PyPDFDirectoryLoader(args.data)
    elif args.doc_type == "txt":
        loader = DirectoryLoader(
            args.data, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
    else:
        raise ValueError(f"Please set the value of `DOC_TYPE` to one of {SUPPORTED_DOC_TYPES}")
    
    # Load and prepare the data
    docs = loader.load()
    chunks = split_documents(docs)
    add_ids_to_chunks(chunks)
    
    # Add records to DB
    add_to_chroma(args.chroma, chunks)


def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 80):
    """Recursively split a list of documents into a longer list of shorter documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(db_path: str, chunks: list[Document]):
    """Add a list of chunks to the db
    """
    # Load the existing database.
    db = Chroma(
        persist_directory=db_path, 
        embedding_function=get_embedding_function()
    )

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = [ch for ch in chunks if ch.metadata["id"] not in existing_ids]

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def add_ids_to_chunks(chunks: list[Document]):
    """Calculate unique IDs and add them to the chunk metadata inplace for each chunk
    """

    # Keep track of chunk count per document
    chunk_id_counts = defaultdict(int)
    def get_next_chunk_count(chunk_id: str) -> int:
        chunk_id_counts[chunk_id] += 1
        return chunk_id_counts[chunk_id]

    for chunk in chunks:

        # Add source to chunk ID
        chunk_id = chunk.metadata.get("source")
        
        # Add page to chunk ID
        if "page" in chunk.metadata:
            chunk_id += f":page={chunk.metadata['page']}"
        
        # Add unique chunk count to chunk ID
        idx = get_next_chunk_count(chunk_id)
        chunk_id += f":chunk={idx}"

        # Save the ID
        chunk.metadata["id"] = chunk_id


def clear_database(db_path: str):
    """Clear the database
    """
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


if __name__ == "__main__":
    main()
