import ast
from collections import defaultdict
import os
import shutil
from tqdm import trange

from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from config import CHROMA_PATH, DATA_PATH, MAX_EMBEDDING_BATCH_SIZE
from get_embedding_function import get_embedding_function


SUPPORTED_DOC_TYPES = ["pdf", "txt"]


def clear_database(db_path: str):
    """Clear the database"""
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def get_loader(doc_type: str, data_path: str):
    """Get the appropriate loader based on document type"""
    if doc_type == "pdf":
        return PyPDFDirectoryLoader(data_path)
    elif doc_type == "txt":
        return DirectoryLoader(
            data_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
    else:
        raise ValueError(f"Please set the value of `doc_type` to one of {SUPPORTED_DOC_TYPES}")


def split_documents_basic_recursive(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Split a set of documents into a larger set of smaller documents with 
    a fixed size and overlap (in number of characters)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def split_python_doc(document: Document) -> list[Document]:
    """
    Split a single code document into semantic chunks (functions and classes)

    This function produces 3 sets of redundant chunks:

    1. Simple chunks made from a constant chunk size and overlap
    2. Chunks based on function definitions
    3. Chunks based on class definitions
    """

    text = document.page_content
    source = document.metadata.get('source')

    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"Couldn't parse file: {source}")
        return []

    class FunctionClassVisitor(ast.NodeVisitor):
        def __init__(self):
            self.chunks = []

        def visit_FunctionDef(self, node):
            start_lineno = node.lineno
            end_lineno = node.end_lineno
            function_text = '\n'.join(text.splitlines()[start_lineno - 1:end_lineno])
            doc = Document(page_content=function_text, metadata={"source": source, "content_type": "function", "name": node.name})
            self.chunks.append(doc)
            if len(function_text) > 1000:
                short_doc = Document(page_content=function_text[:800], metadata={"source": source, "content_type": "function_intro", "name": node.name})
                self.chunks.append(short_doc)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            #start_lineno = node.lineno
            #end_lineno = node.end_lineno
            #class_text = '\n'.join(text.splitlines()[start_lineno - 1:end_lineno])
            class_text = self.extract_class_signature(node)
            doc = Document(page_content=class_text, metadata={"source": source, "content_type": "class", "name": node.name})
            self.chunks.append(doc)
            self.generic_visit(node)

        def extract_class_signature(self, node):
            methods = []
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef):
                    methods.append(self.extract_function_signature(body_item))
            class_signature = f"class {node.name}:\n    " + "\n    ".join(methods)
            return class_signature

        def extract_function_signature(self, node):
            params = [arg.arg for arg in node.args.args]
            param_list = ', '.join(params)
            func_signature = f"def {node.name}({param_list}):"
            return func_signature
    
    visitor = FunctionClassVisitor()
    visitor.visit(tree)
    return visitor.chunks


def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 80, min_chunk_size: int = 0) -> list[Document]:
    """
    Split the code into chunks for RAG embedding

    Perform extra splitting for python files. First, do the normal 
    recursive text splitting to get all text, comments, etc. from 
    the python file. Then, perform a semantic chunking where we 
    get one chunk for each class definition and function definition. 
    """

    # Do a basic recursive splitting on all documents
    chunks = split_documents_basic_recursive(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Do extra semantic splitting for python files
    for doc in documents:
        if doc.metadata.get('source', '').endswith('.py'):
            chunks.extend(split_python_doc(doc))

    # Filter chunks that are too small
    if min_chunk_size:
        chunks = [ch for ch in chunks if len(ch.page_content) >= min_chunk_size]
    
    return chunks


def add_batch_to_chroma(db, batch: list[Document]):
    """Add a batch of documents to chroma db"""
    print(f"➕ Adding new documents: {len(batch)}")
    batch_ids = [chunk.metadata["id"] for chunk in batch]
    db.add_documents(batch, ids=batch_ids)


def add_to_chroma(db_path: str, chunks: list[Document]) -> int:
    """Add a long list of chunks to the db and return the number of items added"""
    db = Chroma(
        persist_directory=db_path, 
        embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [ch for ch in chunks if ch.metadata["id"] not in existing_ids]

    # Load documents into db (batch if necessary)
    if len(new_chunks) > MAX_EMBEDDING_BATCH_SIZE:
        print(f"🔎 Located {len(new_chunks)} new documents")
        for start_idx in trange(0, len(new_chunks), MAX_EMBEDDING_BATCH_SIZE, desc="Loading chunks"):
            add_batch_to_chroma(db, new_chunks[start_idx:start_idx + MAX_EMBEDDING_BATCH_SIZE])
    elif len(new_chunks) > 0:
        add_batch_to_chroma(db, new_chunks)
    else:
        print("✅ No new documents to add")
    
    return len(new_chunks)


def add_ids_to_chunks(chunks: list[Document]):
    """Calculate unique IDs and add them to the chunk metadata inplace for each chunk"""
    chunk_id_counts = defaultdict(int)

    def get_next_chunk_count(chunk_id: str) -> int:
        chunk_id_counts[chunk_id] += 1
        return chunk_id_counts[chunk_id]

    for chunk in chunks:
        chunk_id = chunk.metadata.get("source")

        if "page" in chunk.metadata:
            chunk_id += f":page={chunk.metadata['page']}"

        idx = get_next_chunk_count(chunk_id)
        chunk_id += f":chunk={idx}"

        chunk.metadata["id"] = chunk_id


def load_and_embed_documents(
        chroma_path: str, 
        data_path: str, 
        doc_type: str, 
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 0,
        reset: bool = False,
    ) -> int:
    """
    Main function to load and chunk documents, calculate embeddings, 
    and load them into Chroma DB.

    Parameters:
    -----------
    chroma_path : str
        Path to the Chroma database directory.
    
    data_path : str
        Path to the directory containing documents to be loaded.
    
    doc_type : str
        Type of documents to be loaded (e.g., 'pdf', 'txt').
    
    chunk_size : int, optional
        Maximum size of each document chunk. Default is 1000.
    
    chunk_overlap : int, optional
        Overlap size between consecutive document chunks. Default is 100.
    
    min_chunk_size : int, optional
        Minimum size of each document chunk. Default is 0.
    
    reset : bool, optional
        If True, clears the Chroma database before loading new documents. Default is False.

    Returns:
    --------
    int
        Number of chunks added to the Chroma database.
    """
    if reset:
        print("✨ Clearing Database")
        clear_database(chroma_path)
    
    loader = get_loader(doc_type, data_path)
    docs = loader.load()
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size)
    add_ids_to_chunks(chunks)
    return add_to_chroma(chroma_path, chunks)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--doc_type", type=str, default="pdf", choices=SUPPORTED_DOC_TYPES)
    parser.add_argument("--chroma", type=str, default=CHROMA_PATH, help="Path to chroma db folder")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data folder")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size (in characters)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap (in characters)")
    parser.add_argument("--min_chunk_size", type=int, default=0, help="Minimum chunk size allowed")
    args = parser.parse_args()
    
    load_and_embed_documents(
        args.chroma, 
        args.data, 
        args.doc_type, 
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        reset=args.reset,
    )
