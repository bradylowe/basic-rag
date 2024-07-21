import os
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai.chat_models import ChatOpenAI

from config import OPENAI_API_KEY, CHROMA_PATH, DATA_PATH
from get_embedding_function import get_embedding_function


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--no_response", action="store_true", help="Do not query the LLM, just do nearest neighbors")
    parser.add_argument("--chroma", type=str, default=CHROMA_PATH, help="Path to the chroma db folder")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to data folder")
    parser.add_argument("--k", type=int, default=5, help="Number of relevant resources to retrieve")
    args = parser.parse_args()

    resources, context, prompt, response = query_rag(args.query_text, run_query=not args.no_response, chroma_path=args.chroma, k=args.k)
    sources = get_ids_from_resources(resources)
    formatted_response = format_response_for_cli(response, sources)

    print()
    print('###  PROMPT  ###')
    print()
    print(prompt)
    print()
    print()
    print('###  RESPONSE  ###')
    print()
    print(formatted_response)


def format_response_for_cli(response: str, sources: list[str]):
    return f"Response: {response}\nSources: {sources}"


def query_model(prompt: str):
    #model = Ollama(model="mistral")
    #model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)  # 10x cost, 10x quality
    model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)  # 0.25x cost, 11x quality
    #model = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    return model.invoke(prompt)


def get_ids_from_resources(resources: list):
    return [doc.metadata.get("id", None) for doc in resources]


def query_rag(
        query_text: str, 
        run_query: bool = True, 
        chroma_path: str = '',
        k: int = 5,
    ):
    """
    Query the vector database for resources relevant to the ``query_text``, then 
    optionally pass the context and query to the LLM.

    Here are the steps taken in this function:
    ---
    
    1. Embed the user query
    2. Nearest neighbors search to find the most relevent resources
    3. Generate a prompt for the LLM using the resources and user query
    4. Query the LLM
    5. Return

    Parameters
    ---
    query_text: str
        The text to query the database with.
    run_query: bool
        Whether to query the LLM or not. If False, only nearest neighbors query will run.
    chroma_path: str
        Path to the chroma db
    k: int
        Number of relevant resources to retrieve from the db

    Returns
    ---
    resources: list 
        List of relevant resources returned from DB query
    context: str
        The context for the LLM query
    prompt: str
        The prompt for the LLM query
    response: str
        The response from the LLM query
    """

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=chroma_path or CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    resources = db.similarity_search_with_score(query_text, k=k)
    resources = sorted(resources, key=lambda x: x[1])

    # Separate the resources from their scores
    # resource_scores = [item[1] for item in resources]
    resources = [item[0] for item in resources]

    # Build the context text from the search resources
    # Reverse the order of the resources, LLM remembers the end better
    context_text = "\n\n---\n\n".join([f"[{doc.metadata.get('source', 'Unknown source')}]\n{doc.page_content}" for doc in reversed(resources)])

    # Generate the prompt for the model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # If no query was requested, then exit
    if not run_query:
        return resources, context_text, prompt, None
    
    # Query the LLM
    response = query_model(prompt)

    return resources, context_text, prompt, response


if __name__ == "__main__":
    main()
