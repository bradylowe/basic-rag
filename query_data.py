import os
import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai.chat_models import ChatOpenAI

from config import OPENAI_API_KEY
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

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
    args = parser.parse_args()
    response = query_rag(args.query_text, run_query=not args.no_response, save_prompt=True, save_resources=True)
    print(response)


def save_prompt_to_file(prompt, path='output/prompt.txt'):
    if not os.path.exists('output'):
        os.makedirs('output')
    with open(path, 'w') as f:
        f.write(prompt + '\n')


def save_resources_to_file(results: list[tuple[str, int]], path='output/sorted_resources.txt'):
    if not os.path.exists('output'):
        os.makedirs('output')
    with open(path, 'w', encoding='utf-8') as f:
        for count, result in enumerate(results):
            f.write(f'[{result[1]}]\n{result[0]}\n')
            if count < len(results) - 1:
                f.write('\n---\n\n')


def query_model(prompt: str):
    #model = Ollama(model="mistral")
    #model = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)  # 10x cost, 10x quality
    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    return model.invoke(prompt)


def query_rag(query_text: str, run_query: bool = True, save_prompt: bool = False, save_resources: bool = False):
    """
    Make a query to the RAG system

    Here are the steps taken in this function:
    ---
    
    1. Embed the user query
    2. Nearest neighbors search to find the most relevent resources
    2a. (Optional) Save the sorted list of resources to a file
    3. Generate a prompt for the LLM using the resources and user query
    3a. (Optional) Save the LLM query to a file
    4. Query the LLM
    5. Print the response
    """

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    all_results = db.similarity_search_with_score(query_text, k=100)
    all_results = sorted(all_results, key=lambda x: x[1])
    top_results = all_results[:5]
    
    # Save the sorted references to a text file
    if save_resources:
        save_resources_to_file(all_results)

    # Build the context text from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in top_results])

    # Generate the prompt for the model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # print(prompt)
    if save_prompt:
        save_prompt_to_file(prompt)
    
    # If no query was requested, then exit
    if not run_query:
        return
    
    # Query the LLM
    response = query_model(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in top_results]
    return f"Response: {response}\nSources: {sources}"


if __name__ == "__main__":
    main()
