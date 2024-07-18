# basic-rag

This is a basic implementation of a RAG system with a nice and simple interface.

There are some various functionalities like the ability to load the database 
and run queries through the use of two simple scripts, or you can use the 
streamlit web app instead. 

## Get Started

1. Install the requirements: `pip install -r requirements.txt`
2. Put your files in the "data" folder (currently supporting ".txt" and ".pdf")
3. Add your LLM API key to the ".env" folder (not needed if using Ollama or other local model)
    a. OpenAI is currently supported
    b. If you want to use another LLM provider, just change the imports in "query_data.py" and 
    get_embedding_function.py" and connect your API key
4. Run `python populate_database.py` or `python populate_database.py --doc_type txt`
    a. Run `python populate_database.py -h` to see all the options
5. Run `streamlit run .\app.py` to launch the app into a browser
        1. Run `python query_data.py "Here is my query" --no_response` to do nearest neighbors only

OR

5. Run the CLI: `python query_data.py "Here is my query"`
    a. Run `python query_data.py "Here is my query" --no_response` to run nearest neighbors only
    b. Run `python query_data.py -h` to see all of the options
