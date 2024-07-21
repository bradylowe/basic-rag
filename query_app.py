import streamlit as st

from config import CHROMA_PATH, DATA_PATH
from populate_database import load_embeddings, clear_database
from query_data import query_rag


SUPPORTED_DOC_TYPES = ["pdf", "txt"]

def main():
    st.title("Query App")

    # Tab 1 session state defaults
    if "k" not in st.session_state:
        st.session_state.k = 5
    if "no_response" not in st.session_state:
        st.session_state.no_response = False
    
    # Tab 4 session state defaults
    if "chroma_path" not in st.session_state:
        st.session_state.chroma_path = CHROMA_PATH
    if "dataset_folder" not in st.session_state:
        st.session_state.dataset_folder = DATA_PATH
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 100
    if "min_chunk_size" not in st.session_state:
        st.session_state.min_chunk_size = 60
    if "doc_type" not in st.session_state:
        st.session_state.doc_type = SUPPORTED_DOC_TYPES[0]
    if "recursive" not in st.session_state:
        st.session_state.recursive = False

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Query", "Resources", "Prompt", "Data Loading"])

    with tab1:
        st.header("Query Page")

        k = st.number_input("Number of neighbors to retrieve:", min_value=1, value=st.session_state.k)
        no_response = st.checkbox("Do not query the LLM, just do nearest neighbors", value=st.session_state.no_response)

        def save_state_tab_1():
            st.session_state.k = k
            st.session_state.no_response = no_response

        query = st.text_input("Enter your query:")
        
        if st.button("Submit"):

            save_state_tab_1()

            # Make the query
            resources, context, prompt, response = query_rag(
                query, 
                run_query=not st.session_state.no_response, 
                chroma_path=st.session_state.chroma_path,
                k=st.session_state.k,
            )
            
            # Update session state with responses
            st.session_state.resources = resources
            st.session_state.context = context
            st.session_state.prompt = prompt
            st.session_state.response = response
            
            # Show user
            if response:
                st.success("Successfully fetched relevant resources and queried the LLM.")
                st.markdown(response.content)
            else:
                st.success("Successfully fetched relevant resources without querying the LLM.")
    
    with tab2:
        st.header("Resources Page")
        if 'resources' in st.session_state:
            st.write("Relevant Resources:")
            for i, resource in enumerate(st.session_state.resources):
                st.write(f"{i+1}. {resource.metadata.get('id', 'Unknown ID')}")
                st.write(str(resource.page_content))
        else:
            st.write("Submit a query to see the resources.")

    with tab3:
        st.header("Prompt Page")
        if 'prompt' in st.session_state:
            st.text_area("Prompt sent to the LLM", st.session_state.prompt, height=300)
        else:
            st.write("Submit a query to see the prompt.")
    
    with tab4:
        st.header("Settings and Data Loading")
        
        chroma_path = st.text_input("**Chroma DB Path** (Save to):", value=st.session_state.chroma_path)
        dataset_folder = st.text_input("**Dataset Folder Path** (Load from):", value=st.session_state.dataset_folder)

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            doc_type = st.selectbox("**Document Type**:", ["pdf", "txt"], key=st.session_state.doc_type)
        with col2:
            chunk_size = st.number_input("**Chunk Size**", value=st.session_state.chunk_size)
        with col3:
            chunk_overlap = st.number_input("**Chunk Overlap**", value=st.session_state.chunk_overlap)
        with col4:
            min_chunk_size = st.number_input("**Min Chunk Size**", value=st.session_state.min_chunk_size)
        
        recursive = st.checkbox("Load documents recursively", value=st.session_state.recursive)

        def save_state_tab_4():
            st.session_state.chroma_path = chroma_path
            st.session_state.dataset_folder = dataset_folder
            st.session_state.doc_type = doc_type
            st.session_state.recursive = recursive
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.min_chunk_size = min_chunk_size

        if st.button("Load Embeddings"):
            save_state_tab_4()
            n_added = load_embeddings(
                st.session_state.chroma_path, 
                st.session_state.dataset_folder, 
                st.session_state.doc_type,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
                min_chunk_size=st.session_state.min_chunk_size,
            )
            plural = "" if n_added == 1 else "s"
            st.success(f"Added {n_added} document{plural} to the database")
        
        if st.button("Clear Database"):
            save_state_tab_4()
            clear_database(st.session_state.chroma_path)
            st.success("Database cleared successfully!")


if __name__ == "__main__":
    main()
