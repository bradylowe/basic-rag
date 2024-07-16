import streamlit as st
from query_data import query_rag  # Import the main function from query_data.py


def main():
    st.title("Query App")

    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Query", "Resources", "Prompt"])

    with tab1:
        st.header("Query Page")
        query = st.text_input("Enter your query:")
        no_response = st.checkbox("Do not query the LLM, just do nearest neighbors")
        if st.button("Submit"):
            resources, context, prompt, response = query_rag(query, run_query=not no_response)
            st.session_state.resources = resources
            st.session_state.context = context
            st.session_state.prompt = prompt
            st.session_state.response = response
            if response:
                st.success("Successfully fetched relevant resources and queried the LLM.")
                st.text(response.content)
            else:
                st.success("Successfully fetched relevant resources without querying the LLM.")
    
    with tab2:
        st.header("Resources Page")
        if 'resources' in st.session_state:
            st.write("Relevant Resources:")
            for i, resource in enumerate(st.session_state.resources):
                st.write(f"{i+1}. {resource.metadata.get('id')}")
                st.write(str(resource.page_content))
        else:
            st.write("Submit a query to see the resources.")

    with tab3:
        st.header("Prompt Page")
        if 'prompt' in st.session_state:
            st.text_area("Prompt sent to the LLM", st.session_state.prompt, height=300)
        else:
            st.write("Submit a query to see the prompt.")


if __name__ == "__main__":
    main()
