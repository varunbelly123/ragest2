import streamlit as st
import openai
from dotenv import load_dotenv
import os
import brain  # Import your updated brain.py with .pdf, .docx, .pptx, and .xlsx support

load_dotenv()

# Set the title for the Streamlit app
st.title("RAG Assistant")

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Upload files using Streamlit's file uploader
uploaded_files = st.file_uploader(
    "Upload Text, PDF, DOCX, PPTX, or XLSX Files", type=["txt", "pdf", "docx", "pptx", "xlsx"], accept_multiple_files=True
)

# Initialize session state for chat and documents
if 'text_files' not in st.session_state:
    st.session_state.text_files = []
if 'prompt' not in st.session_state:
    st.session_state.prompt = [{"role": "system", "content": "Welcome to the RAG Assistant! How can I help you today?"}]

# Clear button functionality
if st.button("Clear Chat"):
    st.session_state.prompt = [{"role": "system", "content": "Welcome to the RAG Assistant! How can I help you today?"}]
    st.session_state.text_files = st.session_state.text_files  # Keep the uploaded files

# Display uploaded file names at the top
if uploaded_files:
    uploaded_file_names = [file.name for file in uploaded_files]
    st.session_state.text_files = uploaded_files  # Save uploaded files in session state
    st.write("Uploaded Documents:")
    for name in uploaded_file_names:
        st.write(name)

# Define a function to process text, pdf, docx, pptx, and xlsx files
def process_uploaded_files(files, file_names):
    """
    Process the uploaded files and create an index using the brain module.

    Parameters:
        files (List[File]): List of uploaded files (txt, pdf, docx, pptx, xlsx).
        file_names (List[str]): List of names of the uploaded files.

    Returns:
        index: FAISS index created from the uploaded files.
    """
    if files:
        index = brain.get_index_for_text_files(files, file_names, openai.api_key)
        return index
    else:
        st.error("No files uploaded.")
        return None

# Process uploaded files
index = process_uploaded_files(st.session_state.text_files, [file.name for file in st.session_state.text_files])

# Display previous chat messages
for message in st.session_state.prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User question input
question = st.chat_input("Ask me anything to get started")

# Handle the user's question
if question:
    if index is None:
        with st.chat_message("assistant"):
            st.write("No documents processed. Please upload documents.")
            st.stop()

    # Retrieve relevant documents based on the user's question
    relevant_docs = index.similarity_search(question, k=3)  # Adjust k based on your needs
    context = "\n".join(doc.page_content for doc in relevant_docs)

    # Create a prompt for the assistant that includes the retrieved context
    prompt_template = """
    Welcome to the Assistant!

    I am here to assist you with answering questions based on the content of the provided documents. 
    Here are some ways you can interact with me:

    1. Document-Based Question Answering (QA):
       - Ask me any question related to the document content, and I will provide responses based on the information within.

    2. Section-Specific Queries:
       - Reference key phrases or topics from the document for specific details.

    3. Clarifications:
       - Ask for explanations or clarifications on terms or any part of the document.

    Important: All my responses will be based on the document content provided.

    Context: {context}

    User Question: {question}
    """

    # Update the prompt with the retrieved context and user's question
    st.session_state.prompt[0] = {
        "role": "system",
        "content": prompt_template.format(context=context, question=question),
    }

    st.session_state.prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Display an empty assistant message while waiting for the response
    with st.chat_message("assistant"):
        botmsg = st.empty()

    # Call OpenAI's ChatCompletion API and display the response as it comes
    response = []
    result = ""
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=st.session_state.prompt, stream=True
    ):
        text = chunk.choices[0].get("delta", {}).get("content")
        if text is not None:
            response.append(text)
            result = "".join(response).strip()
            botmsg.write(result)

    # Add the assistant's response to the prompt
    st.session_state.prompt.append({"role": "assistant", "content": result})
