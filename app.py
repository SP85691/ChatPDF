import os
import streamlit as st
from streamlit_chat import message
from PDFAgent import MultiPDFDocAgent
from streamlit_pdf_viewer import pdf_viewer

# Ensure the streamlit state is initialized
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chain' not in st.session_state:
    st.session_state.chain = None

st.title("Custom PDF Chat Agent")
st.subheader('Upload Your Desired PDF and Chat with it 😎', divider='rainbow')

# Selectbox for choosing upload method
upload_method = st.selectbox("Select upload method", ["Local Computer", "Website"])

if upload_method == "Local Computer":
    # File uploader for PDFs
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Create a local directory to save uploaded files if it doesn't exist
        if not os.path.exists("uploaded_pdfs"):
            os.makedirs("uploaded_pdfs")
        
        # Save the uploaded file to the local directory
        file_path = os.path.join("uploaded_pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        # st.write(f"File saved at: {file_path}")

        # Display the PDF in the sidebar
        with st.sidebar:
            # Checkbox for selecting between "groq" and "ollama"
            st.header("Select Language Model")
            selected_model = st.selectbox("Select a Model", ("Groq", "Ollama", "OpenAI", "Google Palm"), placeholder="Please Select a Model")
            st.info(f"Selected Model: {selected_model}")
            # Temperature slider
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

            # Max tokens slider
            max_tokens = st.slider("Max Tokens", min_value=128, max_value=2048, value=256, step=128)

            st.session_state.agent = MultiPDFDocAgent(llm = selected_model, filePath=file_path, temperature=temperature, max_tokens=max_tokens)
            texts, metadatas = st.session_state.agent.load_pdf_locally()
                
            st.header("Uploaded PDF Preview")
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            st.write(f"Preview of '{uploaded_file.name}':")
            if uploaded_file:
                binary_data = uploaded_file.getvalue()
                pdf_viewer(input=binary_data,
                        width=700)
        
        st.session_state.chain = st.session_state.agent.textChunk_to_docObj(texts, metadatas)

    else:
        st.warning("Please upload a PDF file to proceed.")

        # Sidebar message when no file is uploaded
        with st.sidebar:
            st.header("Uploaded PDF Preview")
            st.write("No PDF uploaded.")
        
elif upload_method == "Website":
    # Input box for website URL
    website_url = st.text_input("Enter the URL of the PDF file")

    # Check if URL is provided
    if website_url:
        st.info(f"You entered: {website_url}")
        
        # Display the PDF in the sidebar
        with st.sidebar:
            # Checkbox for selecting between "groq" and "ollama"
            st.header("Select Language Model")
            selected_model = st.selectbox("Select a Model", ("Groq", "Ollama", "OpenAI", "Google Palm"), placeholder="Please Select a Model")
            st.info(f"Selected Model: {selected_model}")
            # Temperature slider
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

            # Max tokens slider
            max_tokens = st.slider("Max Tokens", min_value=128, max_value=2048, value=256, step=128)

            st.session_state.agent = MultiPDFDocAgent(llm = selected_model, filePath=website_url, temperature=temperature, max_tokens=max_tokens)
            texts, metadatas = st.session_state.agent.load_pdf_locally()
            
                
            st.header("Uploaded PDF Preview")
            st.success(f"File '{website_url}' uploaded successfully!")
            st.write(f"Preview of '{website_url}':")
            # if uploaded_file:
            #     binary_data = uploaded_file.getvalue()
            #     pdf_viewer(input=binary_data,
            #             width=700)
        
        st.session_state.chain = st.session_state.agent.textChunk_to_docObj(texts, metadatas)
        
        # Proceed with further processing using the file_path variable
        # For example, you can instantiate the MultiPDFDocAgent and start a chat session

    else:
        st.warning("Please upload a PDF file to proceed.")

        # Sidebar message when no file is uploaded
        with st.sidebar:
            st.header("Uploaded PDF Preview")
            st.write("No PDF uploaded.")
st.divider()

if st.session_state.chain:
    query = st.chat_input("Ask a question about the PDF")
    
    if query:
        # Invoke the chat method
        answer = st.session_state.agent.chat(st.session_state.chain, query)
        
        # Add the query and answer to the chat history
        st.session_state.chat_history.append((query, answer))
        
        # Display the chat history
        for query, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(query)
            message = st.chat_message("assistant")
            message.write(answer)
else:
    st.info("Upload a PDF to start chatting.")
