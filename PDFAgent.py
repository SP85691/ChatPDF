import os
import PyPDF2
from colorama import Fore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma, FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.storage import LocalFileStore
from langchain.globals import set_llm_cache
from langchain_core.documents.base import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.cache import InMemoryCache, SQLiteCache

load_dotenv()

class MultiPDFDocAgent():
    def __init__(self, llm = "groq", temperature=0.7, max_tokens=256, filePath = None):
        if llm == "ollama" or llm == "Ollama" or llm == "1" or llm == "llama3":
            print("Calling Ollama Agent")
            self.llm = ChatOllama(model="llama3.1", temperature=temperature, max_tokens=max_tokens, verbose=True)
        
        elif llm == "groq" or llm == "Groq" or llm == "2":
            print("Calling Groq Agent")
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-8b-8192",
                # model="mixtral-8x7b-32768",
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=True
            )
        
        elif llm == "openai" or llm == "Openai" or llm == "OpenAI":
            print("Calling OpenAI Agent")
            self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=temperature, 
                max_tokens=max_tokens)
        
        # set_llm_cache(InMemoryCache())
        print("Set LLM Cache at 'llm_cache.db'")
        set_llm_cache(SQLiteCache("llm_cache.db"))

        if filePath == None:
            print("No File Path given!")
        else:
            self.filePath = filePath
        
        print(f"\033[91mFile is Being Processed: \033[97m'{self.filePath}'")
        
    def load_pdf_locally(self):
        loader = PyPDFLoader(self.filePath)
        
        pages = loader.load()
        print(f"Number of Pages: {len(pages)}")
    

        # Concatenate all page contents into a single string
        full_text = " ".join([page.page_content for page in pages])

        # Initialize the text splitter
        print("Splitting the Pages text into chunks using 'titoken_encoder'")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=2000, chunk_overlap=400
        )

        # Split the concatenated text
        texts = text_splitter.split_text(full_text)
        metadatas = [{"Sources": f"{i}-pl"} for i in range(len(texts))]
        
        # Print the split texts
        # print(f"Text: {texts}")
        
        # Save the texts array to a file named 'file.txt'
        # with open("file.txt", "w") as file:
        #     for text_chunk in texts:
        #             file.write(text_chunk + "\n")
        
        # The `return texts, metadatas` statement in the `load_pdf_locally` method of the
        # `MultiPDFDocAgent` class is returning two values as a tuple.
        return texts, metadatas
    
    
    def textChunk_to_docObj(self, texts, metadatas):            
        # Create Document objects from text chunks
        documents = [Document(page_content=text_chunk) for text_chunk in texts]
        
        underlying_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        store = LocalFileStore("./cache/")
        print("Generating Embeddings and Saving it to the './cache/'...")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, store, namespace=underlying_embeddings.model
        )
        
        # Create a Chroma Vector Store
        # db = FAISS.from_texts(texts, cached_embedder, metadatas=metadatas)
        db = FAISS.from_documents(documents, embedding=cached_embedder)
        # print(list(store.yield_keys()))
        
        # Initialize Message History for Conversation
        message_history = ChatMessageHistory()

        # Memory for Conversational Context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create Chain that uses the Chroma Vector Store
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            chain_type="stuff",
            retriever=db.as_retriever(), 
            memory=memory,
            return_source_documents=True,
        )
        
        return self.chain
        
    def chat(self, chain, query):
        docs = chain.invoke(query)
        answer = docs["answer"]
        return answer