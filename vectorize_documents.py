from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader,CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Specifying the embedding model to use
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setting up the loader
loader = DirectoryLoader(path="data",
                         glob="./*.csv",
                         loader_cls=UnstructuredFileLoader)
# loader = CSVLoader(file_path='cleaned_linkedin_dataset.csv') 
try:
    documents = loader.load()
except Exception as e:
    print(f"Error loading documents: {e}")
    documents = []

if documents:
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="vector_db_dir"
    )
    print("Documents Vectorized")
else:
    print("No documents loaded or processed.")
