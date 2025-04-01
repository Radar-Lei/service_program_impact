import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

if __name__ == "__main__":

    hg_embeddings = HuggingFaceEmbeddings()
    loader_eco = CSVLoader('eco_ind.csv')
    documents_eco = loader_eco.load()

    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)

    # Split your docs into texts
    texts_eco = text_splitter.split_documents(documents_eco)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    persist_directory = 'docs/chroma_rag/'
    economic_langchain_chroma = Chroma.from_documents(
        documents=texts_eco,
        collection_name="economic_data",
        embedding=hg_embeddings,
        persist_directory=persist_directory
    )