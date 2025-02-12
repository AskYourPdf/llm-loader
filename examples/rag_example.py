"""
Example usage of different document loaders (llm_loader and PyMuPDF) for RAG applications.
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from llm_loader.document_loader import LLMLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# OpenAI API key since we are using the gpt-4o-mini model for question-answering
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Gemini API key since we are using the gemini flash model
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"


def create_rag_chain(retriever, llm):
    """Create a RAG chain with the given retriever and LLM."""
    prompt_template = PromptTemplate.from_template(
        """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise
    Question: {question}
    Context: {context}
    Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )


def process_with_llmloader():
    """Process documents using LLMLoader with Gemini Flash."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Initialize the loader from the llm_loader package
    loader = LLMLoader(
        file_path="./data/test.pdf",
        chunk_strategy="contextual",
        model="gemini/gemini-1.5-flash",
        save_output=True,
        output_dir="./data",
    )

    docs = loader.load_and_split()
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    rag_chain = create_rag_chain(vectorstore.as_retriever(), llm)
    return rag_chain


def process_with_pymupdf():
    """Process documents using PyMuPDF with recursive chunking."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load document with PyMuPDF
    loader = PyMuPDFLoader("./data/test.pdf")
    documents = loader.load()

    # Create text splitter for recursive chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # Split documents into chunks
    docs = text_splitter.split_documents(documents)

    # Create vector store and retriever
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    rag_chain = create_rag_chain(vectorstore.as_retriever(), llm)
    return rag_chain


def main():
    # Example using LLMLoader
    print("\n=== Using LLMLoader ===")
    llm_chain = process_with_llmloader()
    question = "What are the benefits of using o80 in reinforcement learning environments?"
    answer = llm_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Example using PyMuPDF
    print("\n=== Using PyMuPDF ===")
    pymupdf_chain = process_with_pymupdf()
    answer = pymupdf_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
