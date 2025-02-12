"""
Example usage of the llm_loader package for RAG applications.
"""
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from llm_loader.document_loader import LLMLoader
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# OpenAI API key since we are using the gpt-4o-mini model for question-answering
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI"

# Gemini API key since we are using the gemini flash model
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Initialize the loader from the llm_loader package
    loader = LLMLoader(
        file_path="./data/test.pdf",
        chunk_strategy="contextual",
        model="gemini/gemini-1.5-flash",
        # Using google gemini flash as it is cheaper and performs better than other models for OCR
        save_output=True,
        output_dir="/Users/krypton/PycharmProjects/llm-loader/examples/data",
    )

    # Load and split the documents into chunks
    docs = loader.load_and_split()
    # Create a vector store from the documents using FAISS instead of Chroma
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    # Create the prompt template
    prompt_template = PromptTemplate.from_template(
        """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Question: {question}
    Context: {context}
    Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Example usage of the RAG chain
    question = "What are the key components of the o80 framework, and how do they interact?"
    answer = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
