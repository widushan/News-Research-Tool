import os
import time
from html.parser import HTMLParser
from typing import Iterable, List, Sequence

import requests
import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.globals import set_debug

# Maintain backwards-compatible name used in older tutorials
OpenAI = ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS


class SimpleRetrievalQAWithSourcesChain:
    """Minimal RetrievalQA chain compatible with latest LangChain packages."""

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever,
        prompt: ChatPromptTemplate | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt or ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a careful research assistant. Answer using only the"
                    " supplied context. Cite sources in parentheses using the"
                    " provided source names. If context is empty, say you cannot"
                    " answer.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer in"
                    " 4-6 sentences and include a Sources line listing the"
                    " source names you used.",
                ),
            ]
        )

    @staticmethod
    def _format_docs(docs: Iterable[Document]) -> str:
        chunks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("url")
            label = source or f"Chunk-{idx}"
            chunks.append(f"Source: {label}\n{doc.page_content}")
        return "\n\n".join(chunks)

    @staticmethod
    def _collect_sources(docs: Iterable[Document]) -> List[str]:
        seen: List[str] = []
        for doc in docs:
            candidate = doc.metadata.get("source") or doc.metadata.get("url")
            if candidate and candidate not in seen:
                seen.append(candidate)
        return seen

    @classmethod
    def from_chain_type(cls, llm: ChatOpenAI, retriever, **kwargs):
        prompt = kwargs.get("prompt") or kwargs.get("chain_type_kwargs", {}).get("prompt")
        return cls(llm=llm, retriever=retriever, prompt=prompt)

    def invoke(self, inputs):
        if isinstance(inputs, str):
            question = inputs
        elif isinstance(inputs, dict):
            question = inputs.get("question") or inputs.get("query")
            if not question:
                raise ValueError("Provide a 'question' or 'query'.")
        else:
            raise TypeError("Inputs must be a string or dict containing 'question'.")

        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(question)
        else:
            docs = self.retriever.invoke(question)
        context = self._format_docs(docs)
        messages = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        answer_text = response.content if hasattr(response, "content") else str(response)
        return {"answer": answer_text.strip(), "sources": self._collect_sources(docs)}

    def __call__(self, inputs, **_):
        # Accept unused kwargs (e.g., return_only_outputs) for compatibility
        return self.invoke(inputs)


class _HTMLTextExtractor(HTMLParser):
    """Very small helper to strip markup without requiring heavy deps."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _load_documents_from_urls(urls: Sequence[str]) -> List[Document]:
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    documents: List[Document] = []
    for url in urls:
        extractor = _HTMLTextExtractor()
        try:
            response = requests.get(url, headers=default_headers, timeout=30)
            response.raise_for_status()
            extractor.feed(response.text)
            text = extractor.get_text()
            if not text:
                st.warning(f"No readable text found at {url}.")
                continue
            documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load {url}: {exc}")
    return documents

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Research Bot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
vector_store_dir = "faiss_store_openai"

main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

if process_url_clicked:
    # load data
    filtered_urls = [url.strip() for url in urls if url.strip()]
    if not filtered_urls:
        st.warning("Please provide at least one URL before processing.")
        st.stop()

    docs_raw = _load_documents_from_urls(filtered_urls)
    if not docs_raw:
        st.error("Unable to fetch any of the provided URLs.")
        st.stop()

    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = docs_raw
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        chunk_overlap=200,
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index using FAISS's built-in method
    vectorstore_openai.save_local(vector_store_dir)
    main_placeholder.text("Vector Store Saved Successfully!✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.isdir(vector_store_dir):
        # Load the FAISS index using FAISS's built-in method
        vectorstore = FAISS.load_local(
            vector_store_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        chain = SimpleRetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)
        result = chain({"question": query})
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources") or []
        if sources:
            st.subheader("Sources:")
            for source in sources:
                st.write(source)
    else:
        st.warning("No vector store found. Please process URLs first.")