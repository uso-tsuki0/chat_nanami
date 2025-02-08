import json
import os
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import jieba
from rank_bm25 import BM25Okapi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from typing import Optional, List
from sentence_transformers import CrossEncoder
from typing import Annotated
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field, PrivateAttr
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import json
from collections import deque
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)





class BgeRerankerLLM():
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cuda"):
        self.reranker = CrossEncoder(model_name, device=device)

    def _call(self, inputs: str, **kwargs) -> str:
        raise NotImplementedError("Please only use the rerank method")
    
    @property
    def _llm_type(self) -> str:
        return "custom-bge-reranker"

    def rerank(self, query: str, documents: List[str]) -> List[dict]:
        pairs = [(query, doc) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(
            [{"content": doc, "score": score} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True
        )
        return ranked_docs



class BaseRetriever():
    """
    Base retriever with BM25 and vector retrieval
    """
    def __init__(self, docs):
        self.docs = docs
        self.texts = [doc.page_content for doc in docs]
        self.bm25_retriever = BM25Okapi([self.preprocess(text) for text in self.texts])

    def preprocess(self, text):
        return list(jieba.cut(text))

    def bm25_retrieve(self, query, k=50):
        return self.bm25_retriever.get_top_n(self.preprocess(query), self.texts, n=k)
    


class RrfRretriever(BaseRetriever):
    def __init__(self, docs):
        super().__init__(docs)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs = {'device': 'cuda:0'},
            encode_kwargs = {'normalize_embeddings': True}
        )
        self.vector_store = Chroma.from_documents(self.docs, self.embedding_model)
        self.reranker = BgeRerankerLLM()

    def vector_retrieve(self, query, k=50):
        return [doc.page_content for doc in self.vector_store.similarity_search(query, k=k)]

    def rrf(self, vector_results: List[str], text_results: List[str], k=50, m=30):
        doc_scores = {}
        for rank, doc_id in enumerate(vector_results):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
        for rank, doc_id in enumerate(text_results):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank+m)
        sorted_results = [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]]
        return sorted_results
    
    def rerank(self, query, results, rerank_top_k=10):
        candidates = results[:rerank_top_k]
        reranked_results = self.reranker.rerank(query, candidates)
        return reranked_results + results[rerank_top_k:]
        
    def retrieve(self, query, recall_k=50, rerank_k=10, top_k=5):
        vector_result = self.vector_retrieve(query, k=recall_k)
        bm25_result = self.bm25_retrieve(query, k=recall_k)
        rrf_result = self.rrf(vector_result, bm25_result, k=10)
        reranked_result = self.rerank(query, rrf_result, rerank_top_k=rerank_k)
        return reranked_result[:top_k]
    


class ConditionalRetriever(RrfRretriever):
    def __init__(self, docs):
        super().__init__(docs)

    def vector_retrieve(self, query, k=50, characters=[], ):
        return [doc.page_content for doc in self.vector_store.similarity_search(query, k=k, filter=filter)]
