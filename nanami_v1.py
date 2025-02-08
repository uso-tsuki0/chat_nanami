import json
import os
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, SystemMessage
import bs4
from langchain import hub
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
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
import getpass
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field, PrivateAttr
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import json
from collections import deque
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot




from parser import BaseParser
from splitter import LangChainTextSplitter
from retriever import RrfRretriever




        
    

class State(TypedDict):
    messages: Annotated[list, add_messages]


class AdditionInput(BaseModel):
    a: int = Field(..., description="The first integer")
    b: int = Field(..., description="The second integer")


class AdditionTool(BaseTool):
    name: str = "addition_tool"
    description: str = "A tool that adds two numbers together."
    args_schema: Type[BaseModel] = AdditionInput

    def _run(self, a: int, b: int) -> int:
        return a + b

    async def _arun(self, a: int, b: int) -> int:
        return a + b
    

class RetrieveInput(BaseModel):
    query: str = Field(..., description="The query to search for.")


class RetrieveTool(BaseTool):
    """
    A langchain supported tool that retrieves search results.
    """
    name: str = "retrieve_tool"
    description: str = "A tool that retrieves search results."
    args_schema: Type[BaseModel] = RetrieveInput
    return_direct: bool = False

    _retriever: Any = PrivateAttr()

    def __init__(self, retriever):
        super().__init__()
        self._retriever = retriever

    def formatting_results(self, query, docs):
        map = {}
        map["query"] = query
        for i, doc in enumerate(docs):
            map[f"doc_{i}"] = doc
        return json.dumps(map, ensure_ascii=False)

    def _run(self, query: str) -> str:
        """
        Retrieve search results for a given query.
        Parameters
        ----------
        query : str
            The query to search for.
        Returns
        -------
        str
            A JSON-formatted string containing query and search results. The query is stored under the key "query", and the search results are stored under keys "doc_0", "doc_1", etc.

        """
        results = self._retriever.retrieve(query)
        docs = [doc['content'] for doc in results]
        return self.formatting_results(query, docs)

    async def _arun(self, query: str) -> str:
        results = await asyncio.to_thread(self._retriever.retrieve, query)
        docs = [doc['content'] for doc in results]
        return self.formatting_results(query, docs)
    



class BaseConversation():
    """A base class for a conversation agent with no memory state."""
    def __init__(self, model, user_id, retriever):
        self.llm = model
        self.retriever = retriever
        self.add_tools()
        self.build_graph()
        self.config = {"configurable": {"thread_id": f"{user_id}"}}

    def add_tools(self):
        self.tools = [RetrieveTool(self.retriever), TavilySearchResults(k=5)]
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def build_graph(self):
        graph_builder = StateGraph(NonMemoryState)
        tool_node = ToolNode(tools=self.tools)
        #node
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", tool_node)
        #edge
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        self.graph = graph_builder.compile()

    def chatbot(self, state: State):
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def input(self, user_input):
        events = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            self.config, stream_mode="values"
        )
        return events
    

class NonMemoryState(TypedDict):
    """The state of the agent."""
    messages: List[BaseMessage]
    question: str
    analysis: str


class ProblemAnalysis(BaseModel):
    """问题分析结构化输出"""
    problem_decomposition: str = Field(description="问题拆解")
    information_requirements: str = Field(description="信息需求")
    queries: str = Field(
        description="查询内容"
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class CustomMemorySaver(MemorySaver):
    def save(self, state: State, thread_id: str):
        self.states[thread_id] = state


class AgentRagConversation(BaseConversation):
    def __init__(self, model, user_id, retriever, memory_max_step=1):
        self.checkpointer = CustomMemorySaver()
        super().__init__(model, user_id, retriever)

    def add_tools(self):
        self.local_rag_tools = [RetrieveTool(self.retriever)]
        self.online_rearch_tools = [TavilySearchResults(k=5)]
        self.tool_map = {tool.name: tool for tool in self.local_rag_tools}
        self.tool_map.update({tool.name: tool for tool in self.online_rearch_tools})
        self.llm_with_local_rag_tools = self.llm.bind_tools(self.local_rag_tools)
     
    
    def analyzer_v1(self, state: NonMemoryState):
        prompt_template = ChatPromptTemplate.from_messages([
        ('system', """\
        # 角色
        你是一名专业的问题分析师，能够通过系统性思考将复杂问题拆解为可检索的子问题，并提出精确的查询需求。

        # 处理流程
        1. 问题拆解
        - 理解问题的核心意图和背景。
        - 将问题拆解为多个子问题。
        - 明确子问题与主问题的逻辑关系。
        2. 信息需求
        - 明确每个子问题需要哪些具体信息。
        3. 查询内容
        - 根据信息需求，提出具体的查询语句
         
        # 输出规范
        - 确保子问题之间没有过多的重叠。
        - 保持逻辑严谨
        """),
        
        ('human', '{human_input}')
        ])
        cot_chain = prompt_template | self.llm.with_structured_output(ProblemAnalysis)
        question = state["messages"][-1].content
        response = cot_chain.invoke({'human_input': question})
        info_requirements = AIMessage(content=response.information_requirements)
        analysis = response.problem_decomposition
        queries = AIMessage(content=response.queries)
        return {
            "messages": [queries],
            "question": question,
            "analysis": analysis,
        }


    def router(self, state: NonMemoryState):
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', """
            You are a professional tool invocation expert. Your task is to generate precise tool invocation instructions based on the queries.
            """),
            ('human', 'queries: {queries}')
        ])
        queries = state["messages"][-1].content
        question = state["question"]
        tools_chain = prompt_template | self.llm_with_local_rag_tools
        response = tools_chain.invoke({'queries': queries})
        return {
            "messages": [response],
            "question": question,
            "analysis": state["analysis"]
        }
    

    def tool_node(self, state: NonMemoryState):
        outputs = []
        question = state["question"]
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tool_map[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            "messages": outputs,
            "question": question,
            "analysis": state["analysis"]
        }
    

    def formatting_results(self, query, docs):
        map = {}
        map["query"] = query
        for i, doc in enumerate(docs):
            map[f"doc_{i}"] = doc
        return json.dumps(map, ensure_ascii=False)
    

    def document_filter(self, state: NonMemoryState):
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """\
            You are a document evaluation expert responsible for assessing the relevance of retrieved documents to user questions.\
            Your task is to evaluate each document based on the following criteria:\
            - If the document contains keywords or semantic information related to the user question or the query, evaluate it as "true".\
            - If the document is unrelated to the user question or contains inaccurate information, evaluate it as "false".\
            Ensure that your evaluation is based on the content of the document and its relevance to the user question.\
        """
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question} \n \n query: {subquery}"),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        response = []
        question = state["question"]
        for map_str in state["messages"]:
            map = json.loads(json.loads(map_str.content))
            query = map["query"]
            documents = []
            for key in map:
                if key.startswith("doc_"):
                    documents.append(map[key])
            relative_docs = []
            for doc in documents:
                doc_response = retrieval_grader.invoke(
                    {"document": doc, "question": question, "subquery": query}
                )
                if doc_response.binary_score == "yes":
                    relative_docs.append(doc)
            response_str = (self.formatting_results(query, relative_docs))
            response.append(AIMessage(content=response_str))
        return {
            "messages": response,
            "question": question,
            "analysis": state["analysis"]
        }
    

    def online_search(self, state: NonMemoryState):
        """
        Online search for additional information.
        If there is no document for the query after filtering, search for additional information online.
        If there are documents, skip this step.
        """
        for map in state["messages"]:
            map = json.loads(map.content)
            query = map["query"]
            if len(map) == 1:
                search_results = self.tool_map["TavilySearchResults"].invoke({"query": query})
                return {"messages": [AIMessage(json.dumps(search_results))], "question": state["question"], "analysis": state["analysis"]}


    def summarizer(self, state: NonMemoryState):
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', """You are an expert at using retrieved documents to summarize an answer to the query."""),
            ('human', "Query: \n \n {query} \n \n Documents: \n \n {documents}") 
        ])
        summary_chain = prompt_template | self.llm
        response = []
        for map in state['messages']:
            map = json.loads(map.content)
            query = map["query"]
            documents = []
            for key in map:
                if key.startswith("doc_"):
                    documents.append(map[key])
            documents = "\n\n".join(documents)
            summary = summary_chain.invoke({'query': query, 'documents': documents}).content
            content = f"Query: {query} \n Summary: {summary}"
            response.append(content)
        return {"messages": [AIMessage('\n'.join(response))], "question": state["question"], "analysis": state["analysis"]}


    def answer(self, state: NonMemoryState):
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', """You are an expert at reasoning with retrieved documents to answer complex questions."""),
            ('human', "Question: \n \n {question} \n \n Question Decomposition: \n \n {question_analysis} \n \n Information Retrieved: \n \n {documents}")
        ])
        question = state["question"]
        question_analysis = state["analysis"]
        documents = state["messages"][-1].content
        answer_chain = prompt_template | self.llm
        response = answer_chain.invoke({'question': question, 'question_analysis': question_analysis, 'documents': documents})
        return {"messages": [HumanMessage(question), response]}
    

    def build_graph(self):
        graph_builder = StateGraph(NonMemoryState)
        #node
        graph_builder.add_node("analyzer", self.analyzer_v1)
        graph_builder.add_node("router", self.router)
        graph_builder.add_node("tools", self.tool_node)
        graph_builder.add_node("filter", self.document_filter)
        graph_builder.add_node("summarizer", self.summarizer)
        graph_builder.add_node("answer", self.answer)
        #edge
        graph_builder.add_edge(START, "analyzer")
        graph_builder.add_edge("analyzer", "router")
        graph_builder.add_edge("router", "tools")
        graph_builder.add_edge("tools", "filter")
        graph_builder.add_edge("filter", "summarizer")
        graph_builder.add_edge("summarizer", "answer")
        self.graph = graph_builder.compile(checkpointer=self.checkpointer)

    
    def input(self, user_input):
        init_state = {
            "messages": [HumanMessage(content=user_input)],
        } 
        events = self.graph.invoke(
            init_state, self.config, stream_mode="values"
        )
        return events



    


class MultiFuncAgent(BaseConversation):
    def __init__(self, model, user_id, retriever, max_memory_tokens=4096):
        self.checkpointer = CustomMemorySaver()
        super().__init__(model, user_id, retriever)

    def add_tools(self):
        self.tools = [RetrieveTool(self.retriever)]
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def build_graph(self):
        graph_builder = StateGraph(NonMemoryState)
        #node
        graph_builder.add_node("analyzer", self.analyzer_v1)
        graph_builder.add_node("router", self.router)
        graph_builder.add_node("tools", self.tool_node)
        graph_builder.add_node("filter", self.document_filter)
        graph_builder.add_node("summarizer", self.summarizer)
        graph_builder.add_node("answer", self.answer)
        #edge
        graph_builder.add_edge(START, "analyzer")
        graph_builder.add_edge("analyzer", "router")
        graph_builder.add_edge("router", "tools")
        graph_builder.add_edge("tools", "filter")
        graph_builder.add_edge("filter", "summarizer")
        graph_builder.add_edge("summarizer", "answer")
        self.graph = graph_builder.compile(checkpointer=self.checkpointer)
    

    def input(self, user_input):
        init_state = {
            "messages": [HumanMessage(content=user_input)],
        } 
        events = self.graph.invoke(
            init_state, self.config, stream_mode="values"
        )
        return events


    

class AriharaNanami():
    def __init__(self):
        pass


if __name__ == '__main__':
    file_path = 'data/jsons_nanami/002.それは、まさに貧乳だったver.108.ks.json'
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    test_parser = BaseParser()
    file_path = 'data/jsons_nanami/002.それは、まさに貧乳だったver.108.ks.json'
    result = test_parser.parse(file_path)
    result = test_parser.merge_text(result)
    splitter = LangChainTextSplitter(chunk_size=512, chunk_overlap=128, add_start_index=True)
    chunks = splitter.split(result)
    api_map = json.load(open('api_keys.json', 'r', encoding='utf-8'))
    for key in api_map:
        os.environ[key] = api_map[key]
    """llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )"""
    """llm = Moonshot(
        model="moonshot-v1-128k",
        temperature=0,
    )"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )


    retriever = RrfRretriever(docs=chunks)
    conversation = AgentRagConversation(llm, user_id='1123', retriever=retriever, memory_max_step=1)
    conversation.input('请推荐一些学校食堂里可以作为午饭的食物')