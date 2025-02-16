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




class SlidingWindowSet:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
    
    def add(self, item):
        if len(self.window) >= self.window_size:
            removed_item = self.window.popleft()
        if ((not self.window) or (item != self.window[-1])) and (item != 'user throught'):
            self.window.append(item)
    
    def get_set(self):
        return set(self.window)
    
    def get_window(self):
        return list(self.window)




class BaseSplitter:
    def __init__(self, chunk_size=512, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size


    def split_text_for_same_stage(self, conversations):
        '''
        Split the conversation into chunks of text
        conversations: list of dict
        '''
        result = []
        current_chunk = []
        current_length = 0
        for i, turn in enumerate(conversations):
            character = turn['character']
            text_cn = turn['text_cn']
            turn_text = f"{text_cn}\n"
            turn_length = len(turn_text)
            if current_length + turn_length > self.chunk_size:
                result.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(turn_text)
            current_length += turn_length
        if current_chunk:
            result.append(" ".join(current_chunk))
        return result
    

    def split_text(self, conversations):
        '''
        Split the conversation into chunks of text
        conversations: list of dict
        '''
        curr_stage = None
        result = []
        curr_conversation = []
        for map in conversations:
            if map['stage'] != curr_stage:
                if curr_conversation:
                    result.extend(self.split_text_for_same_stage(curr_conversation))
                curr_conversation = []
                curr_stage = map['stage']
            curr_conversation.append(map)
        if curr_conversation:
            result.extend(self.split_text_for_same_stage(curr_conversation))
        return result
    

class LangChainTextSplitter(BaseSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200, add_start_index=True):
        super().__init__(chunk_size=chunk_size)
        self.chunk_overlap = chunk_overlap
        self.langchainsplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
        )

    def transform(self, conversations, stage):
        texts = []
        character_set = set()
        for turn in conversations:
            character = turn['character']
            text_cn = turn['text_cn']
            text = f"{text_cn}\n"
            texts.append(text)
            character_set.add(character)
        doc = Document(
            page_content = ''.join(texts),
            metadata = {'stage': stage, 'characters': ', '.join(list(character_set))}
        )
        return doc
    
    def split(self, conversations):
        # split to bulks when stage changes
        # in each bulk, split to chunks according to chunk_size
        print('splitting text')
        curr_stage = None
        docs = []
        curr_conversation = []
        for map in conversations:
            if map['stage'] != curr_stage:
                if curr_conversation:
                    docs.append(self.transform(curr_conversation, curr_stage))
                curr_conversation = []
                curr_stage = map['stage']
            curr_conversation.append(map)
        if curr_conversation:
            docs.append(self.transform(curr_conversation, curr_stage))
        return self.langchainsplitter.split_documents(docs)
    

class CharacterLruSplitter(LangChainTextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200, add_start_index=True, window_size=4):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
        self.window_size = window_size
    
    def transform(self, conversations, stage, characters):
        texts = []
        for turn in conversations:
            text_cn = turn['text_cn']
            text = f"{text_cn}\n"
            texts.append(text)
        doc = Document(
            page_content = ''.join(texts),
            metadata = {'stage': stage, 'characters': ', '.join(characters)}
        )
        return doc
    
    def split(self, conversations):
        # split to bulks when stage and most current characters changes
        # in each bulk, split to chunks according to chunk_size
        print('splitting text')
        curr_stage = None
        curr_character_set = SlidingWindowSet(self.window_size)
        character_set = SlidingWindowSet(self.window_size)
        docs = []
        curr_conversation = []
        for map in conversations:
            character_set.add(map['character'])
            if map['stage'] != curr_stage:
                if curr_conversation:
                    docs.append(self.transform(curr_conversation, curr_stage, sorted(list(curr_character_set.get_set()))))
                curr_conversation = []
                curr_character_set = SlidingWindowSet(self.window_size)
                character_set = SlidingWindowSet(self.window_size)
                character_set.add(map['character'])
            elif (character_set.get_set() != curr_character_set.get_set()) and (len(curr_character_set.get_window()) >= self.window_size):
                if curr_conversation:
                    docs.append(self.transform(curr_conversation, curr_stage, sorted(list(curr_character_set.get_set()))))
                curr_conversation = []
            curr_conversation.append(map)
            curr_stage = map['stage']
            curr_character_set.add(map['character'])
        if curr_conversation:
            docs.append(self.transform(curr_conversation, curr_stage, sorted(list(curr_character_set.get_set()))))
        return self.langchainsplitter.split_documents(docs)
    

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





class BaseParser:
    def __init__(self):
        pass

    def find_tuple(self, data_list, key):
        for tuple in data_list:
            if isinstance(tuple, list) and tuple[0] == key:
                return tuple

    def parse_diaologue(self, map):
        diaologue_tuple = map[1]
        character = diaologue_tuple[2][0]
        if character is None:
            character = 'user throught'
        text_jp = diaologue_tuple[0][1]
        text_en = diaologue_tuple[1][1]
        text_cn = diaologue_tuple[2][1]
        is_dialogue = (text_cn[0]=='「' and text_cn[-1]=='」') or (text_cn[0]=='『' and text_cn[-1]=='』')
        if is_dialogue:
            text_cn = text_cn[1:-1]
            text_jp = text_jp[1:-1]
        return character, text_jp, text_en, text_cn, is_dialogue
    
    def parse_stage(self, map):
        stage_tuple = self.find_tuple(map[4]['data'], 'stage')
        show_mode = stage_tuple[2].get('showmode', None)
        if show_mode and show_mode != 0:
            return stage_tuple[2].get('redraw', {}).get('imageFile', {}).get('file', None)
        else:
            return None
        
    def parse_face(self, map):
        face_tuple = self.find_tuple(map[4]['data'], 'face')
        if not face_tuple:
            return None
        show_mode = face_tuple[2].get('showmode', None)
        if show_mode and show_mode != 0:
            return face_tuple[2].get('redraw', {}).get('imageFile', {}).get('options', {}).get('face', None)
        else:
            return None
        
    
    def parse(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = []
        for i, scene in enumerate(data["scenes"]):
            if 'texts' in scene:
                for j, map in enumerate(scene["texts"]):
                    try:
                        character, text_jp, text_en, text_cn, is_diaologe = self.parse_diaologue(map)
                        stage = self.parse_stage(map)
                        face = self.parse_face(map)
                        result.append({
                            'character': character,
                            'text_jp': text_jp,
                            'text_en': text_en,
                            'text_cn': text_cn,
                            'is_dialogue': is_diaologe,
                            'stage': stage,
                            'face': face
                        })
                    except Exception as e:
                        print(f'Error at scene {i} map {j}')
                        print(e)
        return result
    

    def merge_text(self, map_list):
        result = []
        text_cache = {
            'text_jp': '',
            'text_en': '',
            'text_cn': ''
        }
        curr_character = None
        curr_is_dialogue = None
        curr_face = None
        curr_stage = None
        for map in map_list:
            if map['character'] == curr_character and map['is_dialogue'] == curr_is_dialogue and map['face'] == curr_face and map['stage'] == curr_stage:
                if text_cache['text_jp'][-1] in ['。', '！', '？', '…']:
                    text_cache['text_jp'] += map['text_jp']
                else:
                    text_cache['text_jp'] += ('。' + map['text_jp'])
                text_cache['text_en'] += (' ' + map['text_en'])
                if text_cache['text_cn'][-1] in ['。', '！', '？', '…']:
                    text_cache['text_cn'] += map['text_cn']
                else:
                    text_cache['text_cn'] += ('。' + map['text_cn'])
            else:
                if curr_character:
                    result.append({
                        'character': curr_character,
                        'text_jp': text_cache['text_jp'],
                        'text_en': text_cache['text_en'],
                        'text_cn': text_cache['text_cn'],
                        'is_dialogue': curr_is_dialogue,
                        'face': curr_face,
                        'stage': curr_stage
                    })
                curr_character = map['character']
                curr_is_dialogue = map['is_dialogue']
                curr_face = map['face']
                curr_stage = map['stage']
                text_cache = {
                    'text_jp': map['text_jp'],
                    'text_en': map['text_en'],
                    'text_cn': map['text_cn']
                }
        return result[1:]
    

    def filter(self, map_list, user_name='晓', character_name='七海'):
        pass
    

    def convert(self, map_list, user_name, character_name='七海'):
        retriever = BaseRetriever()
        system_prompt = f'I want you to act as an actress in this story. You will play the role of {character_name}. {retriever.get_intro(character_name)}'
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
    

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
        graph_builder = StateGraph(State)
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
    

class AgentState(TypedDict):
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
        self.tools = [RetrieveTool(self.retriever)]
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm_with_tools = self.llm.bind_tools(self.tools)
     
    
    def analyzer_v1(self, state: AgentState):
        prompt_template = ChatPromptTemplate.from_messages([
        ('system', """\
        # 角色
        你是一名专业的问题分析师，能够通过系统性思考将复杂问题拆解为可检索的子问题，并提出精确的查询需求。

        # 处理流程
        1. 问题拆解
        - 理解问题的核心意图和背景。
        - 将问题拆解为多个子问题。
        2. 信息需求
        - 明确每个子问题需要哪些具体信息。
        3. 查询内容
        - 根据信息需求，提出具体的查询语句。
         
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


    def router(self, state: AgentState):
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', """You are an expert at querying with tools. You can make multiple tool calls."""),
            ('human', 'queries: {queries}')
        ])
        queries = state["messages"][-1].content
        question = state["question"]
        tools_chain = prompt_template | self.llm_with_tools
        response = tools_chain.invoke({'queries': queries})
        return {
            "messages": [response],
            "question": question,
            "analysis": state["analysis"]
        }
    

    def tool_node(self, state: AgentState):
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
    

    def document_filter(self, state: AgentState):
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question or the query, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
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


    def summarizer(self, state: AgentState):
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


    def answer(self, state: AgentState):
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
        graph_builder = StateGraph(AgentState)
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

    


if __name__ == '__main__':
    file_path = 'data/jsons/002.それは、まさに貧乳だったver.108.ks.json'
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    test_parser = BaseParser()
    file_path = 'data/jsons/002.それは、まさに貧乳だったver.108.ks.json'
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