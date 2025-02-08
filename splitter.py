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
    