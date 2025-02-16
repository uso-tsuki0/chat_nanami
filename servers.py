from nanami_v1 import AgentRagConversation
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from langserve import add_routes
from parser import BaseParser
from splitter import LangChainTextSplitter
from retriever import RrfRretriever
import os
import json
from langchain_openai import ChatOpenAI
import os

import asyncio
import threading
from collections import deque
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from langserve import RemoteRunnable
from langchain.schema import HumanMessage

class ChatRequest(BaseModel):
        input: str


def run_llm_server():
    app = FastAPI(
        title="LangChain Streaming Chat",
        version="1.0",
        description="A FastAPI server with streaming response using LangChain and LangServe",
    )


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
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )
    retriever = RrfRretriever(docs=chunks)
    conversation = AgentRagConversation(llm, user_id='1123', retriever=retriever, memory_max_step=1)
    conversation.stream_server()



# 创建 FastAPI 应用
app = FastAPI()

# 定义全局队列：输入队列和回复队列
input_queue = deque()
response_queue = deque()

# 指向 LLM Server 的端点（假设 LLM Server 已在 8000 端口启动）
rag_agent = RemoteRunnable("http://localhost:8000/chat")

# 定义 Pydantic 模型，供用户提交输入
class MessageInput(BaseModel):
    message: str

@app.post("/push_input")
def push_input(msg: MessageInput):
    """
    将用户输入放入输入队列。
    """
    input_queue.append(msg.message)
    return {"status": "input received"}

@app.get("/pop_message")
def pop_message():
    """
    弹出并返回回复队列中最早的一条消息；若队列为空则返回 None。
    前端可持续轮询此接口来获取 LLM 回复。
    """
    if response_queue:
        return {"message": response_queue.popleft()}
    else:
        return {"message": None}

def process_input_queue():
    """
    后台线程：每隔一秒检查输入队列，
    如果有用户输入，则调用异步函数将输入发送到 LLM Server，
    并将 LLM 返回的回复存入回复队列。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        if input_queue:
            user_msg = input_queue.popleft()
            try:
                loop.run_until_complete(send_to_llm(user_msg))
            except Exception as e:
                print("Error sending to LLM:", e)
        loop.run_until_complete(asyncio.sleep(1))

async def send_to_llm(user_message: str):
    """
    使用 rag_agent.astream 发送消息到 LLM Server，
    分段获取回复，并将每个回复内容放入回复队列。
    """
    prompt = {
        "messages": [HumanMessage(content=user_message)],
        "sys_messages": [],
        "question": "",
        "retrieval_cache": [],
        "analysis": "",
        "retry_count": 0
    }
    async for resp in rag_agent.astream(prompt):
        # 根据实际返回结构，此处假设回复存储在键 "messages" 内
        for node, node_msg in resp.items():
            for m in node_msg.get("messages", []):
                print("Received message:", m.content)
                response_queue.append(m.content)

def run_msg_server(host="0.0.0.0", port=9000):
    """
    启动消息中台服务：
      - 启动后台线程处理输入队列；
      - 启动 FastAPI 服务器监听指定端口。
    """
    threading.Thread(target=process_input_queue, daemon=True).start()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    t1 = threading.Thread(target=run_llm_server)
    t2 = threading.Thread(target=run_msg_server)
    t1.start()
    t2.start()
    t1.join()
    t2.join()