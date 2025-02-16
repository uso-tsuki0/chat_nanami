import asyncio
from langserve import RemoteRunnable
from langchain.schema import HumanMessage

# 连接到服务端点
rag_agent = RemoteRunnable("http://localhost:8000/chat")

async def main():
    prompt = {
        "messages": [HumanMessage(content="请介绍一下langchain")],
        "sys_messages": [],
        "question": "",
        "retrieval_cache": [],
        "analysis": "",
        "retry_count": 0
    }

    # 使用 astream 返回异步流式结果
    async for msg in rag_agent.astream(prompt):
        print(msg, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
