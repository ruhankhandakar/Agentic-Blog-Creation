from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv

from src.llms.groq_llm import GroqLLM
from src.graphs.graph_builder import GraphBuilder

load_dotenv()

os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

app=FastAPI()

@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    
    topic= data.get("topic","")
    language = data.get("language", '')

    groqllm=GroqLLM()
    llm=groqllm.get_llm()

    graph_builder=GraphBuilder(llm)
    
    if topic:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})
        
    return {"data": state}
