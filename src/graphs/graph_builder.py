from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from src.llms.groq_llm import GroqLLM
from src.states.blog_state import BlogState
from src.nodes.blog_node import BlogNode

class GraphBuilder:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.graph = StateGraph(BlogState)

    def build_topic_graph(self):

        blog_node = BlogNode(self.llm)

        self.graph.add_node('title_creation', blog_node.title_creation)
        self.graph.add_node('content_generation', blog_node.content_generation)

        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)

        return self.graph
    
    def setup_graph(self, usecase: str):
        if usecase == "topic":
            self.build_topic_graph()

        return self.graph.compile()


# llm=GroqLLM().get_llm()

# graph_builder=GraphBuilder(llm)
# graph=graph_builder.build_language_graph().compile()