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
    
    def build_language_graph(self):
        """Build the graph for blog generation with inputs topic and language
        """
        blog_node = BlogNode(self.llm)

        self.graph.add_node('title_creation', blog_node.title_creation)
        self.graph.add_node('content_generation', blog_node.content_generation)
        self.graph.add_node('hindi_translation', lambda state: blog_node.translation({**state, "current_language": "hindi"}))
        self.graph.add_node('french_translation', lambda state: blog_node.translation({**state, "current_language": "french"}))
        self.graph.add_node('route', blog_node.route)

        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "route")
        self.graph.add_conditional_edges(
            "route", 
            blog_node.route_decision, 
            {"hindi": "hindi_translation", "french": "french_translation"}
        )
        self.graph.add_edge("hindi_translation", END)
        self.graph.add_edge("french_translation", END)

        return self.graph
    
    def setup_graph(self, usecase: str):
        if usecase == "topic":
            self.build_topic_graph()
        if usecase == "language":
            self.build_language_graph()

        return self.graph.compile()


# Below code is for langsmith langgraph studio
llm=GroqLLM().get_llm()

graph_builder=GraphBuilder(llm)
graph=graph_builder.setup_graph(usecase="topic")