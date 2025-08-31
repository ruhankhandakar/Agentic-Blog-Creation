from langchain_groq import ChatGroq

from src.states.blog_state import BlogState

class BlogNode:
    def __init__(self, llm: ChatGroq):
        self.llm = llm

    def title_creation(self, state: BlogState):
        """
        create the title for the blog
        """
        if "topic" in state and state["topic"]:
            prompt = """
                You are an expert blog content writer. Use Markdown formatting.
                Generate a blog title for the {topic}. This title should be creative and SEO friendly. And it should be short and concise. Max 2 lines
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {"blog":{"title": response.content}}
        
    def content_generation(self, state: BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """
                You are expert blog writer. Use Markdown formatting.
                Generate a detailed blog content with detailed breakdown for the {topic}. And the tile of the blog 
                is {title}.
            """
            system_message = system_prompt.format(topic=state["topic"], title=state['blog']['title'])
            response = self.llm.invoke(system_message)
            return {"blog": {"title": state['blog']['title'], "content": response.content}}