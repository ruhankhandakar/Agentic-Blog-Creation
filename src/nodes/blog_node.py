from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.states.blog_state import BlogState, Blog

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
    
    def route(self, state: BlogState):
        return {"current_language": state['current_language'] }
    
    def translation(self, state: BlogState):
        translation_prompt="""
        Translate the following content into {current_language}.
        - Maintain the original tone, style, and formatting.
        - Adapt cultural references and idioms to be appropriate for {current_language}.

        ORIGINAL CONTENT:
        {blog_content}
        """

        blog_content=state["blog"]["content"]
        messages = [
            HumanMessage(
                translation_prompt.format(
                    current_language=state["current_language"],
                    blog_content=blog_content
                )
            )
        ]

        translated_content = self.llm.with_structured_output(Blog).invoke(messages)

        return {"blog": {"content": translated_content}}

    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """
        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french": 
            return "french"
        else:
            return state['current_language']