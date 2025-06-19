import os
from langchain.chat_models import ChatOpenAI
from llms.base import LLMWrapper

class OpenAIModel(LLMWrapper):
    def get_model(self):
        return ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
