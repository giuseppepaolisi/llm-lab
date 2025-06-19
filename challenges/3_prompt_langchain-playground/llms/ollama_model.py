from langchain.chat_models import ChatOllama
from llms.base import LLMWrapper

class OllamaModel(LLMWrapper):
    def __init__(self, model="deepseek-r1:8b"):
        self.model = model

    def get_model(self):
        return ChatOllama(model=self.model)
