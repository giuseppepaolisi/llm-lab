import requests
from inference.base import BaseLLM

class OllamaModel(BaseLLM):
    def __init__(self, model_name="deepseek"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, temperature=0.7, max_tokens=512) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
