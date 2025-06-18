import os
import openai
from inference.base import BaseLLM
from dotenv import load_dotenv

load_dotenv()

openai.api_key=os.getenv("OPENAI_API_KEY")

class OpenAIModel(BaseLLM):
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate(self, prompt: str, temperature=0.7, max_tokens=512) -> str:
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
