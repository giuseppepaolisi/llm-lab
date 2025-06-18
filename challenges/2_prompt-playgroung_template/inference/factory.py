from inference.openai_model import OpenAIModel
from inference.ollama_model import OllamaModel
from inference.base import BaseLLM

def get_llm(model_type: str) -> BaseLLM:
    if model_type.startswith("gpt-"):
        return OpenAIModel(model_name=model_type)
    elif any(model_type.startswith(prefix) for prefix in ["deepseek", "llama", "mistral", "mixtral"]):
        return OllamaModel(model_name=model_type)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
