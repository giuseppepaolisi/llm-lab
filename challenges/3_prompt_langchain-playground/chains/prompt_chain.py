from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pathlib import Path

def load_prompt(expert: str) -> str:
    """Load prompt template from text file"""
    base_dir = Path(__file__).parent.parent
    template_path = base_dir / "prompts" / f"{expert}.txt"
    
    if not template_path.exists():
        available = [f.stem for f in template_path.parent.glob("*.txt")]
        raise ValueError(
            f"Template per esperto '{expert}' non trovato.\n"
            f"Available templates: {', '.join(available)}"
        )
    
    return template_path.read_text(encoding="utf-8")

def build_chain(llm, template_str: str) -> LLMChain:
    """Build LLM chain from template string"""
    prompt = PromptTemplate.from_template(template_str)
    return LLMChain(llm=llm, prompt=prompt)