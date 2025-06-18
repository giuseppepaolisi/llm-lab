import os
from pathlib import Path

class PromptTemplateLoader:
    def __init__(self, expert: str, templates_dir="templates"):
        current_dir = Path(__file__).parent
        templates_path = current_dir / templates_dir
        self.template_path = templates_path / f"{expert}.txt"
        
        print(f"Looking for template at: {self.template_path}")
        
        if not self.template_path.exists():
            available = [f.stem for f in templates_path.glob("*.txt")]
            raise ValueError(
                f"Template per esperto '{expert}' non trovato.\n"
                f"Available templates: {', '.join(available)}"
            )

    def render(self, user_input: str) -> str:
        with open(self.template_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template.replace("{input}", user_input)