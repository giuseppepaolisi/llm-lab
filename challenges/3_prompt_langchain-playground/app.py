import streamlit as st
from llms.openai_model import OpenAIModel
from llms.ollama_model import OllamaModel
from chains.prompt_chain import build_chain, load_prompt

st.title("💬 LangChain Prompt Playground")

question = st.text_area("Domanda")
expert = st.selectbox("Tipo di esperto", ["python_helper", "devops_helper"])
provider = st.selectbox("LLM Provider", ["openai", "deepseek", "ollama"])

if st.button("Genera risposta"):
    if provider == "openai":
        llm = OpenAIModel().get_model()
    else:
        llm = OllamaModel().get_model()

    # Correctly load prompt template
    template_str = load_prompt(expert)
    chain = build_chain(llm, template_str)
    response = chain.run({"question": question})

    st.subheader("🧠 Risposta")
    st.write(response)