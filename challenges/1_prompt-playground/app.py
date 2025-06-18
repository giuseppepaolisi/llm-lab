import streamlit as st
from inference.factory import get_llm

st.set_page_config(page_title="Prompt Playground", layout="wide")
st.title("🧠 Prompt Playground")

model = st.selectbox("Seleziona un modello:", ["gpt-3.5-turbo", "gpt-4", "mistral", "llama3", "gemma", "deepseek-r1:8b", "llama2", "qwen"])
prompt = st.text_area("Scrivi il prompt:", height=200)
temperature = st.slider("Creatività (temperature)", 0.0, 1.0, 0.7)
submit = st.button("Esegui")

if submit and prompt:
    llm = get_llm(model)
    with st.spinner("Chiamando il modello..."):
        response = llm.generate(prompt, temperature=temperature)
    st.subheader("Risposta:")
    st.write(response)
