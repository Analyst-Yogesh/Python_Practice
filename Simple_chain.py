from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

prompt=PromptTemplate(
    template='Generate 1 interesting fact about {topic}',
    input_variables=["topic"]
)

llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text generation"
)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

chain= prompt | model | parser

# Streamlit UI
st.set_page_config(page_title="Fact Generator", page_icon="✨")

st.title("✨ Fact Generator App")
st.write("Enter a topic and get 5 interesting facts!")

# Input box
topic = st.text_input("Enter a topic:", "")

if st.button("Generate Facts"):
    if topic.strip():
        with st.spinner("Generating facts..."):
            result = chain.invoke({"topic": topic})
        st.subheader("Generated Facts:")
        st.write(result)
    else:
        st.warning("⚠️ Please enter a topic first!")