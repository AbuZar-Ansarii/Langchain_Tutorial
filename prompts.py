from langchain_google_genai import  ChatGoogleGenerativeAI
from dotenv import  load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.title("PAPER SUMMARIZE TOOL")
paper_input = st.selectbox("Select Paper",["Attention all you need","Transformer","Evolution"])
style_input = st.selectbox("Select Style",["Short","Medium","Length"])
paper_code = st.selectbox("select code",["code heavy","theory based","math heavy"])

user_prompt = PromptTemplate(template = '''explain me{paper_input} in this {style_input}format with this {paper_code} if avalilable .else print this paper code is not avalilable''',
                             input_variables = ['paper_input','style_input','paper_code'])
input = user_prompt.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "paper_code":paper_code

})

if st.button("Generate"):
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
    result = model.invoke(input)
    st.text(result.content)

