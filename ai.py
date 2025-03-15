from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st. title("GEMINI 1.5 CHAT BOT")
while True:
    query = st.text_area("Enter your query...")
    if query.lower() == "exit":
        break
    else:
        if st.button("Generate"):
            model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
            result = model.invoke(query)
            st.text(result.content)
