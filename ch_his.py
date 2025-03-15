from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

g_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
user_query = input("Enter Your Query.. ")

history = [
    SystemMessage(content="your are a helpful assistant"),
    HumanMessage(content=user_query)
]

result = g_model.invoke(user_query)
history.append(AIMessage(result.content))
print(result.content)
print(history)