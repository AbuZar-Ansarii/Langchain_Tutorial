from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv

from ch_his import history

# history = []
# load_dotenv()
# while True:
#     user_input = input("Ask Me...  ")
#     history.append(HumanMessage(user_input))
#     if user_input.lower() == "exit":
#         break
#     else:
#         model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
#         result = model.invoke(history)
#         history.append(AIMessage(result.content))
#         print(result.content)


load_dotenv()

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

chat_history = []

# Load chat history from file
try:
    with open("chat_history.txt", "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            human_msg = lines[i].strip().replace("Human: ", "")
            ai_msg = lines[i + 1].strip().replace("AI: ", "")
            chat_history.append(HumanMessage(content=human_msg))
            chat_history.append(AIMessage(content=ai_msg))
except FileNotFoundError:
    print("Chat history file not found. Starting with an empty history.")
except IndexError:
    print("Chat history file is improperly formatted.")

while True:
    user_input = input("Ask Me...  ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
    chain = chat_template | model
    result = chain.invoke({'chat_history': chat_history, 'user_input': user_input})

    chat_history.append(AIMessage(content=result.content))
    print(result.content)

    # Save chat history back to file
    with open("chat_history.txt", "w") as f:
        for message in chat_history:
            if isinstance(message, HumanMessage):
                f.write(f"Human: {message.content}\n")
            elif isinstance(message, AIMessage):
                f.write(f"AI: {message.content}\n")