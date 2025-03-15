from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()
from langchain_core.prompts import PromptTemplate

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )
#
# model =ChatHuggingFace(llm=llm)


# 1 prompt
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
template1 = PromptTemplate(
    template='write a report on {topic}',
    input_variable = ['topic']
)

# 2 prompt
template2 = PromptTemplate(
    template='write 5 line summary on the following text.\n {text}',
    input_variable=['text']
)

prompt1 = template1.invoke({'topic':'attention all you need paper'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})

result1 = model.invoke(prompt2)

print(result1.content)