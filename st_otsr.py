
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# 1 prompt
template1 = PromptTemplate(
    template='write a report on {topic}',
    input_variable = ['topic']
)

# 2 prompt
template2 = PromptTemplate(
    template='write 5 line summary on the following text.\n {text}',
    input_variable=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke(({'topic':"attention is all you need paper"}))

print(result)