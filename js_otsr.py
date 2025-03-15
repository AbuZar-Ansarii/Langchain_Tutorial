from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
parser = JsonOutputParser()

template = PromptTemplate(
    template = "give the name ,age, work, and city of Batman .\n{format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
# print(prompt)
# result = model.invoke(prompt)
# print(result.content)
# final_result = parser.parse(result.content)
# print(final_result)
# print(final_result["name"])

chain = template | model | parser
result = chain.invoke({})
print(result)