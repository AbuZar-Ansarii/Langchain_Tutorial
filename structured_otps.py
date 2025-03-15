
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

schema = [
    ResponseSchema(name='fact_1',description='fact 1 about the topic'),
    ResponseSchema(name='fact_2',description='fact 2 about the topic'),
    ResponseSchema(name='fact_3',description='fact 3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "giv 3 fact about the {topic}.\n{format_instruction}",
    input_variables = ['topic'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic':'black hole'})
print(result)