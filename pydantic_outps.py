from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from sympy.physics.units import temperature

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro',temperature=1)


class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age : int = Field(description='Age is always greater than 18')
    city : str = Field(description='City of the person')

parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = 'generate a name ,age and city of a fictional {place} person.\n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction':parser.get_format_instructions()}

)

chain = template | model | parser
result = chain.invoke({'place':'Arab'})
print(result)