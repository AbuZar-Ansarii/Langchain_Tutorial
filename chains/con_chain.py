from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import Field,BaseModel
from typing import Literal

load_dotenv()

parser = StrOutputParser()
model1 = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

class Feedback(BaseModel):
    sentiment:Literal['positive','negative'] = Field(description="give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)


prompt1 = PromptTemplate(
    template = 'classify the sentiment of the  following feedback text into positive and negative\n{feedback}\n {format_instruction}',
    input_variables = ["feedback"],
    partial_variables = {'format_instruction':parser2.get_format_instructions()}
)
c_chain = prompt1 | model1 | parser2
prompt2 = PromptTemplate(
    template = 'generate a appropriate response to this positive feedback \n{feedback}.',
    input_variables = ['feedback']
)
prompt3 = PromptTemplate(
    template = 'generate a appropriate response to this negative feedback \n{feedback}.',
    input_variables = ['feedback']
)

b_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',prompt2 | model1 | parser),
    (lambda x:x.sentiment == 'negative',prompt3 |model1 | parser ),
    RunnableLambda(lambda x:"could not find sentiment")
)
chain = c_chain | b_chain

result = chain.invoke({'feedback':'i am coming to borderland'})
print(result)