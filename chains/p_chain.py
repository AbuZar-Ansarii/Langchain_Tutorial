from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()


parser = StrOutputParser()
model1 = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
model2 = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
text = '''
Amazon Web Services, Inc. (AWS) is a subsidiary of Amazon that provides on-demand cloud computing platforms and APIs to individuals, companies, and governments, on a metered, pay-as-you-go basis. Clients will often use this in combination with autoscaling (a process that allows a client to use more computing in times of high application usage, and then scale down to reduce costs when there is less traffic). These cloud computing web services provide various services related to networking, compute, storage, middleware, IoT and other processing capacity, as well as software tools via AWS server farms. This frees clients from managing, scaling, and patching hardware and operating systems. One of the foundational services is Amazon Elastic Compute Cloud (EC2), which allows users to have at their disposal a virtual cluster of computers, with extremely high availability, which can be interacted with over the internet via REST APIs, a CLI or the AWS console. AWS's virtual computers emulate most of the attributes of a real computer, including hardware central processing units (CPUs) and graphics processing units (GPUs) for processing; local/RAM memory; hard-disk (HDD)/SSD storage; a choice of operating systems; networking; and pre-loaded application software such as web servers, databases, and customer relationship management (CRM).

AWS services are delivered to customers via a network of AWS server farms located throughout the world. Fees are based on a combination of usage (known as a "Pay-as-you-go" model), hardware, operating system, software, and networking features chosen by the subscriber requiring various degrees of availability, redundancy, security, and service options. Subscribers can pay for a single virtual AWS computer, a dedicated physical computer, or clusters of either.[7] Amazon provides select portions of security for subscribers (e.g. physical security of the data centers) while other aspects of security are the responsibility of the subscriber (e.g. account management, vulnerability scanning, patching). AWS operates from many global geographical regions, including seven in North America.[8]
'''
prompt1 = PromptTemplate(
    template = 'generate notes from given {text}',
    input_variables = ['text']
)
prompt2 = PromptTemplate(
    template = 'generate question and answers form given \n{text}',
    input_variables = ['text']
)
prompt3 = PromptTemplate(
    template = 'marge these two provided document into a single document , notes - {notes}, questions/answers - {quiz}.',
    input_variables = ['notes','quiz']
)

p_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser})

m_chain = prompt3 | model1 | parser

chain = p_chain | m_chain

result = chain.invoke({"text":text})
print(result)
print(chain.get_graph().draw_ascii())
