# from typing import TypedDict
#
# class Person(TypedDict):
#     name : str
#     age : int
# new_person: Person = {"name":"jack","age":56}
#
# print(new_person)


# from typing import TypedDict,Annotated
# # from pydantic import BaseModel
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
#
# load_dotenv()
#
# # load model
# model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
#
# # schema
#
# class Review(TypedDict):
#     summary : Annotated[str,"a brief summary of this review "]
#     sentiment :Annotated[str,"return sentiment"]
#     words : int
#
# structured_model = model.with_structured_output(Review)
#
# result = structured_model.invoke("I feel like I accomplished something here even though now in my reality seemingly all I did here was suffer and die...")
# print(result)







