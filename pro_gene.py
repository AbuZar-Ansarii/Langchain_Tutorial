
from langchain_core.prompts import PromptTemplate




user_prompt = PromptTemplate(template = '''explain me{paper_input} in this {style_input}format with this {paper_code} if avalilable .else print this paper code is not avalilable''',
                             input_variables = ['paper_input','style_input','paper_code'],
                             validate_template = True)


user_prompt.save("template.json")
print("saved successfully")