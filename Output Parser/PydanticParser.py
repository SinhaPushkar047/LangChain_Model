from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=1.4
)

class Person(BaseModel):
    name: str=Field(description='Name of the Person')
    age: int=Field(description='Age of the Person')
    city: str=Field(description='City where the person live')


parser=PydanticOutputParser(pydantic_object=Person)


template=PromptTemplate(
    template='Predict the name, age, city of a indian person {format_instruction}',
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain= template | model | parser

result=chain.invoke({})

print(result)