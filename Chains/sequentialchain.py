from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    temperature=0.6
)

parser1=StrOutputParser()


class Response(BaseModel):
    point: list[str]=Field(description='All information should be in point wise')

parser2=PydanticOutputParser(pydantic_object=Response)


template1= PromptTemplate(
    template='Give detail information about {name}',
    input_variables=['name']
)

template2= PromptTemplate(
    template='Give the highlight of the {text} in bullet points \n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

chain= template1 | model | parser1 | template2 | model | parser2

result= chain.invoke({'name':'NimsDai'})

print(result)