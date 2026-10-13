from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)

template1=PromptTemplate(
    template="Write a detail report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Give me a 5 line summary. \n{text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'Ai rising'})

print(result)