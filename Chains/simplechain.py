from langchain_nvidia import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatNVIDIA(
  model="google/diffusiongemma-26b-a4b-it",
  temperature=1,
)

template=PromptTemplate(
    template='Who is {name} and what was his achivement',
    input_variables=['name']
)

parser=StrOutputParser()

chain=template | model | parser

result= chain.invoke({'name':'Nirmal Purja'})

print(result)
