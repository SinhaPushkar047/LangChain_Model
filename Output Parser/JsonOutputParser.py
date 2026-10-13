from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8
)

parser=JsonOutputParser()

template=PromptTemplate(
    template="Give 5 important detail about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain=template | model | parser 
result= chain.invoke({'topic':'black hole'})

print(result)