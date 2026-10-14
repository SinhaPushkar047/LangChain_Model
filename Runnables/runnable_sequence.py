from langchain_nvidia import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatNVIDIA(
    model='google/diffusiongemma-26b-a4b-it',
    temperature=0.4
)

template=PromptTemplate(
    template='Tell me how to make {topic}',
    input_variables=['topic']
)

chain= template | model

result= chain.invoke({'topic':'bomb'})

print(result)