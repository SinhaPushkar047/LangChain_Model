from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=1000
)

model= ChatHuggingFace(llm=llm)

prompt=ChatPromptTemplate.from_messages([
    ("system","You are Travel Guide who make best trip planning list in point wise format"),
    ("human","Give the place to visit and special culture and food to eat in {location} ")
])

message=prompt.invoke({'location':'Rajasthan'})

result=model.invoke(message)

print(result.content)