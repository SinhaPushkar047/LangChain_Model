from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=1000
)

model= ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template="Explain {topic} in detail in simple language",
    input_variables=['topic']
)

prompt=template.invoke({'topic':'CNN'})

result=model.invoke(prompt)

print(result.content)