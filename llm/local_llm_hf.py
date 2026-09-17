from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
import os

os.environ['HF_HOME']='D:/huggingface_cache'

llm=HuggingFacePipeline.from_model_id(
    model_id="openai/gpt-oss-120b",
    task='text-generation'
)

model=ChatHuggingFace(llm)

result=model.invoke("What is the capital of India")

print(result.content)