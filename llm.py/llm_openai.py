from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model=OpenAI(model="gpt-5.6-luna")

result=model.invoke("Write a 5 line on AI")
print(result)