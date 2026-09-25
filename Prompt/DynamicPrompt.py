from langchain_core.prompts import ChatPromptTemplate

prompt= ChatPromptTemplate([
    ("human","Explain {topic} in simple language and in detail")
])

result=prompt.format(topic="CNN")

print(result)