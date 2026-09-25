from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages([("human","Explain CNN in simple language"),
                          ("human","Difference between CNN and ANN")])

print(prompt)
