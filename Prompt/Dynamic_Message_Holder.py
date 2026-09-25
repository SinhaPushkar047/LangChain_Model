from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    provider='groq',
    task='text-generation',
    max_new_tokens=1000
)

model=ChatHuggingFace(llm=llm)


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

prompt=ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])

while True:
    question=input("You: ")

    if question.lower()=="exit":
        break

    message=prompt.invoke({
        "chat_history":chat_history,
        "question": question
    })

    result=model.invoke(message)

    print("AI: ", result.content)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result.content))