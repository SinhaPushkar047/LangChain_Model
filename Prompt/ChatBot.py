from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)

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

    result=llm.invoke(message)

    print("AI: ", result.content)

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=result.content))