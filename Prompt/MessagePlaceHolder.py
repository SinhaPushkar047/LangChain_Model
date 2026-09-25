from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider='groq',
    task='text-generation',
    max_new_tokens=1000
)

model=ChatHuggingFace(llm=llm)

prompt=ChatPromptTemplate.from_messages([
    ("system","You are a experienced ECE Proffesor with a high knowledge"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{question}")
])

chat_history = []

with open(r'Prompt\chat_history.txt') as f:
    for line in f:
        line = line.strip().rstrip(',')
        if line:
            chat_history.append(eval(line))

print(chat_history)

message = prompt.invoke({'chat_history':chat_history, 'question':'What is Pn junction diode'})

result= model.invoke(message) 

print(result.content)
