from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "My Name is Pushkar Sinha",
    "I am Bokaro Steel City",
    "Currently studying in NIT Jamshedpur"
]

vector=embedding.embed_documents(documents)

print(str(vector))