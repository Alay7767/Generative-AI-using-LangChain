from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is Capital of India",
    "Tokyo is Capital of Japan",
    "Washington is Capital of USA",
    "Paris is Capital of France",
]

result = embedding.embed_documents(documents)

print(str(result))