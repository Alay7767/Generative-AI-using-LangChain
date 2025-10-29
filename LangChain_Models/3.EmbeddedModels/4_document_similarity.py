from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is regarded as one of the best modern-day batsmen.",
    "Lionel Messi is an Argentine forward known for his dribbling skills.",
    "Michael Jordan is widely regarded as the greatest basketball player.",
    "Usain Bolt is the fastest man ever to run on a track.",
    "Stephen Curry revolutionized basketball with his 3-point shooting.",
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is : ", score)