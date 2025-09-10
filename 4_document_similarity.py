from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is a great cricketer known for his batting skills",
    "Sachin Tendulkar is a legendary cricketer known as the God of Cricket",   
    "Lionel Messi is a world-class footballer known for his dribbling and scoring abilities",
    "Cristiano Ronaldo is a top footballer known for his goal-scoring prowess",
    "The Eiffel Tower is an iconic landmark located in Paris, France",
    "The Great Wall of China is a historic fortification stretching across northern China"
]

query = "Tell me about virat kohli"

doc_embeddings = embeddings.embed_documents(documents)

query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("Query: ", query)
print(documents[index])
print("Similarity Score: ", score)



