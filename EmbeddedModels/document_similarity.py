from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv(dotenv_path="../.env")

client = InferenceClient(
    token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

# Generate document embeddings
doc_embeddings = [
    client.feature_extraction(
        text=doc,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    for doc in documents
]

# Generate query embedding
query_embedding = client.feature_extraction(
    text=query,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Similarity calculation
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]

print("Query:", query)
print("Most Similar Document:", documents[index])
print("Similarity Score:", score)