from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Create documents
print("Creating documents...")

docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

print(f"Created {len(docs)} documents")

# Create embeddings
print("\nCreating OpenAI Embeddings...")
embeddings = OpenAIEmbeddings()

# Create vector store
print("Creating Chroma vector store...")
vector_store = Chroma(
    collection_name="sample",
    embedding_function=embeddings,
    persist_directory="my_chroma_db"
)

# Add documents with custom IDs
ids = ["virat", "rohit", "dhoni", "bumrah", "jadeja"]

print("Adding documents to Chroma...")
vector_store.add_documents(docs, ids=ids)
print("Documents added successfully!")

# View stored documents
print("\n" + "=" * 60)
print("ALL DOCUMENTS")
print("=" * 60)

all_docs = vector_store.get(include=["documents", "metadatas"])

print("IDs:")
print(all_docs["ids"])

print("\nMetadata:")
print(all_docs["metadatas"])

# Similarity Search
print("\n" + "=" * 60)
print("SIMILARITY SEARCH")
print("=" * 60)

results = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2
)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)

# Similarity Search with Score
print("\n" + "=" * 60)
print("SIMILARITY SEARCH WITH SCORES")
print("=" * 60)

results = vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2
)

for i, (doc, score) in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Score:", score)
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)

# Metadata Filtering
print("\n" + "=" * 60)
print("METADATA FILTERING")
print("=" * 60)

results = vector_store.similarity_search(
    query="cricket player",
    filter={"team": "Chennai Super Kings"},
    k=5
)

for i, doc in enumerate(results, start=1):
    print(f"\nCSK Player {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)

# Update document
print("\n" + "=" * 60)
print("UPDATING VIRAT DOCUMENT")
print("=" * 60)

updated_doc = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id="virat",
    document=updated_doc
)

print("Document updated successfully!")

# Verify update
updated_data = vector_store.get(
    ids=["virat"],
    include=["documents", "metadatas"]
)

print("\nUpdated Document:")
print(updated_data)

# Delete document
print("\n" + "=" * 60)
print("DELETING VIRAT DOCUMENT")
print("=" * 60)

vector_store.delete(ids=["virat"])

print("Document deleted successfully!")

# Verify deletion
final_data = vector_store.get(
    include=["documents", "metadatas"]
)

print("\nRemaining IDs:")
print(final_data["ids"])

print("\nTotal Documents Remaining:", len(final_data["ids"]))

print("\nProgram completed successfully!")



# from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from dotenv import load_dotenv

# load_dotenv()

# # Create LangChain documents for IPL players

# doc1 = Document(
#         page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
#         metadata={"team": "Royal Challengers Bangalore"}
#     )
# doc2 = Document(
#         page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
#         metadata={"team": "Mumbai Indians"}
#     )
# doc3 = Document(
#         page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
#         metadata={"team": "Chennai Super Kings"}
#     )
# doc4 = Document(
#         page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
#         metadata={"team": "Mumbai Indians"}
#     )
# doc5 = Document(
#         page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
#         metadata={"team": "Chennai Super Kings"}
#     )

# docs = [doc1, doc2, doc3, doc4, doc5]

# vector_store = Chroma(
#     embedding_function=OpenAIEmbeddings(),
#     persist_directory='my_chroma_db',
#     collection_name='sample'
# )

# vector_store.add_documents(docs)

# vector_store.get(include=['embeddings','documents', 'metadatas'])


# # search documents
# vector_store.similarity_search(
#     query='Who among these are a bowler?',
#     k=2
# )

# # search with similarity score
# vector_store.similarity_search_with_score(
#     query='Who among these are a bowler?',
#     k=2
# )

# # meta-data filtering
# vector_store.similarity_search_with_score(
#     query="",
#     filter={"team": "Chennai Super Kings"}
# )


# # update documents
# updated_doc1 = Document(
#     page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
#     metadata={"team": "Royal Challengers Bangalore"}
# )

# vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# # view documents
# vector_store.get(include=['embeddings','documents', 'metadatas'])

# # delete document
# vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# # view documents
# vector_store.get(include=['embeddings','documents', 'metadatas'])