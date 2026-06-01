from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# Step 1: Fetch Transcript
# ==========================================

video_id = "Gfr50f6ZBvo"

print("Fetching transcript...")

api = YouTubeTranscriptApi()

transcript = api.fetch(video_id)

text = " ".join(snippet.text for snippet in transcript)

print(f"Transcript Length: {len(text)} characters")

# ==========================================
# Step 2: Split Transcript
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([text])

print(f"Total Chunks Created: {len(chunks)}")

# ==========================================
# Step 3: Create Vector Store
# ==========================================

print("Creating embeddings and vector store...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector store created successfully!")

# ==========================================
# Step 4: Create Retriever
# ==========================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ==========================================
# Step 5: Ask Question
# ==========================================

question = (
    "Is the topic of nuclear fusion discussed in this video? "
    "If yes then what was discussed?"
)

print("\nQuestion:")
print(question)

retrieved_docs = retriever.invoke(question)

print("\nRetrieved Chunks:")
print("=" * 60)

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\nChunk {i}")
    print("-" * 30)
    print(doc.page_content[:500])

# ==========================================
# Step 6: Build Context
# ==========================================

context_text = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)

# ==========================================
# Step 7: Prompt
# ==========================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.

If the answer cannot be found in the context,
say "I don't know based on the transcript."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)

final_prompt = prompt.invoke(
    {
        "context": context_text,
        "question": question
    }
)

# ==========================================
# Step 8: LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("\nGenerating Answer...")

response = llm.invoke(final_prompt)

print("\nAnswer:")
print("=" * 60)
print(response.content)