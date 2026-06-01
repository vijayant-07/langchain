from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# =====================================
# Step 1: Get Transcript
# =====================================

video_id = "Gfr50f6ZBvo"

api = YouTubeTranscriptApi()

transcript = api.fetch(video_id)

text = " ".join(snippet.text for snippet in transcript)

print("Transcript fetched successfully")

# =====================================
# Step 2: Split Transcript
# =====================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.create_documents([text])

print(f"Created {len(documents)} chunks")

# =====================================
# Step 3: Create Vector Store
# =====================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = FAISS.from_documents(
    documents,
    embeddings
)

print("Vector Store Created")

# =====================================
# Step 4: Create Retriever
# =====================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# =====================================
# Step 5: Prompt
# =====================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY from the provided transcript.

If the answer is not present in the transcript,
say "I don't know based on the transcript."

Transcript Context:
{context}

Question:
{question}
""")

# =====================================
# Step 6: LLM
# =====================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =====================================
# Step 7: Helper Function
# =====================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =====================================
# Step 8: RAG Chain
# =====================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================
# Step 9: Ask Question
# =====================================

question = "Is nuclear fusion discussed in this video?"

response = rag_chain.invoke(question)

print("\nAnswer:")
print(response)