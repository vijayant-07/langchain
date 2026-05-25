from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Always specify model explicitly
model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template='Write a summary for the following poem:\n{poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

# Load text file
loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))

# First document
print(docs[0].page_content)
print(docs[0].metadata)

# Create chain
chain = prompt | model | parser

# Invoke chain
result = chain.invoke({
    'poem': docs[0].page_content
})

print(result)