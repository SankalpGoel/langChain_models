from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

embeddings=OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.",
    "It aims to make it easy to build language model applications by chaining together different components."
]
result=embeddings.embed_documents(documents)

print(str(result))