from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_tokens=10)

"""
Use Case                                                         Recommended Temperature

Factual answers (math, code, facts)                              0.0 0.3

Balanced response (general QA, explanations)                     0.5-0.7

Creative writing, storytelling, jokes                            0.9-1.2

Maximum randomness (wild ideas, brainstorming)                   1.5+
"""

response = model.invoke("What is the capital of India?")    

print(response)

print(response.content)
