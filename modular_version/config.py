"""
Configuration settings for the interview agent
"""
from langchain_ollama import ChatOllama

# LLM Model Configuration
model = ChatOllama(model="llama3.2:1b")

# File paths (update these with your actual paths)
RESUME_PATH = "C:/Users/Akash/Desktop/Langgraph/Pre_Int_Agent/sample_cv.pdf"
JD_PATH = "C:/Users/Akash/Desktop/Langgraph/Pre_Int_Agent/sample_jd.txt"

# Interview Configuration
THREAD_ID = "1"