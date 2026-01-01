"""
Configuration settings for the interview agent
"""
from langchain_ollama import ChatOllama

# LLM Model Configuration
model = ChatOllama(model="llama3.2:1b")

# File paths - provide relative paths to your resume and job description files
# Files are located in the parent directory (one level up from modular_version/)
RESUME_PATH = "../sample_cv.pdf"
JD_PATH = "../sample_jd.txt"

# Interview Configuration
THREAD_ID = "1"
