import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MIN_SCORE = 0.30          # drop chunks below this cosine similarity
CHUNK_SIZE, OVERLAP = 800, 150
MAX_FILES, MAX_MB = 10, 25
ALLOWED = {".pdf", ".txt", ".md", ".docx"}
