# Ask Your Documents

```
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

copy .env.example .env         # then put your Groq key inside .env
uvicorn app:app --reload       # open http://127.0.0.1:8000
```

Upload PDF, TXT, MD or DOCX files in the page. They are indexed in the background, then you can ask questions.
Uploaded originals are deleted after indexing; only the searchable index is kept in `chroma_db/`.
API docs: http://127.0.0.1:8000/docs
