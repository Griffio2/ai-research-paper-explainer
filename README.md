# AI Research Paper Explainer

Upload any research paper PDF and chat with it using RAG + Gemini.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## How it works
1. PDF gets chunked into 1000 character overlapping pieces
2. Each chunk is embedded into a vector using Gemini embeddings
3. Vectors are stored in a FAISS index
4. User question is embedded and matched against chunks
5. Top 4 chunks + question go to Gemini 2.5 Flash
6. Gemini answers based only on the retrieved context
```

Final project structure:
```
ai-research-explainer/
├── venv/
├── .env
├── app.py
├── utils.py
├── requirements.txt
└── README.md