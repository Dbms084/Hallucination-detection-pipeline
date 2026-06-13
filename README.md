# Hallucination Detection & LLM Evaluation Platform

An AI evaluation platform that measures factual correctness of LLM responses using multiple evaluation strategies:

- Exact Match
- Keyword Match
- Semantic Similarity (Sentence Transformers)

Features:
- TruthfulQA Benchmark
- Gemini Integration
- Response Caching
- Evaluation Dashboard
- Threshold Analysis
- Failure Analysis
- Visual Analytics

Tech Stack:
- Python
- Streamlit
- Pandas
- Sentence Transformers
- Gemini API
- Matplotlib

Project Structure:
'''text
hallucination-detection-platform/

├── src/
│   ├── dataset/
│   ├── models/
│   ├── evaluators/
│   ├── scoring/
│   ├── storage/
│   ├── cache/
│   └── visualization/
│
├── results/
│   ├── csv/
│   └── visualizations/
│
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
