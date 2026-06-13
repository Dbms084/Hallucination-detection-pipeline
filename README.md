# Hallucination Detection & LLM Evaluation Platform

An AI evaluation platform that measures factual correctness of LLM responses using multiple evaluation strategies:

- Exact Match
- Keyword Match
- Semantic Similarity (Sentence Transformers)

## Features:
- TruthfulQA Benchmark
- Gemini Integration
- Response Caching
- Evaluation Dashboard
- Threshold Analysis
- Failure Analysis
- Visual Analytics

## Tech Stack:
- Python
- Streamlit
- Pandas
- Sentence Transformers
- Gemini API
- Matplotlib

## 📂 Project Structure

```text
hallucination-detection-platform/
├── src/
│   ├── dataset/         # Data loading and preprocessing pipelines
│   ├── models/          # LLM integrations and wrapper interfaces
│   ├── evaluators/      # Core hallucination detection logic
│   ├── scoring/         # Metric calculation and scoring algorithms
│   ├── storage/         # Database and persistent storage layers
│   ├── cache/           # Response caching mechanisms
│   └── visualization/   # Dashboard charts and plotting utilities
├── results/
│   ├── csv/             # Exported evaluation metrics and raw logs
│   └── visualizations/  # Saved static plots and performance charts
├── dashboard.py         # Streamlit/Gradio frontend application
├── main.py              # CLI entry point for running evaluations
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
