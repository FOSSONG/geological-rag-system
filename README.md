Dual-mode RAG (HF Inference API) — Local run notes

1. Create venv and activate
   python -m venv venv
   .\venv\Scripts\activate

2. Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

3. Create .env in project root (use .env.example)
   HF_TOKEN=hf_your_token_here
   HF_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-7B

4. Run
   streamlit run app.py

First time: embeddings are computed via HF feature-extraction pipeline and a FAISS index is built and cached in faiss_index/. Subsequent runs reuse the local index (no re-download).
Generation always uses HF inference endpoint; no large files are downloaded locally.
