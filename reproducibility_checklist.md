Reproducibility Checklist (filled)

Code and build:
- Repository contains app.py, utils.py, requirements.txt, README.md, .env.example
- Exact commit: (add commit hash here before submission)

Data and preprocessing:
- Documents: use public documents or include instructions to download them.
- Preprocessing: utils.clean_and_normalize_text and utils.semantic_chunk are included and deterministic.

Models and hyperparameters:
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- HF generation model: HF_MODEL_ID set in .env (e.g., google/flan-t5-small)
- Chunk size: sentences_per_chunk (UI parameter); default 18
- Overlap: overlap (UI parameter); default 3

Random seeds and determinism:
- Index and sentence-transformers are deterministic for the same environment and model versions.
- No stochastic sampling is used for generation in code (do_sample=False).

Compute and hardware:
- Embedding computation: CPU or GPU; times will vary. Report GPU type if used.
- Generation: via Hugging Face Inference API (cloud); no local GPU required.

Evaluation scripts:
- Provide evaluation harness (not included here) to reproduce paper metrics: retrieval precision@k, entity F1, hallucination rate.

Licenses:
- Code: recommend MIT
- Models: check HF model license and include in paper.

Notes:
- Set MODEL_KNOWLEDGE_CUTOFF in .env to the model cutoff date you will report in the paper.
