---
title: Multimodal Meme Self-Harm
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.23.3
python_version: "3.10"
app_file: app.py
pinned: false
---

# Deploy UI Gradio (ambil model dari HF Hub)

Folder ini untuk Hugging Face Spaces.

## Cara pakai

1. Pastikan model sudah di-upload dulu dari folder `deploy_hf_model`.
2. Set environment variable di Hugging Face Space:
   - `MODEL_REPO_ID = username/nama-repo-model`
3. Gunakan file berikut di Space:
   - `app.py`
   - `requirements.txt`

## Fitur UI

- Input gambar
- Input teks
- Output label prediksi
- Output confidence score per kelas

UI ini tidak menyimpan checkpoint lokal. Semua file model diunduh dari Hugging Face Model Hub.
