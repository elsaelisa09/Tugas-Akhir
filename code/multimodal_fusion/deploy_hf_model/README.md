# Deploy Model Only (archA)

Folder ini khusus untuk upload model ke Hugging Face Model Hub.

## Isi folder

- `modeling_archA.py`: arsitektur model untuk inference
- `model_config.json`: konfigurasi model dan mapping label
- `inference.py`: loader + fungsi prediksi lokal
- `push_to_hub.py`: script upload ke Hugging Face

Checkpoint yang dipakai berasal dari folder output:

- `../output/bestmodel_mlA-class-imbalance-FL.pth`

Konfigurasi training model yang dibawa ke deploy:

- `imbalance_strategy`: `focal`
- `focal_gamma`: `2.0`

"Otak" model untuk inference adalah file checkpoint `.pth` di atas.

## Langkah deploy model

1. Install dependency:
   - `pip install -r requirements.txt`
2. Login HF:
   - `huggingface-cli login`
3. Upload model:
   - `python push_to_hub.py --repo_id username/nama-repo-model`

Opsional repo private:

- `python push_to_hub.py --repo_id username/nama-repo-model --private`

## Catatan

- Deploy ini baru model saja (tanpa UI), sesuai request.
- UI Gradio disiapkan terpisah di folder `deploy_hf_space`, dan akan mengambil file model dari Hub.
