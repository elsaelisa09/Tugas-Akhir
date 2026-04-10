import importlib.util
import json
import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection


MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "elsaelisayohana09/multimodal_memeselfharm")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_module(module_file, module_name="modeling_archA_hub"):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HubArchAInference:
    def __init__(self, repo_id):
        self.repo_id = repo_id

        model_file = hf_hub_download(repo_id=repo_id, filename="modeling_archA.py", repo_type="model")
        config_file = hf_hub_download(repo_id=repo_id, filename="model_config.json", repo_type="model")

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        if str(self.config.get("checkpoint_format", "pth")).lower() != "pth":
            raise ValueError("Model Hub harus menggunakan checkpoint format .pth")

        if not str(self.config["checkpoint_filename"]).lower().endswith(".pth"):
            raise ValueError("checkpoint_filename harus file .pth")

        ckpt_file = hf_hub_download(
            repo_id=repo_id,
            filename=self.config["checkpoint_filename"],
            repo_type="model",
        )

        module = _load_module(model_file)
        CLIPElectraFusion = module.CLIPElectraFusion

        self.clip_processor = CLIPImageProcessor.from_pretrained(self.config["clip_model_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["electra_model_name"])

        clip_model = CLIPVisionModelWithProjection.from_pretrained(self.config["clip_model_name"])
        electra_model = AutoModel.from_pretrained(self.config["electra_model_name"])

        self.model = CLIPElectraFusion(
            clip_model=clip_model,
            electra_model=electra_model,
            fusion_img_dim=int(self.config["fusion_img_dim"]),
            fusion_text_dim=int(self.config["fusion_text_dim"]),
            num_classes=int(self.config["num_classes"]),
            freeze_encoders=False,
            fusion_method=self.config["fusion_method"],
        )

        ckpt = torch.load(ckpt_file, map_location=DEVICE)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(DEVICE)
        self.model.eval()

        self.max_len = int(self.config.get("max_len", 128))
        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}

    @torch.no_grad()
    def predict(self, image, text):
        if image is None:
            raise ValueError("Gambar wajib diisi")

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        tokenized = self.tokenizer(
            str(text or ""),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        pixel_values = clip_inputs["pixel_values"].to(DEVICE)
        input_ids = tokenized["input_ids"].to(DEVICE)
        attention_mask = tokenized["attention_mask"].to(DEVICE)

        logits, _, _ = self.model(pixel_values, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=-1)

        pred_idx = int(pred_idx.item())
        conf = float(conf.item())

        pred_label = self.id2label[pred_idx]
        scores = {self.id2label[i]: float(probs[i].item()) for i in range(len(probs))}

        text_out = (
            f"Prediksi: {pred_label}\n"
            f"Confidence: {conf:.4f}\n"
            f"NON-SELF-HARM: {scores.get('NON-SELF-HARM', 0.0):.4f}\n"
            f"SELF-HARM: {scores.get('SELF-HARM', 0.0):.4f}"
        )
        return text_out, scores


runner = None


def _get_runner():
    global runner
    if runner is None:
        runner = HubArchAInference(MODEL_REPO_ID)
    return runner


def infer(image, text):
    try:
        model_runner = _get_runner()
        text_out, _ = model_runner.predict(image, text)
        return text_out
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Multimodal Self-Harm Classifier") as demo:
    gr.Markdown("# Multimodal Self-Harm Classifier")
    gr.Markdown("Upload gambar + isi teks")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input Gambar")
        text_input = gr.Textbox(lines=4, label="Input Teks yang Terlihat di Gambar")

    submit_btn = gr.Button("Prediksi")

    result_text = gr.Textbox(label="Hasil Prediksi")
    submit_btn.click(
        fn=infer,
        inputs=[image_input, text_input],
        outputs=[result_text],
    )

    gr.Markdown("---\nTugas Akhir Elsa Elisa Yohana Sianturi, NIM 122140135")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
