import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from modeling_archA import CLIPElectraFusion


class ArchAInference:
    def __init__(self, base_dir=None, device=None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        config_path = self.base_dir / "model_config.json"
        with config_path.open("r", encoding="utf-8") as f:
            self.config = json.load(f)

        checkpoint_path = self.base_dir / self.config["checkpoint_filename"]
        if checkpoint_path.suffix.lower() != ".pth":
            raise ValueError(
                f"Checkpoint harus .pth, dapat: {checkpoint_path.name}"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {checkpoint_path}")

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

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise ValueError(
                "Checkpoint .pth harus berisi key 'model_state_dict'."
            )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.max_len = int(self.config.get("max_len", 128))
        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}

    @torch.no_grad()
    def predict(self, image, text):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        tokenized = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        pixel_values = clip_inputs["pixel_values"].to(self.device)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        logits, _, _ = self.model(pixel_values, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        conf, pred_idx = torch.max(probs, dim=-1)

        pred_idx = int(pred_idx.item())
        conf = float(conf.item())

        return {
            "label": self.id2label[pred_idx],
            "confidence": conf,
            "probabilities": {
                self.id2label[i]: float(probs[i].item()) for i in range(len(probs))
            },
        }


if __name__ == "__main__":
    runner = ArchAInference()
    sample_text = "i feel tired and hopeless"
    sample_image_path = runner.base_dir / "sample.jpg"

    if not sample_image_path.exists():
        raise FileNotFoundError("Tambahkan sample.jpg di folder deploy_hf_model untuk test cepat.")

    out = runner.predict(sample_image_path, sample_text)
    print(out)
