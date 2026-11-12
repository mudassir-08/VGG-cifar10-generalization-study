
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr
import time

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

CLASS_EMOJI = {
    "Airplane": "✈️", "Automobile": "🚗", "Bird": "🐦",
    "Cat": "🐱",  "Deer": "🦌",   "Dog": "🐶",
    "Frog": "🐸", "Horse": "🐴",  "Ship": "🚢", "Truck": "🚛",
}

# ── Model ─────────────────────────────────────────────────────────────────────
class VGGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(cb(3,64),   cb(64,64),   nn.MaxPool2d(2))
        self.block2 = nn.Sequential(cb(64,128),  cb(128,128), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(cb(128,256), cb(256,256), cb(256,256))
        self.block4 = nn.Sequential(cb(256,512), cb(512,512), cb(512,512), nn.MaxPool2d(2))
        self.block5 = nn.Sequential(cb(512,512), cb(512,512), cb(512,512), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256,10)
        )
    def forward(self, x):
        for b in [self.block1,self.block2,self.block3,self.block4,self.block5]: x = b(x)
        return self.classifier(x)

# ── Load checkpoint ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGClassifier().to(DEVICE)
ckpt  = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

BEST_EPOCH = ckpt.get("epoch", "—")
BEST_ACC   = ckpt.get("best_acc", None)
ACC_STR    = f"{BEST_ACC*100:.2f}%" if BEST_ACC else "—"
DEVICE_STR = "GPU · CUDA" if DEVICE.type == "cuda" else "CPU"

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616])
])

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return "—", "—", "—", "—", "—", "—"

    t0     = time.perf_counter()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    latency = (time.perf_counter() - t0) * 1000

    top5   = probs.topk(5)
    p_idx  = top5.indices[0].item()
    p_name = CLASSES[p_idx]
    p_conf = probs[p_idx].item()

    label    = f"{CLASS_EMOJI[p_name]}  {p_name}"
    conf_str = f"{p_conf*100:.2f}%"
    lat_str  = f"{latency:.1f} ms"

    bar_rows = []
    for i in range(5):
        idx  = top5.indices[i].item()
        prob = top5.values[i].item()
        filled = round(prob * 28)
        bar = "█" * filled + "░" * (28 - filled)
        bar_rows.append(f"{CLASSES[idx]:<12}  {bar}  {prob*100:.2f}%")
    top5_str = "\n".join(bar_rows)

    entropy   = -(probs * (probs + 1e-9).log()).sum().item()
    certainty = "High" if p_conf > 0.75 else "Medium" if p_conf > 0.45 else "Low"

    meta = (
        f"Device       :  {DEVICE_STR}\n"
        f"Latency      :  {lat_str}\n"
        f"Precision    :  float32 inference  (float16 AMP during training)\n"
        f"Certainty    :  {certainty}  (entropy = {entropy:.3f} nats)\n"
        f"Checkpoint   :  epoch {BEST_EPOCH}  ·  best val acc {ACC_STR}"
    )

    return label, conf_str, lat_str, certainty, top5_str, meta


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* Container */
.gradio-container {
    max-width: 1060px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* Header strip */
.header-strip {
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 20px;
    margin-bottom: 28px;
}

/* Metric pill badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* Primary prediction output — large, bold */
#pred-box textarea {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #111827 !important;
    background: #f9fafb !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 10px !important;
    text-align: center !important;
    letter-spacing: 0.5px !important;
}

/* Confidence output */
#conf-box textarea {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1d4ed8 !important;
    background: #eff6ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 10px !important;
    text-align: center !important;
}

/* Latency output */
#lat-box textarea {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #065f46 !important;
    background: #ecfdf5 !important;
    border: 1px solid #a7f3d0 !important;
    border-radius: 10px !important;
    text-align: center !important;
}

/* Certainty output */
#cert-box textarea {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #92400e !important;
    background: #fffbeb !important;
    border: 1px solid #fde68a !important;
    border-radius: 10px !important;
    text-align: center !important;
}

/* Top-5 bar chart — monospace */
#top5-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 13px !important;
    line-height: 1.85 !important;
    color: #1f2937 !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Meta / diagnostics */
#meta-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
    font-size: 12.5px !important;
    line-height: 1.9 !important;
    color: #374151 !important;
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}

/* Analyze button */
#analyze-btn button {
    background: #1d4ed8 !important;
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    border-radius: 10px !important;
    border: none !important;
    height: 48px !important;
    transition: background 0.15s ease !important;
}
#analyze-btn button:hover {
    background: #1e40af !important;
}

/* Upload box */
#img-upload .upload-container {
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    background: #f9fafb !important;
}

/* Section dividers */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 10px;
    margin-top: 24px;
}

/* Footer */
.footer-strip {
    border-top: 1px solid #e5e7eb;
    padding-top: 18px;
    margin-top: 32px;
    font-size: 13px;
    color: #6b7280;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="ModernVGG-BN · CIFAR-10",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace"],
    ),
    css=CSS
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("""
<div class="header-strip">

# ModernVGG-BN &nbsp;·&nbsp; CIFAR-10 Image Classifier

Modernized VGG-16 backbone with BatchNorm, trained from scratch using AMP, AdamW,
CosineAnnealingLR, gradient clipping, and early stopping.
Upload any image to run a live forward pass through the trained model.

**Author:** Malik Muhammad Mudassir Iqbal &nbsp;·&nbsp;
[HuggingFace ↗](https://huggingface.co/Mudassir-08) &nbsp;·&nbsp;
[LinkedIn ↗](https://www.linkedin.com/in/malik-muhammad-mudassir-iqbal-21906336a)

</div>
""")

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left col — input
        with gr.Column(scale=5, min_width=280):
            gr.Markdown('<div class="section-label">Input</div>')
            img_input = gr.Image(
                type="pil",
                label="Upload image",
                height=260,
                elem_id="img-upload"
            )
            analyze_btn = gr.Button(
                "🔬  Run Inference",
                variant="primary",
                size="lg",
                elem_id="analyze-btn"
            )
            gr.Markdown(
                "<p style='font-size:12.5px; color:#6b7280; margin-top:10px; line-height:1.6'>"
                "Accepts any image format. Model was trained on CIFAR-10 at 224×224.<br>"
                "Best results: objects, animals, vehicles on clean backgrounds."
                "</p>"
            )

        # Right col — outputs
        with gr.Column(scale=7, min_width=360):

            gr.Markdown('<div class="section-label">Prediction</div>')
            pred_out = gr.Textbox(
                label="Predicted class",
                interactive=False,
                elem_id="pred-box"
            )

            with gr.Row():
                conf_out = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                    elem_id="conf-box",
                    scale=1
                )
                lat_out = gr.Textbox(
                    label="Latency",
                    interactive=False,
                    elem_id="lat-box",
                    scale=1
                )
                cert_out = gr.Textbox(
                    label="Certainty",
                    interactive=False,
                    elem_id="cert-box",
                    scale=1
                )

            gr.Markdown('<div class="section-label">Top-5 class probabilities</div>')
            top5_out = gr.Textbox(
                label="",
                lines=6,
                interactive=False,
                elem_id="top5-box"
            )

            gr.Markdown('<div class="section-label">Inference diagnostics</div>')
            meta_out = gr.Textbox(
                label="",
                lines=5,
                interactive=False,
                elem_id="meta-box"
            )

    # ── Wire up ───────────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=predict,
        inputs=img_input,
        outputs=[pred_out, conf_out, lat_out, cert_out, top5_out, meta_out]
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(f"""
<div class="footer-strip">

**Architecture** &nbsp; VGG-16 backbone · BatchNorm2d after every Conv · AdaptiveAvgPool head · ~14.7M params
&nbsp;&nbsp;|&nbsp;&nbsp;
**Training** &nbsp; AdamW · CosineAnnealingLR · AMP float16 · GradScaler · gradient clipping · early stopping
&nbsp;&nbsp;|&nbsp;&nbsp;
**Best val acc** &nbsp; {ACC_STR} at epoch {BEST_EPOCH}
&nbsp;&nbsp;|&nbsp;&nbsp;
**Device** &nbsp; {DEVICE_STR}
&nbsp;&nbsp;|&nbsp;&nbsp;
**License** &nbsp; MIT

</div>
""")

if __name__ == "__main__":
    demo.launch()