import os

import gradio as gr
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# Define SmallCNN architecture
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*56*56, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# Load models back into memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get the base directory (parent of 'code' folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")

model_paths = {
    "SmallCNN": os.path.join(MODELS_DIR, "SmallCNN.pth"),
    "ResNet50_Aug": os.path.join(MODELS_DIR, "ResNet50_Aug.pth"),
    "ResNet50_NoAug": os.path.join(MODELS_DIR, "ResNet50_NoAug.pth"),
    "ViT_Tiny": os.path.join(MODELS_DIR, "ViT_Tiny.pth")
}

def create_model(name):
    if name == "cnn":
        return SmallCNN().to(DEVICE)
    elif name == "resnet":
        m = models.resnet50(pretrained=True)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m.to(DEVICE)
    elif name == "vit":
        m = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2)
        return m.to(DEVICE)

def load_model(model_type, path):
    model = create_model(model_type)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

# Models will be loaded when script is run
loaded_models = {}

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction function for single model
def predict_single_model(model, model_name, image):
    """Run prediction for a single model"""
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    labels = ["Normal", "Pneumonia"]
    pred_idx = probs.argmax()
    return {
        "model": model_name,
        "predicted_class": labels[pred_idx],
        "normal_confidence": float(probs[0]) * 100,
        "pneumonia_confidence": float(probs[1]) * 100,
        "confidence": float(probs[pred_idx]) * 100
    }

# Prediction function for all models
def predict_all_models(image):
    """Run prediction on all loaded models and return comparison"""
    if not loaded_models:
        return "No models loaded!", None

    results = []
    for model_name, model in loaded_models.items():
        try:
            result = predict_single_model(model, model_name, image)
            results.append(result)
        except Exception as e:
            results.append({
                "model": model_name,
                "predicted_class": "Error",
                "normal_confidence": 0.0,
                "pneumonia_confidence": 0.0,
                "confidence": 0.0,
                "error": str(e)
            })

    # Create comparison DataFrame
    df = pd.DataFrame(results)
    df = df[["model", "predicted_class", "normal_confidence", "pneumonia_confidence", "confidence"]]
    df.columns = ["Model", "Prediction", "Normal %", "Pneumonia %", "Confidence %"]

    # Format percentages
    for col in ["Normal %", "Pneumonia %", "Confidence %"]:
        df[col] = df[col].apply(lambda x: f"{x:.2f}%")

    # Create HTML visualization
    html_output = create_comparison_html(results)

    return df, html_output

def create_comparison_html(results):
    """Create an HTML visualization of model comparisons"""
    html = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .results-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .results-title {
            text-align: center;
            color: #000000;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-subtitle {
            text-align: center;
            color: #212529;
            font-size: 16px;
            margin-bottom: 30px;
            font-weight: 500;
        }
        .models-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .model-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-top: 4px solid;
        }
        .model-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        }
        .model-card.vit { border-top-color: #6366f1; }
        .model-card.resnet { border-top-color: #10b981; }
        .model-card.cnn { border-top-color: #f59e0b; }
        .model-name {
            font-size: 20px;
            font-weight: 600;
            color: #000000 !important;
            margin: 0 0 16px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .prediction-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            margin: 12px 0;
            text-align: center;
            width: 100%;
        }
        .prediction-normal {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .prediction-pneumonia {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .confidence-bar-container {
            margin: 16px 0;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            font-weight: 500;
            color: #212529 !important;
            margin-bottom: 6px;
        }
        .confidence-label span {
            color: #212529 !important;
        }
        .confidence-bar-bg {
            background-color: #e9ecef;
            border-radius: 10px;
            height: 28px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 13px;
            font-weight: 600;
            transition: width 0.5s ease;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        .bar-normal {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .bar-pneumonia {
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        }
        .overall-confidence {
            margin-top: 20px;
            padding: 16px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            text-align: center;
        }
        .overall-confidence-label {
            font-size: 14px;
            color: #212529 !important;
            font-weight: 600;
        }
        .overall-confidence-value {
            font-size: 28px;
            font-weight: 700;
            color: #000000 !important;
            margin-top: 8px;
        }
        .agreement-banner {
            margin-top: 30px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .agreement-consensus {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            color: #155724;
        }
        .agreement-disagreement {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            color: #856404;
        }
    </style>
    <div class="results-container">
        <h2 class="results-title">📊 Model Comparison Results</h2>
        <p class="results-subtitle">Analysis from all trained models</p>

        <div class="models-grid">
    """

    # Model type mapping for styling
    model_type_map = {
        "ViT_Tiny": "vit",
        "ResNet50_Aug": "resnet",
        "ResNet50_NoAug": "resnet",
        "SmallCNN": "cnn"
    }

    for result in results:
        model_type = model_type_map.get(result['model'], '')
        pred_class = result['predicted_class']
        pred_class_lower = pred_class.lower()

        html += f"""
            <div class="model-card {model_type}">
                <h3 class="model-name" style="color: #000000 !important;">🤖 {result['model']}</h3>
                <div class="prediction-badge prediction-{pred_class_lower}">
                    {pred_class}
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span style="color: #212529 !important;">Normal</span>
                        <span style="color: #212529 !important;">{result['normal_confidence']:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill bar-normal" style="width: {result['normal_confidence']}%;">
                            {result['normal_confidence']:.1f}%
                        </div>
                    </div>
                </div>
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span style="color: #212529 !important;">Pneumonia</span>
                        <span style="color: #212529 !important;">{result['pneumonia_confidence']:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill bar-pneumonia" style="width: {result['pneumonia_confidence']}%;">
                            {result['pneumonia_confidence']:.1f}%
                        </div>
                    </div>
                </div>
                <div class="overall-confidence">
                    <div class="overall-confidence-label" style="color: #212529 !important;">Overall Confidence</div>
                    <div class="overall-confidence-value" style="color: #000000 !important;">{result['confidence']:.2f}%</div>
                </div>
            </div>
        """

    html += """
        </div>
    """

    # Add agreement section
    predictions = [r['predicted_class'] for r in results]
    unique_predictions = set(predictions)
    if len(unique_predictions) == 1:
        html += f"""
        <div class="agreement-banner agreement-consensus">
            ✅ <strong>Consensus Reached!</strong> All models agree: <strong>{predictions[0]}</strong>
        </div>
        """
    else:
        html += f"""
        <div class="agreement-banner agreement-disagreement">
            ⚠️ <strong>Models Disagree:</strong> {', '.join(unique_predictions)}
        </div>
        """

    html += """
    </div>
    """
    return html

def create_demo():
    """Create Gradio interface with loaded models"""
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px 20px;
        border-radius: 16px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 42px;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        margin: 15px 0 0 0;
        font-size: 18px;
        opacity: 0.95;
    }
    .upload-section {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        padding: 16px 32px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        margin-top: 20px !important;
    }
    .analyze-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5) !important;
    }
    .info-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 30px;
        border-radius: 16px;
        margin-top: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .info-section h3 {
        color: #000000;
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 15px;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .info-card.vit { border-left-color: #6366f1; }
    .info-card.resnet { border-left-color: #10b981; }
    .info-card.cnn { border-left-color: #f59e0b; }
    .info-card strong {
        color: #000000;
        font-size: 16px;
        display: block;
        margin-bottom: 8px;
    }
    .info-card p {
        color: #212529;
        margin: 0;
        font-size: 14px;
        line-height: 1.6;
    }
    """

    with gr.Blocks(title="Pneumonia Detection Demo", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML("""
            <div class="main-header">
                <h1>🫁 Pneumonia Detection from Chest X-rays</h1>
                <p>Upload a chest X-ray image to compare predictions from all trained models</p>
                <p style="font-size: 14px; opacity: 0.9;">All models will analyze the image simultaneously and display their results side-by-side</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML('<div class="upload-section">')
                    image_input = gr.Image(
                        type="pil",
                        label="📤 Upload Chest X-ray Image",
                        height=450,
                        show_label=True
                    )
                    gr.HTML('</div>')
                    predict_btn = gr.Button(
                        "🔍 Analyze with All Models",
                        variant="primary",
                        size="lg",
                        elem_classes=["analyze-btn"]
                    )

            with gr.Column(scale=2):
                html_output = gr.HTML(label="📊 Model Comparison Results")
                with gr.Accordion("📋 Detailed Results Table", open=False):
                    table_output = gr.Dataframe(
                        label="",
                        wrap=True,
                        headers=["Model", "Prediction", "Normal %", "Pneumonia %", "Confidence %"],
                        interactive=False
                    )

        gr.HTML("""
            <div class="info-section">
                <h3>📊 Model Information</h3>
                <div class="info-grid">
                    <div class="info-card vit">
                        <strong>🤖 ViT_Tiny</strong>
                        <p>Vision Transformer - Best overall performance with 83% accuracy and 0.96 AUC score</p>
                    </div>
                    <div class="info-card resnet">
                        <strong>🔬 ResNet50_Aug</strong>
                        <p>ResNet-50 with data augmentation - High AUC but shows class imbalance</p>
                    </div>
                    <div class="info-card resnet">
                        <strong>🔬 ResNet50_NoAug</strong>
                        <p>ResNet-50 without augmentation - High AUC but shows class imbalance</p>
                    </div>
                    <div class="info-card cnn">
                        <strong>🧠 SmallCNN</strong>
                        <p>Baseline CNN from scratch - Provides balanced predictions across classes</p>
                    </div>
                </div>
            </div>
        """)

        predict_btn.click(
            fn=predict_all_models,
            inputs=[image_input],
            outputs=[table_output, html_output]
        )

    return demo

if __name__ == "__main__":
    print("Loading models...")
    print(f"Models directory: {MODELS_DIR}")

    # Load all models
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"✓ Loading {name}...")
            try:
                if name == "SmallCNN":
                    loaded_models[name] = load_model("cnn", path)
                elif "ResNet50" in name:
                    loaded_models[name] = load_model("resnet", path)
                elif name == "ViT_Tiny":
                    loaded_models[name] = load_model("vit", path)
                print(f"  ✓ {name} loaded successfully")
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")
        else:
            print(f"✗ Missing {name} at {path}")

    if not loaded_models:
        print("\n❌ No models loaded! Please ensure model files exist in the Models/ directory.")
        exit(1)

    print(f"\n✓ Successfully loaded {len(loaded_models)} model(s)")
    print("\nStarting Gradio interface...")

    # Create and launch demo after models are loaded
    demo = create_demo()
    demo.launch(share=True)
