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
    html = "<div style='font-family: Arial, sans-serif; padding: 20px;'>"
    html += "<h2 style='text-align: center; color: #2c3e50;'>Model Comparison Results</h2>"

    html += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;'>"

    for result in results:
        html += f"""
        <div style='border: 3px solid #3498db; border-radius: 10px; padding: 15px; background-color: #ebf5fb;'>
            <h3 style='margin-top: 0; color: #2c3e50;'>{result['model']}</h3>
            <p style='font-size: 18px; font-weight: bold; color: #e74c3c; margin: 10px 0;'>
                {result['predicted_class']}
            </p>
            <div style='margin: 10px 0;'>
                <div style='margin: 5px 0;'>
                    <span style='color: #3498db;'>Normal:</span>
                    <div style='background-color: #ecf0f1; border-radius: 5px; height: 20px; margin-top: 5px;'>
                        <div style='background-color: #3498db; height: 100%; width: {result['normal_confidence']}%; border-radius: 5px; text-align: center; color: white; font-size: 12px; line-height: 20px;'>
                            {result['normal_confidence']:.1f}%
                        </div>
                    </div>
                </div>
                <div style='margin: 5px 0;'>
                    <span style='color: #e74c3c;'>Pneumonia:</span>
                    <div style='background-color: #ecf0f1; border-radius: 5px; height: 20px; margin-top: 5px;'>
                        <div style='background-color: #e74c3c; height: 100%; width: {result['pneumonia_confidence']}%; border-radius: 5px; text-align: center; color: white; font-size: 12px; line-height: 20px;'>
                            {result['pneumonia_confidence']:.1f}%
                        </div>
                    </div>
                </div>
            </div>
            <p style='margin-top: 10px; font-size: 14px; color: #7f8c8d;'>
                Confidence: <strong>{result['confidence']:.2f}%</strong>
            </p>
        </div>
        """

    html += "</div>"

    # Add agreement section
    predictions = [r['predicted_class'] for r in results]
    unique_predictions = set(predictions)
    if len(unique_predictions) == 1:
        html += "<div style='margin-top: 20px; padding: 15px; background-color: #d5f4e6; border-radius: 10px; text-align: center;'>"
        html += f"<h3 style='color: #27ae60;'>✓ All models agree: <strong>{predictions[0]}</strong></h3>"
        html += "</div>"
    else:
        html += "<div style='margin-top: 20px; padding: 15px; background-color: #fff3cd; border-radius: 10px; text-align: center;'>"
        html += f"<h3 style='color: #856404;'>⚠ Models disagree: {', '.join(unique_predictions)}</h3>"
        html += "</div>"

    html += "</div>"
    return html

def create_demo():
    """Create Gradio interface with loaded models"""
    with gr.Blocks(title="Pneumonia Detection Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🫁 Pneumonia Detection from Chest X-rays

            Upload a chest X-ray image to compare predictions from all trained models.
            All models will analyze the image simultaneously and display their results side-by-side.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-ray Image",
                    height=400
                )
                predict_btn = gr.Button("🔍 Analyze with All Models", variant="primary", size="lg")

            with gr.Column(scale=2):
                html_output = gr.HTML(label="Model Comparison")
                table_output = gr.Dataframe(
                    label="Detailed Results Table",
                    wrap=True,
                    headers=["Model", "Prediction", "Normal %", "Pneumonia %", "Confidence %"]
                )

        gr.Markdown(
            """
            ### 📊 Model Information
            - **ViT_Tiny**: Vision Transformer (Best overall performance - 83% accuracy, 0.96 AUC)
            - **ResNet50_Aug**: ResNet-50 with data augmentation (High AUC but class imbalance)
            - **ResNet50_NoAug**: ResNet-50 without augmentation (High AUC but class imbalance)
            - **SmallCNN**: Baseline CNN from scratch (Balanced predictions)
            """
        )

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
