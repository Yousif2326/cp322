import gradio as gr
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os

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

# Prediction function
def predict(image, model_choice):
    model = loaded_models[model_choice]
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
    labels = ["Normal", "Pneumonia"]
    pred_idx = probs.argmax()
    return {
        "Predicted Class": labels[pred_idx],
        "Confidence": f"{probs[pred_idx]*100:.2f}%",
        "Model Used": model_choice
    }

def create_demo():
    """Create Gradio interface with loaded models"""
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Chest X-ray"),
            gr.Radio(list(loaded_models.keys()), label="Choose Model",
                    value=list(loaded_models.keys())[0] if loaded_models else "ViT_Tiny")
        ],
        outputs=[
            gr.JSON(label="Prediction Output")
        ],
        title="Pneumonia Detection Demo",
        description=(
            "Upload a chest X-ray image and select a trained model to predict whether "
            "the patient shows signs of pneumonia or is normal."
        ),
        examples=None
    )

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
