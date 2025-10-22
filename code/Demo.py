import gradio as gr
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F

# Load models back into memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_paths = {
    "SmallCNN": "/content/saved_models/SmallCNN.pth",
    "ResNet50_Aug": "/content/saved_models/ResNet50_Aug.pth",
    "ResNet50_NoAug": "/content/saved_models/ResNet50_NoAug.pth",
    "ViT_Tiny": "/content/saved_models/ViT_Tiny.pth"
}

def create_model(name):
    if name=="cnn": return SmallCNN().to(DEVICE)
    elif name=="resnet":
        m=models.resnet50(pretrained=True)
        m.fc=nn.Linear(m.fc.in_features,2)
        return m.to(DEVICE)
    elif name=="vit":
        m=timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=2)
        return m.to(DEVICE)

def load_model(model_type, path):
    model = create_model(model_type)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

loaded_models = {
    "SmallCNN": load_model("cnn", model_paths["SmallCNN"]),
    "ResNet50_Aug": load_model("resnet", model_paths["ResNet50_Aug"]),
    "ResNet50_NoAug": load_model("resnet", model_paths["ResNet50_NoAug"]),
    "ViT_Tiny": load_model("vit", model_paths["ViT_Tiny"])
}

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

# Build Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Chest X-ray"),
        gr.Radio(list(loaded_models.keys()), label="Choose Model", value="ResNet50_Aug")
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

demo.launch(share=True)