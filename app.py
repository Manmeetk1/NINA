import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

import numpy as np
import cv2
from PIL import Image

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="NINA - DR Detection",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------
# DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)

    model.load_state_dict(torch.load("nina_final_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------------------
# TRANSFORM
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ---------------------------
# GRAD-CAM
# ---------------------------
def generate_gradcam(model, image):

    gradients = []
    activations = []

    target_layer = model.features[-1]

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    input_tensor = transform(image).unsqueeze(0).to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1,2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    image_np = np.array(image.resize((224,224)))

    overlay = heatmap * 0.4 + image_np

    return overlay, pred_class.item()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🩺 NINA Dashboard")
st.sidebar.markdown("### Upload Retinal Image")

uploaded_file = st.sidebar.file_uploader("", type=["jpg","jpeg","png"])

st.sidebar.markdown("---")
st.sidebar.info("AI-based Diabetic Retinopathy Detection System")

# ---------------------------
# MAIN HEADER
# ---------------------------
st.title("👁️ NINA - DR Detection System")
st.markdown("AI-powered retinal analysis with explainable predictions")

# ---------------------------
# MAIN DISPLAY
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    overlay, pred = generate_gradcam(model, image)

    # Status indicator
    if pred == 0:
        status = "🟢 Normal"
    elif pred == 1 or pred == 2:
        status = "🟡 At Risk"
    else:
        status = "🔴 Critical"

    # Layout
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Original Retinal Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Grad-CAM Analysis")
        st.image(overlay.astype(np.uint8), width="stretch")

    # Prediction box
    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Diagnosis", classes[pred])

    with col4:
        st.metric("Condition", status)

else:
    st.info("👈 Please upload a retinal image from the sidebar to begin analysis.")