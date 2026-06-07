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
# DR STAGE INFO (Description + Treatment)
# ---------------------------
dr_info = {
    0: {
        "description": (
            "No signs of diabetic retinopathy are detected. The retina appears healthy with no "
            "visible microaneurysms, hemorrhages, or abnormal blood vessel changes. However, "
            "people with diabetes remain at ongoing risk and should maintain regular eye screenings."
        ),
        "symptoms": [
            "No visible retinal damage",
            "Normal blood vessel appearance",
            "No fluid leakage detected"
        ],
        "treatment": [
            "🔍 **Routine annual dilated eye exam** by an ophthalmologist",
            "🩸 **Strict blood sugar (HbA1c) control** — target below 7%",
            "💊 **Manage blood pressure** (target < 130/80 mmHg) and cholesterol",
            "🥗 **Healthy lifestyle**: balanced diet, regular exercise, no smoking",
            "📋 **Regular follow-up** with your diabetologist or endocrinologist"
        ],
        "follow_up": "Every 12 months",
        "urgency": "Routine"
    },
    1: {
        "description": (
            "Mild Non-Proliferative Diabetic Retinopathy (NPDR). Small microaneurysms — tiny "
            "balloon-like swellings in the retina's blood vessels — are present. These are the "
            "earliest signs of retinal damage caused by diabetes. Vision is usually not affected "
            "at this stage, but close monitoring is essential."
        ),
        "symptoms": [
            "Microaneurysms (tiny dot-like red spots)",
            "Possible minor retinal hemorrhages",
            "Mild vascular changes in the retina"
        ],
        "treatment": [
            "👁️ **Dilated eye exam every 9–12 months** to monitor progression",
            "🩸 **Tight glycemic control** — HbA1c < 7% is critical to slow progression",
            "💊 **Blood pressure & lipid management** with medications if needed",
            "🚭 **Quit smoking** — nicotine accelerates retinal vessel damage",
            "🥗 **Dietary modifications**: low glycemic index foods, reduce saturated fats",
            "⚕️ **No laser or surgical treatment** required at this stage"
        ],
        "follow_up": "Every 9–12 months",
        "urgency": "Monitor Closely"
    },
    2: {
        "description": (
            "Moderate Non-Proliferative Diabetic Retinopathy (NPDR). The retinal blood vessels "
            "are more noticeably damaged. Blockages begin to develop, preventing parts of the retina "
            "from receiving adequate blood supply. Dot and blot hemorrhages, hard exudates (fat/protein "
            "deposits), and cotton-wool spots may be visible. Risk of vision loss increases "
            "significantly at this stage."
        ),
        "symptoms": [
            "Multiple microaneurysms and dot-blot hemorrhages",
            "Hard exudates (yellowish deposits of lipids)",
            "Cotton-wool spots (white fluffy patches indicating nerve damage)",
            "Possible mild macular edema causing slight vision blur"
        ],
        "treatment": [
            "👁️ **Dilated eye exam every 6 months** — close ophthalmologist supervision",
            "💉 **Anti-VEGF injections** (e.g., Ranibizumab, Bevacizumab) if macular edema is present",
            "🔴 **Focal laser photocoagulation** may be recommended to seal leaking blood vessels",
            "🩸 **Aggressive blood sugar management** — consult your endocrinologist for insulin/medication adjustment",
            "💊 **Fenofibrate** (lipid-lowering drug) has shown benefit in slowing moderate NPDR",
            "🧪 **Fluorescein Angiography (FFA)** or **OCT scan** to assess macular involvement",
            "🏥 **Referral to a retinal specialist** is strongly advised"
        ],
        "follow_up": "Every 6 months",
        "urgency": "Prompt Attention Required"
    },
    3: {
        "description": (
            "Severe Non-Proliferative Diabetic Retinopathy (NPDR). Extensive retinal damage is "
            "occurring. Large areas of the retina are losing blood supply (ischemia), which triggers "
            "the eye to send distress signals for new blood vessel growth. Characterized by the "
            "'4-2-1 rule': hemorrhages in all 4 quadrants, venous beading in 2+ quadrants, or "
            "intraretinal microvascular abnormalities (IRMA) in 1+ quadrant. High risk of "
            "converting to Proliferative DR within 1 year."
        ),
        "symptoms": [
            "Extensive hemorrhages in all four retinal quadrants",
            "Venous beading (irregular, sausage-link appearance of veins)",
            "Intraretinal microvascular abnormalities (IRMA)",
            "Severe reduction in retinal blood flow",
            "Possible significant visual disturbance or dark spots"
        ],
        "treatment": [
            "🚨 **Urgent referral to a retinal specialist** — do not delay",
            "🔴 **Pan-Retinal Photocoagulation (PRP) laser therapy** — widely applied to prevent new vessel growth",
            "💉 **Intravitreal Anti-VEGF injections** (Aflibercept, Ranibizumab, or Bevacizumab) every 4–6 weeks",
            "🧪 **OCT-Angiography and Fluorescein Angiography** for detailed retinal mapping",
            "🏥 **Possible hospitalization** if vitreous hemorrhage or retinal detachment risk is identified",
            "💊 **Systemic control**: strict BP < 130/80, HbA1c < 7%, statin therapy",
            "📅 **Follow-up every 2–3 months** with retinal imaging at each visit"
        ],
        "follow_up": "Every 2–3 months",
        "urgency": "Urgent"
    },
    4: {
        "description": (
            "Proliferative Diabetic Retinopathy (PDR) — the most advanced and vision-threatening stage. "
            "New, fragile blood vessels (neovascularization) grow abnormally on the retinal surface and "
            "into the vitreous gel. These vessels bleed easily, causing vitreous hemorrhage, and their "
            "fibrous tissue can pull on the retina leading to tractional retinal detachment. Without "
            "immediate treatment, this stage can result in severe vision loss or complete blindness."
        ),
        "symptoms": [
            "Neovascularization (abnormal new blood vessel growth on retina/disc)",
            "Vitreous hemorrhage — sudden vision loss, floaters, or dark streaks",
            "Tractional retinal detachment — flashes of light, shadow over vision",
            "Neovascular glaucoma (increased eye pressure due to new vessels)",
            "Possible sudden, complete loss of vision"
        ],
        "treatment": [
            "🚨 **EMERGENCY — Immediate retinal specialist consultation required**",
            "🔴 **Extensive Pan-Retinal Photocoagulation (PRP)** — 1,200–1,600 laser burns to peripheral retina",
            "💉 **Intravitreal Anti-VEGF therapy** (Aflibercept/Bevacizumab) — monthly injections to shrink new vessels",
            "🏥 **Vitrectomy surgery** — removal of vitreous gel if hemorrhage or retinal detachment is present",
            "⚕️ **Intravitreal steroids** (Triamcinolone/Dexamethasone implant) for persistent macular edema",
            "👁️ **Scleral buckle or silicone oil tamponade** in cases of retinal detachment",
            "💊 **Maximal systemic control**: HbA1c, blood pressure, renal function monitoring",
            "📅 **Monthly follow-up** with retinal imaging; surgery follow-up as directed by surgeon"
        ],
        "follow_up": "Monthly or as directed by surgeon",
        "urgency": "Emergency"
    }
}

# Urgency color map
urgency_colors = {
    "Routine": "🟢",
    "Monitor Closely": "🟡",
    "Prompt Attention Required": "🟠",
    "Urgent": "🔴",
    "Emergency": "🆘"
}

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

# DR Stage Reference in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 DR Stage Reference")
stage_colors = ["🟢 No DR", "🟡 Mild", "🟠 Moderate", "🔴 Severe", "🆘 Proliferative"]
for s in stage_colors:
    st.sidebar.markdown(f"- {s}")

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

    # Layout — Images
    col1, col2 = st.columns([1,1])

    with col1:
        st.subheader("Original Retinal Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Grad-CAM Analysis")
        st.image(overlay.astype(np.uint8), width="stretch")

    # Prediction metrics
    st.markdown("---")
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Diagnosis", classes[pred])

    with col4:
        st.metric("Condition", status)

    with col5:
        info = dr_info[pred]
        urgency_icon = urgency_colors.get(info["urgency"], "⚪")
        st.metric("Follow-up Required", f"{urgency_icon} {info['follow_up']}")

    # ---------------------------
    # DR STAGE DETAILS SECTION
    # ---------------------------
    st.markdown("---")
    st.subheader(f"📋 Clinical Details: {classes[pred]}")

    info = dr_info[pred]

    # Description
    st.markdown("#### 🔬 About This Stage")
    st.info(info["description"])

    # Symptoms + Treatment side by side
    sym_col, treat_col = st.columns([1, 1.2])

    with sym_col:
        st.markdown("#### 🩺 Common Signs & Symptoms")
        for symptom in info["symptoms"]:
            st.markdown(f"- {symptom}")

    with treat_col:
        st.markdown("#### 💊 Recommended Treatment & Management")
        for step in info["treatment"]:
            st.markdown(f"- {step}")

    # Urgency banner
    st.markdown("---")
    urgency = info["urgency"]
    urgency_icon = urgency_colors.get(urgency, "⚪")

    if urgency == "Emergency":
        st.error(f"{urgency_icon} **{urgency}** — Seek immediate medical attention from a retinal specialist. Vision loss may be irreversible if treatment is delayed.")
    elif urgency == "Urgent":
        st.error(f"{urgency_icon} **{urgency}** — Please schedule an appointment with a retinal specialist as soon as possible.")
    elif urgency == "Prompt Attention Required":
        st.warning(f"{urgency_icon} **{urgency}** — Consult an ophthalmologist promptly. Early intervention can prevent progression.")
    elif urgency == "Monitor Closely":
        st.warning(f"{urgency_icon} **{urgency}** — Continue regular monitoring and maintain good diabetic control.")
    else:
        st.success(f"{urgency_icon} **{urgency}** — No active treatment needed. Maintain healthy lifestyle and annual screening.")

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer**: This AI-generated analysis is intended for screening purposes only and does not replace "
        "a professional medical diagnosis. Always consult a qualified ophthalmologist or retinal specialist for "
        "clinical decisions and treatment planning."
    )

else:
    st.info("👈 Please upload a retinal image from the sidebar to begin analysis.")

    # Show DR stage overview when no image is uploaded
    st.markdown("---")
    st.subheader("📚 Diabetic Retinopathy — Stage Overview")

    for i, cls in enumerate(classes):
        info = dr_info[i]
        urgency_icon = urgency_colors.get(info["urgency"], "⚪")
        with st.expander(f"{urgency_icon} Stage {i}: {cls} — {info['urgency']}"):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Recommended Follow-up:** {info['follow_up']}")
