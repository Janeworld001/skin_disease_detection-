import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🧴",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #f5ede3 0%, #ffffff 100%);
}

/* Title */
.title {
    text-align: center;
    color: #5e270b;
    font-size: 42px;
    font-weight: 600;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #6e4b3a;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(94, 39, 11, 0.08);
    border: 1px solid #f0e6dc;
}

/* Section titles */
.section-title {
    color: #5e270b;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Button */
.stButton > button {
    background-color: #5e270b;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #7a3b17;
    color: white;
}
            
# [data-testid="stFileUploader"] button {
#     display: none;
# }

/* Progress bar color */
div[data-testid="stProgress"] > div > div {
    background-color: #c69b6f;
}
            
.card-header {
    background-color: white;
    color: #5e270b;
    padding: 12px;
    border-radius: 16px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 10px;
}
            
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title"> Skin Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to detect skin conditions using AI</div>', unsafe_allow_html=True)

st.write("")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("Correcr_skin_disease_model.pt")

model = load_model()

# ---------- UPLOAD SECTION ----------
st.markdown("""
<div style="
    color:#5e270b;
    font-size:18px;
    font-weight:600;
    margin-bottom:-25px;
">
Upload Skin Image
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# ---------- MAIN ----------
if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])



    # ---------- IMAGE CARD ----------
    with col1:
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📷 Uploaded Image</div>', unsafe_allow_html=True)

        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PREDICTION CARD ----------
    with col2:
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🔍 Prediction Panel</div>', unsafe_allow_html=True)

        if st.button("Run Prediction", use_container_width=True):

            with st.spinner("Analyzing image..."):
                results = model.predict(image)

                probs = results[0].probs
                names = results[0].names

                class_id = probs.top1
                confidence = probs.top1conf.item()
                label = names[class_id]

                # ---------- RESULT (CUSTOM) ----------
                st.markdown(f"""
                <div style="
                    background:#f5ede3;
                    padding:12px;
                    border-radius:10px;
                    color:#5e270b;
                    font-weight:600;
                    margin-bottom:10px;
                ">
                Prediction: {label}
                </div>
                """, unsafe_allow_html=True)

                # ---------- CONFIDENCE ----------
                st.markdown('<div class="section-title"> Confidence</div>', unsafe_allow_html=True)
                st.progress(float(confidence))

                st.markdown(f"""
                <div style="color:#6e4b3a; font-size:14px; margin-bottom:10px;">
                Confidence: {confidence:.2%}
                </div>
                """, unsafe_allow_html=True)

                # ---------- TOP 3 ----------
                st.markdown('<div class="section-title"> Top Predictions</div>', unsafe_allow_html=True)

                top_ids = np.argsort(probs.data.cpu().numpy())[::-1][:3]

                for i in top_ids:
                    st.markdown(f"""
                    <div style="
                        background:#faf6f1;
                        padding:8px;
                        border-radius:8px;
                        margin-bottom:5px;
                        color:#5e270b;
                    ">
                    {names[i]} — {probs.data[i]:.2%}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


else:
    st.markdown("""
    <div style="
        background:#f5ede3;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:#6e4b3a;
        font-size:18px;
        border:1px dashed #c69b6f;
    ">
    Upload a skin image to start analysis
    </div>
    """, unsafe_allow_html=True)