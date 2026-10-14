import streamlit as st
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

st.title("⚡ Fast Face Registration & Verification (InsightFace)")

# -------------------------
# Load model only once
# -------------------------
@st.cache_resource
def load_model():
    with st.spinner("Loading Face Model..."):
        app = FaceAnalysis(name="buffalo_sc")
        app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode
        return app

model = load_model()

# -------------------------
# Temporary in-memory storage
# -------------------------
if "face_db" not in st.session_state:
    st.session_state.face_db = {}

option = st.radio("Choose Option", ["Register", "Verify"])

# -------------------------
# Image Input Options
# -------------------------
input_type = st.radio("Select Image Source", ["Upload Image", "Use Camera"])

image = None

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif input_type == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# -------------------------
# Process Image
# -------------------------
if image is not None:

    image_np = np.array(image)

    with st.spinner("Detecting face..."):
        faces = model.get(image_np)

    if len(faces) == 0:
        st.error("No face detected!")
    else:
        embedding = faces[0].embedding

        # -------------------------
        # REGISTER
        # -------------------------
        if option == "Register":
            name = st.text_input("Enter Name")

            if st.button("Register User"):
                if name.strip() == "":
                    st.warning("Please enter a valid name.")
                else:
                    with st.spinner("Registering user..."):
                        st.session_state.face_db[name] = embedding
                    st.success(f"{name} registered successfully!")

        # -------------------------
        # VERIFY
        # -------------------------
        elif option == "Verify":
            if st.button("Verify User"):

                if len(st.session_state.face_db) == 0:
                    st.warning("No users registered yet!")
                else:
                    with st.spinner("Verifying user..."):

                        best_match = None
                        best_score = -1

                        for stored_name, stored_embedding in st.session_state.face_db.items():
                            similarity = cosine_similarity(
                                [embedding],
                                [stored_embedding]
                            )[0][0]

                            if similarity > best_score:
                                best_score = similarity
                                best_match = stored_name

                    if best_score > 0.5:
                        st.success(f"Recognized as: {best_match}")
                        
                    else:
                        st.error("No matching user found")

                    st.write(f"Similarity Score: {best_score:.2f}")