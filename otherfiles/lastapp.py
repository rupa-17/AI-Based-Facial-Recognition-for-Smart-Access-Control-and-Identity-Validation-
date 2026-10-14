import streamlit as st
import numpy as np
import os
import json
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(page_title="Face Recognition System", layout="centered")

os.environ["INSIGHTFACE_HOME"] = "./models"

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "faces.json")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({}, f)

# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def load_face_db():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_face_db(db):
    with open(DATA_FILE, "w") as f:
        json.dump(db, f, indent=4)

# =========================================================
# LOAD MODEL (CACHED)
# =========================================================

@st.cache_resource
def load_model():
    with st.spinner("Loading Face Recognition Model..."):
        app = FaceAnalysis(name="buffalo_sc")
        app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode
        return app

model = load_model()

# =========================================================
# UI
# =========================================================

st.title("⚡ Persistent Face Registration & Verification")

face_db = load_face_db()

option = st.radio("Select Operation", ["Register", "Verify", "Delete User"])

# =========================================================
# IMAGE INPUT
# =========================================================

if option != "Delete User":

    input_type = st.radio("Select Image Source", ["Upload Image", "Use Camera"])

    image = None

    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_type == "Use Camera":
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    # =========================================================
    # PROCESS IMAGE
    # =========================================================

    if image is not None:

        # Show image in small centered box
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if option == "Register":
                st.image(image, caption="Image for Registration", width=250)
            elif option == "Verify":
                st.image(image, caption="Image for Verification", width=250)

        image_np = np.array(image)

        with st.spinner("Detecting face..."):
            faces = model.get(image_np)

        if len(faces) == 0:
            st.error("No face detected!")
        else:
            embedding = faces[0].embedding

            # =========================================================
            # REGISTER
            # =========================================================

            if option == "Register":
                name = st.text_input("Enter Name")

                if st.button("Register User"):
                    if name.strip() == "":
                        st.warning("Enter a valid name.")
                    else:
                        with st.spinner("Registering user..."):
                            face_db[name] = embedding.tolist()
                            save_face_db(face_db)
                        st.success(f"{name} registered successfully!")

            # =========================================================
            # VERIFY
            # =========================================================

            elif option == "Verify":

                if st.button("Verify User"):

                    if len(face_db) == 0:
                        st.warning("No users registered yet!")
                    else:
                        with st.spinner("Verifying user..."):

                            best_match = None
                            best_score = -1

                            for stored_name, stored_embedding in face_db.items():
                                similarity = cosine_similarity(
                                    [embedding],
                                    [np.array(stored_embedding)]
                                )[0][0]

                                if similarity > best_score:
                                    best_score = similarity
                                    best_match = stored_name

                        if best_score > 0.4:
                            st.success(f"Recognized as: {best_match}")
                        else:
                            st.error("No matching user found")

                        st.write(f"Similarity Score: {best_score:.2f}")

# =========================================================
# DELETE USER
# =========================================================

if option == "Delete User":

    if len(face_db) == 0:
        st.warning("No users available to delete.")
    else:
        user_to_delete = st.selectbox("Select User", list(face_db.keys()))

        if st.button("Delete User"):
            with st.spinner("Deleting user..."):
                del face_db[user_to_delete]
                save_face_db(face_db)
            st.success(f"{user_to_delete} deleted successfully!")