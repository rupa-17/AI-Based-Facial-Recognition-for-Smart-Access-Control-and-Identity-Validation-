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

st.set_page_config(page_title="Face Recognition System", layout="wide")

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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_sc")
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app

model = load_model()
face_db = load_face_db()

# =========================================================
# SIDEBAR (LEFT)
# =========================================================

st.sidebar.title("⚙ Controls")

option = st.sidebar.radio(
    "Select Operation",
    ["Register", "Verify", "Delete User"]
)

input_type = None
if option != "Delete User":
    input_type = st.sidebar.radio(
        "Select Image Source",
        ["Upload Image", "Use Camera"]
    )

# =========================================================
# MAIN AREA (RIGHT)
# =========================================================

st.title("⚡ AI Based Facial Recognition for Smart Access Control and Identity Validaton")

image = None

if option != "Delete User":

    # Show uploader OR camera on RIGHT based on sidebar selection
    if input_type == "Upload Image":
        uploaded_file = st.file_uploader(
            "Drag and Drop Image Here",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_type == "Use Camera":
        camera_image = st.camera_input("Live Camera Capture")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    if image is not None:

        # Small centered preview
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, width=250)

        image_np = np.array(image)

        with st.spinner("Detecting face..."):
            faces = model.get(image_np)

        if len(faces) == 0:
            st.error("No face detected!")
        else:
            embedding = faces[0].embedding

            # REGISTER
            if option == "Register":
                name = st.text_input("Enter Name")

                if st.button("Register User"):
                    if name.strip() == "":
                        st.warning("Enter a valid name.")
                    else:
                        face_db[name] = embedding.tolist()
                        save_face_db(face_db)
                        st.success(f"{name} registered successfully!")

            # VERIFY
            elif option == "Verify":

                if st.button("Verify User"):

                    if len(face_db) == 0:
                        st.warning("No users registered yet!")
                    else:
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

    st.subheader("Delete Registered User")

    if len(face_db) == 0:
        st.warning("No users available to delete.")
    else:
        user_to_delete = st.selectbox(
            "Select User",
            list(face_db.keys())
        )

        if st.button("Delete User"):
            del face_db[user_to_delete]
            save_face_db(face_db)
            st.success(f"{user_to_delete} deleted successfully!")