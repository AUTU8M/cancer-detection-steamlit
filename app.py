import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Constants
IMAGE_SIZE = (350, 350)  # Updated to match model input
CLASS_LABELS = {
    0: "Adenocarcinoma",
    1: "Large Cell Carcinoma", 
    2: "Normal",
    3: "Squamous Cell Carcinoma"
}

# CSS for results box
RESULTS_BOX_STYLE = """
    border: 0px solid #e0e0e0;
    border-radius: 8px;
    padding: 20px;
    background-color: transparent;
    width: 100%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
"""

@st.cache_resource
def load_cached_model():
    return load_model("trained_lung_cancer_model.h5")

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def main():
    st.set_page_config(page_title="Lung Cancer Detector", page_icon="🫁", layout="centered")

    st.markdown(f"""
    <style>
        .results-box {{
            {RESULTS_BOX_STYLE}
        }}
        .stImage > img {{
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            object-fit: contain;
            max-height: 400px;
        }}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        show_confidence = st.checkbox("Show confidence score", value=True)
        st.markdown("---")
        st.markdown("**Note:** This tool assists but doesn't replace professional diagnosis.")

    # Title and file uploader
    st.title("Lung Cancer Detector")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CT scan image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file:
        col1, col2 = st.columns([1.4, 1])  # Side-by-side layout

        with col1:
            st.image(uploaded_file, caption="Uploaded CT Scan", use_container_width=True)

        with col2:
            with st.spinner("Analyzing..."):
                try:
                    model = load_cached_model()
                    processed_img = preprocess_image(uploaded_file)
                    prediction = model.predict(processed_img)

                    pred_class = np.argmax(prediction)
                    confidence = np.max(prediction) * 100

                    st.markdown('<div class="results-box">', unsafe_allow_html=True)

                    # Title area
                    st.markdown("""
                    <h3 style="margin-top: 0; margin-bottom: 10px;">🧪 Diagnosis Report</h3>
                    """, unsafe_allow_html=True)

                    # Custom smaller text
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <h5 style="margin: 0 0 4px; font-size: 25px;">Prediction</h5>
                        <p style="font-size: 15px; font-weight: 400; color: #fff;">{CLASS_LABELS[pred_class]}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if show_confidence:
                        st.markdown(f"""
                        <div style="margin-bottom: 8px;">
                            <h5 style="margin: 0 0 4px; font-size: 25px;">Confidence</h5>
                            <p style="font-size: 15px; font-weight: 400; color: #fff;">{confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

                    if pred_class == 2:
                        st.success("No signs of malignancy detected")

                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    # Cancer Info
    with st.expander("ℹ️ About Cancer Types", expanded=False):
        st.markdown("""
        **Lung Cancer Types:**
        - **Adenocarcinoma**: Most common type, often found in outer lung areas.
        - **Squamous Cell Carcinoma**: Linked to smoking, found in central lungs.
        - **Large Cell Carcinoma**: Rare but aggressive.
        - **Normal**: Healthy lung tissue.
        """)

if __name__ == "__main__":
    main()
