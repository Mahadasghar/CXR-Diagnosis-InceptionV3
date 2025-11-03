import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# --- ARCHITECTURE RECREATION IMPORTS ---
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Rescaling, Normalization, Input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
# ---------------------------------------

st.set_page_config(
    page_title="High-Sensitivity CXR Diagnostic AI",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- CONFIGURATION ---
# CRITICAL UPDATE: The model is now a weights file.
MODEL_PATH = 'best_cxr_model.weights.h5' 
OPTIMAL_THRESHOLD = 0.45 
IMG_SHAPE = (299, 299, 3) # InceptionV3 input size


# --- MODEL ARCHITECTURE RECREATION ---
def create_model_architecture():
    """Rebuilds the Keras Functional API model structure for inference."""
    
    # We load the InceptionV3 base model structure without its pre-trained weights initially
    base_model = InceptionV3(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights=None 
    )
    
    # Define the input layer
    inputs = Input(shape=IMG_SHAPE)

    # 1. Rescaling (0-255 to 0-1): This layer was in your notebook before the base model.
    x = Rescaling(1./255, name="rescaling_input")(inputs)

    # 2. Normalization (0-1 to -1 to 1): This layer was in your notebook before the base model.
    # NOTE: The augmentation layers (RandomRotation, etc.) are omitted as they are only for training.
    x = Normalization(mean=[0.5, 0.5, 0.5], variance=[0.25, 0.25, 0.25], name="inceptionv3_normalization")(x)

    # 3. Base Model Feature Extraction
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)

    # 4. Classification Head (Matches your final layers)
    x = Dropout(0.2)(x) 
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="CXR_Classifier")
    return model


# --- FUNCTIONS ---
@st.cache_resource # Cache the model loading for speed
def load_model():
    """Loads the model architecture and then the saved weights."""
    st.info("Rebuilding model architecture and loading weights...")
    try:
        # 1. Create the empty model structure
        model = create_model_architecture()
        
        # 2. Load the saved weights (from model.save_weights())
        model.load_weights(MODEL_PATH)
        
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.code(f"Please ensure '{MODEL_PATH}' is in the directory and the model architecture is correct.", language='text') 
        st.stop()


def preprocess_image(img, target_size=(299, 299)):
    """Pre-processes the uploaded image for input to the saved Keras model."""
    
    # Only performs PIL to NumPy conversion and resizing.
    # The Rescaling and Normalization layers are handled inside the Keras model.
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # We remove the old tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

# --- STREAMLIT APP LAYOUT ---

# Load the model outside the main function
model = load_model()



st.title("🩺 High-Sensitivity Chest X-Ray (CXR) Diagnostic Model")
st.markdown("### Powered by InceptionV3 and Optimized for Clinical Sensitivity")

st.sidebar.header("Input Image")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CXR Image (JPG or PNG)", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.sidebar.image(img, caption='Uploaded CXR Image', use_column_width=True)

    # Prediction button
    if st.sidebar.button("Run Prediction"):
        with st.spinner('Analyzing Image...'):
            try:
                # 1. Preprocess (Resizing/Array Conversion)
                processed_img = preprocess_image(img)

                # 2. Predict (Model handles its own internal normalization)
                prediction = model.predict(processed_img)[0][0]
                
                # 3. Apply Threshold (Your key contribution!)
                if prediction >= OPTIMAL_THRESHOLD:
                    result_text = "POSITIVE (Anomaly Detected)"
                    st.error(f"⚠️ Diagnosis: {result_text}")
                else:
                    result_text = "NEGATIVE (Clear)"
                    st.success(f"✅ Diagnosis: {result_text}")

                # 4. Display Details
                st.subheader("Model Output Details")
                st.metric(
                    label="Model Confidence Score (0-1)", 
                    value=f"{prediction:.4f}"
                )
                st.info(f"Using Optimal Threshold: **{OPTIMAL_THRESHOLD}** (Selected to maximize Sensitivity/Recall).")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

st.markdown(
    """
    ---
    **Model Note:** This model utilizes a Transfer Learning approach with the InceptionV3 architecture. The diagnostic classification is based on an optimized prediction cutoff of **0.45** to ensure maximum detection of true positive cases (high sensitivity).
    """
)