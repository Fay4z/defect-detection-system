import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
from pathlib import Path

@st.cache_resource
def load_model(model_path="model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, image, input_size=(224, 224)):
    img = cv2.resize(image, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction[0][0]

def classify_frame(frame, interpreter):
    score = predict(interpreter, frame)
    if score > 0.5:
        label = f"Normal ✅ ({score:.2f})"
        color = (0, 255, 0)
    else:
        label = f"Defected ❌ ({score:.2f})"
        color = (255, 0, 0)
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

st.title("📷 Product Defect Detection")
st.markdown("Detect whether a product is **Normal** or **Defected** using a TFLite model.")

model_path = st.text_input("Enter TFLite model path:", "model.tflite")
if model_path:
    interpreter = load_model(model_path)

    option = st.selectbox("Choose Input Method", ["Upload Image", "Upload Video", "Use Webcam"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_img = classify_frame(img_rgb.copy(), interpreter)
            st.image(result_img, channels="RGB")

    elif option == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frame = classify_frame(frame_rgb.copy(), interpreter)
                stframe.image(result_frame, channels="RGB")

            cap.release()

    elif option == "Use Webcam":
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            ret, frame = camera.read()
            if not ret:
                st.write("❌ Could not access webcam.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_frame = classify_frame(frame_rgb.copy(), interpreter)
            FRAME_WINDOW.image(result_frame)

        camera.release()
