import cv2
import numpy as np
import tensorflow as tf
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mtcnn import MTCNN

# Determine model directory (check local ./models/ first, then repo root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = BASE_DIR

xception_model_path = os.path.join(MODELS_DIR, "xception_model.h5")
efficientnet_model_path = os.path.join(MODELS_DIR, "efficientnet_model.h5")

# Load models
print("=" * 50)
print(" Deepfake Detection System - FaceForensics++")
print("=" * 50)
print("Loading model(s)...")

xception_model = None
if os.path.exists(xception_model_path):
    print("Loading Xception model...")
    xception_model = tf.keras.models.load_model(xception_model_path)

efficientnet_model = None
if os.path.exists(efficientnet_model_path):
    print("Loading EfficientNet-B0 (+Spatial Attention) model...")
    efficientnet_model = tf.keras.models.load_model(efficientnet_model_path)

if xception_model is None and efficientnet_model is None:
    raise FileNotFoundError(
        f"No model found in {MODELS_DIR}. Please place 'efficientnet_model.h5' or 'xception_model.h5' in {MODELS_DIR}"
    )

print("Initializing MTCNN face detector...")
detector = MTCNN()
print("Models and face detector ready!\n")

def predict_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None

    print(f"Analyzing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Could not read frames from video.")
        cap.release()
        return None

    frame_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
    predictions = []
    
    for i, idx in enumerate(frame_indices, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)
        
        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue
            face = cv2.resize(face, (224, 224))
            face_normalized = face / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            
            if xception_model is not None and efficientnet_model is not None:
                prob_xception = xception_model.predict(face_input, verbose=0)[0][0]
                prob_eff = efficientnet_model.predict(face_input, verbose=0)[0][0]
                prob = (0.5 * prob_xception) + (0.5 * prob_eff)
            elif efficientnet_model is not None:
                prob = efficientnet_model.predict(face_input, verbose=0)[0][0]
            else:
                prob = xception_model.predict(face_input, verbose=0)[0][0]

            predictions.append(float(prob))
            print(f"  Frame {i}/{len(frame_indices)} - Fake probability: {float(prob)*100:.2f}%")
    
    cap.release()
    
    if not predictions:
        print("\nNo faces detected in sampled frames.")
        return None
    
    avg_prob = float(np.mean(predictions))
    label = "FAKE" if avg_prob > 0.5 else "REAL"
    confidence = avg_prob * 100 if label == "FAKE" else (1 - avg_prob) * 100
    
    print("\n" + "=" * 40)
    print(f" RESULT SUMMARY")
    print("=" * 40)
    print(f" Faces Analyzed : {len(predictions)} frames")
    print(f" Classification : {label}")
    print(f" Confidence     : {confidence:.2f}%")
    print(f" Fake Likelihood: {avg_prob * 100:.2f}%")
    print("=" * 40)
    return label, confidence, avg_prob

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1].strip('"\'')
    else:
        sample_dir = r"C:\Users\KIIT0001\Downloads\archive (10)\FaceForensics++_C23\DeepFakeDetection"
        default_video = ""
        if os.path.exists(sample_dir):
            samples = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(".mp4")]
            if samples:
                default_video = samples[0]

        if not default_video:
            local_samples = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith(".mp4")]
            if local_samples:
                default_video = local_samples[0]

        prompt = "Enter video path"
        if default_video:
            prompt += f"\n[Press Enter to use sample: {os.path.basename(default_video)}]\n> "
        else:
            prompt += ": "

        user_input = input(prompt).strip('"\'')
        video_path = user_input if user_input else default_video

    if video_path:
        predict_video(video_path)
    else:
        print("No video path provided.")
