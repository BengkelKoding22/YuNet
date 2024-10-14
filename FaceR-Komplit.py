import cv2
import numpy as np
import pickle
import os
import argparse

from yunet import YuNet
from sface import SFace
from facial_fer_model import FacialExpressionRecog  # Pastikan ini sesuai dengan file Anda

# Kombinasi backend dan target
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# Argument parser untuk mengatur parameter input
parser = argparse.ArgumentParser(description='YuNet, SFace and Facial Expression Recognition')
parser.add_argument('--input', '-i', type=str, help='Set input to a specific image; omit if using the camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2023mar.onnx', help="Set YuNet model file.")
parser.add_argument('--face_model', type=str, default='face_recognition_sface_2021dec.onnx', help="Set SFace model file.")
parser.add_argument('--expression_model', type=str, default='facial_expression_recognition_mobilefacenet_2022july.onnx', help="Set Facial Expression Recognition model file.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pairs:
                        {:d}: (default) OpenCV + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Set minimum confidence for face detection.')
args = parser.parse_args()

# Tentukan backend dan target berdasarkan parameter
backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]

# Buat instance dari kelas YuNet dengan pengaturan backend dan target
yunet = YuNet(
    modelPath=args.model,
    inputSize=[320, 320],  # Ukuran input untuk deteksi wajah
    confThreshold=args.conf_threshold,
    nmsThreshold=0.3,
    topK=5000,
    backendId=backend_id,
    targetId=target_id
)

# Buat instance dari kelas SFace dengan pengaturan backend dan target
sface = SFace(
    modelPath=args.face_model,
    backendId=backend_id,
    targetId=target_id
)

# Buat instance dari FacialExpressionRecog
emotion_recognition = FacialExpressionRecog(modelPath=args.expression_model, backendId=backend_id, targetId=target_id)

# Initialize the webcam if input is not specified
cap = cv2.VideoCapture(args.input if args.input else 0)

# Set the frame width and height to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

# Load known face embeddings from the saved file
embedding_file = 'saved_embeddings.pkl'  # File untuk embeddings yang disimpan
known_face_embeddings = {}

if os.path.exists(embedding_file):
    with open(embedding_file, 'rb') as file:
        known_face_embeddings = pickle.load(file)
    print("Known face embeddings loaded from saved file.")
else:
    print("No saved embeddings found. Starting with empty embeddings.")

# Flag untuk pause
paused = False

# Main loop for real-time face detection, recognition, and emotion detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    if not paused:
        yunet.setInputSize((frame.shape[1], frame.shape[0]))
        faces = yunet.infer(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h, conf = face[:5].astype(int)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue

                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0 or w < 10 or h < 10:
                    continue

                # Get face embedding using SFace
                face_embedding = sface.infer(frame, face).flatten()
                if face_embedding.shape != (128,):
                    continue

                label = "Unknown"
                max_similarity = 0

                if known_face_embeddings:
                    for known_label, embeddings in known_face_embeddings.items():
                        for known_embedding in embeddings:
                            similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
                            if similarity > max_similarity and similarity > args.conf_threshold:
                                max_similarity = similarity
                                label = f"{known_label} ({similarity:.2f})"

                # Emotion Recognition
                input_blob = cv2.dnn.blobFromImage(face_roi, 1/255.0, (112, 112))  # Resize to the expected input size for your model
                emotion_recognition._model.setInput(input_blob)
                emotion_preds = emotion_recognition._model.forward()
                emotion_id = np.argmax(emotion_preds)

                # Define emotion labels (adjust according to your model)
                emotion_labels = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
                emotion_label = emotion_labels[emotion_id] if emotion_id < len(emotion_labels) else "Unknown"

                # Combine name label and emotion
                full_label = f"{label} - {emotion_label}"

                cv2.putText(frame, full_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("Face and Emotion Recognition", frame)

    key = cv2.waitKey(1)

    # Kontrol tombol
    if key == ord('p'):  # Tombol untuk pause
        paused = not paused
        if paused:
            print("Paused. Press any key to save embeddings.")
            label = input("Enter label for the embedding: ")  # Input label dari pengguna
    elif paused and key != -1:  # Saat dalam kondisi pause, simpan embeddings
        # Simpan embeddings saat di pause
        if label != "Unknown":
            if label not in known_face_embeddings:
                known_face_embeddings[label] = []
            known_face_embeddings[label].append(face_embedding)

            with open(embedding_file, 'wb') as file:
                pickle.dump(known_face_embeddings, file)
            print(f"Saved embeddings for {label}.")

    if key == ord('q'):  # Tombol untuk keluar
        break

cap.release()
cv2.destroyAllWindows()