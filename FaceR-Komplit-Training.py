import cv2
import numpy as np
import pickle
import os
import argparse
from PIL import Image
import pillow_heif  # For HEIC support with Pillow
from yunet import YuNet
from sface import SFace
from facial_fer_model import FacialExpressionRecog

# Kombinasi backend dan target
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU]
]

# Argument parser untuk mengatur parameter input
parser = argparse.ArgumentParser(description='YuNet, SFace and Facial Expression Recognition')
parser.add_argument('--folder', '-f', type=str, default='./FotoLanyard', help='Folder path containing images to process.')
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

# Load known face embeddings from the saved file
embedding_file = 'saved_embeddings.pkl'  # File untuk embeddings yang disimpan
known_face_embeddings = {}

if os.path.exists(embedding_file):
    with open(embedding_file, 'rb') as file:
        known_face_embeddings = pickle.load(file)
    print("Known face embeddings loaded from saved file.")
else:
    print("No saved embeddings found. Starting with empty embeddings.")

# Process each image in the specified folder
for filename in os.listdir(args.folder):
    filepath = os.path.join(args.folder, filename)

    # Skip non-image files
    if not (filename.lower().endswith('.heic') or filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
        continue

    # Load image and convert if HEIC
    if filename.lower().endswith('.heic'):
        heif_image = pillow_heif.read_heif(filepath)
        image = Image.frombytes(
            heif_image.mode, 
            heif_image.size, 
            heif_image.data,
            "raw"
        )
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(filepath)

    if image is None:
        print(f"Error: Could not load image {filename}.")
        continue

    # Run YuNet to detect faces in the image
    yunet.setInputSize((image.shape[1], image.shape[0]))
    faces = yunet.infer(image)

    if faces is not None:
        # Find the largest face in the image
        largest_face = max(faces, key=lambda face: face[2] * face[3])

        x, y, w, h, conf = largest_face[:5].astype(int)
        face_roi = image[y:y+h, x:x+w]

        # Get face embedding using SFace
        face_embedding = sface.infer(image, largest_face).flatten()

        if face_embedding.shape == (128,):
            label = os.path.splitext(filename)[0]  # Use filename (without extension) as the label
            if label not in known_face_embeddings:
                known_face_embeddings[label] = []
            known_face_embeddings[label].append(face_embedding)

            print(f"Saved embeddings for {label}.")
        else:
            print(f"Error: Could not generate a valid face embedding for {filename}.")
    else:
        print(f"No face detected in {filename}.")

# Save the updated embeddings
with open(embedding_file, 'wb') as file:
    pickle.dump(known_face_embeddings, file)
print("All embeddings saved.")
