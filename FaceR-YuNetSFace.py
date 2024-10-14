import cv2
import numpy as np
import pickle
import os
import argparse

# Kombinasi backend dan target
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]

# Argument parser untuk mengatur parameter input
parser = argparse.ArgumentParser(description='YuNet and SFace Face Recognition')
parser.add_argument('--input', '-i', type=str, help='Set input to a specific image; omit if using the camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2023mar.onnx', help="Set YuNet model file.")
parser.add_argument('--face_model', type=str, default='face_recognition_sface_2021dec.onnx', help="Set SFace model file.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pairs:
                        {:d}: (default) OpenCV + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.5, help='Set minimum confidence for face detection.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes with IOU >= nms_threshold.')
parser.add_argument('--embedding_output', type=str, default='face_embeddings.pkl', help="Specify the filename for saved embeddings.")
parser.add_argument('--save', '-s', action='store_true', help='Save output file with bounding boxes and confidence levels.')
args = parser.parse_args()

# Tentukan backend dan target berdasarkan parameter
backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]

# Load YuNet model for face detection
yunet = cv2.FaceDetectorYN.create(
    args.model,
    "",  # Placeholder for the configuration file
    (640, 640),  # Input size (width, height)
    args.conf_threshold,  # Confidence threshold
    args.nms_threshold,  # NMS threshold
    5000  # Top K results
)

# Load SFace model for face recognition
sface = cv2.FaceRecognizerSF.create(
    args.face_model,
    ""
)

# Initialize the webcam if input is not specified
cap = cv2.VideoCapture(args.input if args.input else 0)

# Set the frame width and height to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

# Load known face embeddings
embedding_file = args.embedding_output
if os.path.exists(embedding_file):
    with open(embedding_file, 'rb') as file:
        known_face_embeddings = pickle.load(file)
    print("Known face embeddings loaded and validated.")
else:
    known_face_embeddings = None
    print("No known face embeddings found.")

# Main loop for real-time face detection and recognition
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    yunet.setInputSize((frame.shape[1], frame.shape[0]))
    faces = yunet.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h, conf = face[:5].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue

            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0 or w < 10 or h < 10:
                continue

            face_embedding = sface.feature(face_roi).flatten()
            if face_embedding.shape != (128,):
                continue

            label = "Unknown"
            max_similarity = 0

            if known_face_embeddings is not None:
                for known_label, embeddings in known_face_embeddings.items():
                    for known_embedding in embeddings:
                        similarity = np.dot(face_embedding, known_embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding))
                        similarity = similarity.item() if np.size(similarity) == 1 else similarity
                        if similarity > max_similarity and similarity > args.conf_threshold:
                            max_similarity = similarity
                            label = f"{known_label} ({similarity:.2f})"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if args.save:
        cv2.imwrite("output.jpg", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
