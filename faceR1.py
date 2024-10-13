import argparse
import numpy as np
import cv2 as cv
import onnxruntime as ort
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Fungsi untuk mengecek versi OpenCV
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install the latest opencv-python: python3 -m pip install --upgrade opencv-python"

from yunet import YuNet  # Asumsi `YuNet` ada dalam file yunet.py

# Kombinasi backend dan target
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

# Argument parser untuk mengatur parameter input
parser = argparse.ArgumentParser(description='YuNet and FaceNet Integration')
parser.add_argument('--input', '-i', type=str, help='Set input to a specific image; omit if using the camera.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2023mar.onnx', help="Set YuNet model file.")
parser.add_argument('--facenet_model', type=str, default='facenet_model.onnx', help="Path to FaceNet ONNX model.")
parser.add_argument('--conf_threshold', type=float, default=0.9, help='Set minimum confidence for face detection.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes with IOU >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--embedding_output', type=str, default='embeddings.pkl', help="Specify the filename for saved embeddings.")
parser.add_argument('--save', '-s', action='store_true', help='Save output file with bounding boxes and confidence levels.')
parser.add_argument('--vis', '-v', action='store_true', help='Visualize results in a new window.')
args = parser.parse_args()

# Fungsi untuk visualisasi
def visualize(image, results, labels, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det, label in zip(results, labels):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
        conf = det[-1]
        label_text = f"{label} ({conf:.2f})"
        cv.putText(output, label_text, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

# Fungsi untuk ekstraksi embedding menggunakan FaceNet
def extract_embedding(face, ort_session):
    face = cv.resize(face, (160, 160))
    face = face.astype('float32')
    face = (face - 127.5) / 128.0  # Normalisasi
    face = np.transpose(face, (2, 0, 1))  # Mengubah dari (160, 160, 3) ke (3, 160, 160)
    face = np.expand_dims(face, axis=0)   # Tambahkan dimensi batch menjadi (1, 3, 160, 160)
    embedding = ort_session.run(None, {'input': face})
    return embedding[0]

# Fungsi untuk mengenali wajah berdasarkan embedding
def recognize_face(embedding, embeddings_db, threshold=0.6):
    best_match = "Unknown"
    best_score = threshold

    for name, db_embedding in embeddings_db.items():
        similarity = cosine_similarity(embedding.reshape(1, -1), db_embedding.reshape(1, -1))[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    return best_match

if __name__ == '__main__':
    backend_id = backend_target_pairs[0][0]
    target_id = backend_target_pairs[0][1]

    # Load model YuNet
    model = YuNet(modelPath=args.model,
                  inputSize=[320, 320],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  backendId=backend_id,
                  targetId=target_id)

    # Load FaceNet model dengan ONNX Runtime
    facenet_session = ort.InferenceSession(args.facenet_model)

    # Load embeddings database from .pkl file
    try:
        with open(args.embedding_output, 'rb') as f:
            embeddings_db = pickle.load(f)
    except FileNotFoundError:
        print(f"{args.embedding_output} not found. Initializing an empty database.")
        embeddings_db = {}  # Inisialisasi sebagai dictionary kosong

    if args.input:
        image = cv.imread(args.input)
        h, w, _ = image.shape
        model.setInputSize([w, h])
        results = model.infer(image)

        labels = []
        for det in results:
            x1, y1, w, h = det[0:4].astype(int)
            face = image[y1:y1+h, x1:x1+w]
            embedding = extract_embedding(face, facenet_session)
            label = recognize_face(embedding, embeddings_db)
            labels.append(label)

        image = visualize(image, results, labels)

        if args.save:
            cv.imwrite('result.jpg', image)

        if args.vis:
            cv.imshow('YuNet Demo', image)
            cv.waitKey(0)
    else:
        cap = cv.VideoCapture(0)
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        model.setInputSize([w, h])

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break

            tm.start()
            results = model.infer(frame)
            tm.stop()

            labels = []
            for det in results:
                x1, y1, w, h = det[0:4].astype(int)
                face = frame[y1:y1+h, x1:x1+w]
                embedding = extract_embedding(face, facenet_session)
                label = recognize_face(embedding, embeddings_db)
                labels.append(label)

            frame = visualize(frame, results, labels, fps=tm.getFPS())
            cv.imshow('YuNet Demo', frame)
            tm.reset()

        cap.release()
        cv.destroyAllWindows()
