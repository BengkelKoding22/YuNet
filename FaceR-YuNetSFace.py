import cv2
import numpy as np
import pickle
import os

# Load YuNet model for face detection
yunet = cv2.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx', 
    "",  # Placeholder for the configuration file
    (320, 320),  # Input size (width, height)
    0.5,  # Confidence threshold
    0.3,  # NMS threshold
    5000  # Top K results
)

# Load SFace model for face recognition
sface = cv2.FaceRecognizerSF.create(
    'face_recognition_sface_2021dec.onnx', 
    ""
)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the frame width and height to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def cosine_similarity(embedding1, embedding2):
    # Flatten embeddings to ensure they are 1D vectors of the same size
    embedding1 = np.array(embedding1).flatten()
    embedding2 = np.array(embedding2).flatten()
    
    # Ensure both embeddings have the correct shape (128,)
    if embedding1.shape != (128,) or embedding2.shape != (128,):
        print(f"Invalid embedding shape: {embedding1.shape} vs {embedding2.shape}")
        raise ValueError("Embedding shapes are not aligned for cosine similarity calculation")
    
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Load known face embeddings from pickle file (if exists)
embedding_file = 'face_embeddings.pkl'
if os.path.exists(embedding_file):
    with open(embedding_file, 'rb') as file:
        known_face_embeddings = pickle.load(file)  # Now it's a dictionary
    
    # Ensure all embeddings have the correct shape
    for label, embeddings in known_face_embeddings.items():
        corrected_embeddings = []
        for embedding in embeddings:
            # Reshape embedding if necessary
            embedding = np.array(embedding).flatten()
            if embedding.shape == (128,):
                corrected_embeddings.append(embedding)
            else:
                print(f"Warning: Found embedding of unexpected shape {embedding.shape} for label '{label}'")
        
        # Update with corrected embeddings
        known_face_embeddings[label] = corrected_embeddings

    print("Known face embeddings loaded and validated.")
else:
    known_face_embeddings = None
    print("No known face embeddings found. Please add some to the file.")

# Set a threshold for recognition
similarity_threshold = 0.5  # Cosine similarity threshold

# Real-time face detection and recognition
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize image to model input size for face detection
    yunet.setInputSize((frame.shape[1], frame.shape[0]))

    # Face Detection
    faces = yunet.detect(frame)
    if faces[1] is not None:
        for face in faces[1]:
            box = face[:4].astype(int)  # x, y, w, h
            confidence = face[-1]

            # Draw bounding box around detected face
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

            # Check if bounding box is within image bounds
            x, y, w, h = box
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                print("Warning: Bounding box is out of image bounds. Skipping this detection.")
                continue
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Check if face ROI is not empty and has sufficient size
            if face_roi.size == 0 or w < 10 or h < 10:  # Adjust size threshold as needed
                print("Warning: Detected face ROI is empty or too small. Skipping this detection.")
                continue

            # Get face embedding
            face_embedding = sface.feature(face_roi).flatten()

            # Check embedding shape
            if face_embedding.shape != (128,):
                print(f"Error: Face embedding shape mismatch - got shape {face_embedding.shape}")
                continue  # Skip this embedding if shape is incorrect

            # Initialize variables for label and max similarity
            label = "Unknown"
            max_similarity = 0

            # Only attempt recognition if we have known embeddings
            if known_face_embeddings is not None:
                for known_label, embeddings in known_face_embeddings.items():
                    for known_embedding in embeddings:
                        similarity = cosine_similarity(face_embedding, known_embedding)
                        if similarity > max_similarity and similarity > similarity_threshold:
                            max_similarity = similarity
                            label = f"{known_label} ({similarity:.2f})"

            # Display the label
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Display the frame with detections
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
