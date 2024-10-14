from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
from PIL import Image

# Load YuNet model for face detection with lower confidence threshold and larger input size
yunet = cv2.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx', 
    "",  # Placeholder for the configuration file
    (640, 640),  # Larger input size for better detection on smaller faces
    0.5,  # Lower confidence threshold for increased sensitivity
    0.3,  # NMS threshold
    5000  # Top K results
)

# Load SFace model for face recognition
sface = cv2.FaceRecognizerSF.create(
    'face_recognition_sface_2021dec.onnx', 
    ""
)

# Path to the training dataset
dataset_path = "./Dataset"  # Replace with your dataset path
embedding_file = 'face_embeddings.pkl'

# Dictionary to store embeddings with labels
face_embeddings = {}

# Non-Maximum Suppression function to filter out overlapping boxes
def apply_nms(boxes, threshold=0.5):
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # Sort by confidence
    picked_boxes = []

    while boxes:
        current = boxes.pop(0)
        picked_boxes.append(current)
        boxes = [box for box in boxes if iou(box, current) < threshold]

    return picked_boxes

def iou(box1, box2):
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / float(union_area)

# Loop through the dataset
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue  # Skip non-directory files

    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)

        # Open the image using PIL to handle potential corrupt JPEG data
        try:
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert("RGB")  # Convert to RGB if necessary
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format (BGR)
        except Exception as e:
            print(f"Failed to open {image_path} due to {e}")
            continue

        # Resize image to model input size for face detection
        yunet.setInputSize((img.shape[1], img.shape[0]))

        # Face Detection
        faces = yunet.detect(img)
        if faces[1] is not None:
            # Filter boxes with NMS
            boxes = [face[:5].astype(int) for face in faces[1]]  # Include confidence in box
            filtered_boxes = apply_nms(boxes, threshold=0.5)  # Apply NMS with a 0.5 threshold

            for box in filtered_boxes:
                x, y, w, h, conf = box

                # Extract face ROI
                face_roi = img[y:y+h, x:x+w]
                
                # Validate and obtain face embedding
                face_embedding = sface.feature(face_roi).flatten()
                
                # Check if embedding is valid
                if face_embedding.shape == (128,):
                    # Save cropped face image
                    cropped_image_path = f"cropped_faces/{label}_{image_file}"
                    os.makedirs("cropped_faces", exist_ok=True)
                    cv2.imwrite(cropped_image_path, face_roi)

                    # Add embedding to dictionary under the label
                    if label not in face_embeddings:
                        face_embeddings[label] = []
                    face_embeddings[label].append(face_embedding)

                    print(f"Processed {image_path} for label '{label}', and saved cropped face.")
                else:
                    print(f"Invalid embedding for {image_path}. Skipping this face.")
        else:
            print(f"No face detected in {image_path}.")

# Save the embeddings to a pickle file
with open(embedding_file, 'wb') as file:
    pickle.dump(face_embeddings, file)

print(f"Face embeddings saved to {embedding_file}.")
