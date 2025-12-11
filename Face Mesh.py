import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split

# Load FER-2013
df = pd.read_csv('fer2013.csv')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

data = []
labels = []

for idx, row in df.iterrows():
    pixels = np.fromstring(row['pixels'], sep=' ').reshape(48, 48).astype(np.uint8)
    image = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            landmark_points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
            data.append(landmark_points.flatten())
            labels.append(row['emotion'])
    
    if idx % 1000 == 0:
        print(f'Processed {idx} images')

# Save to npy
np.save('data.npy', np.array(data))
np.save('labels.npy', np.array(labels))
