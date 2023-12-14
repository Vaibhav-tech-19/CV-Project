import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.sleep = 0
        self.drowsy = 0
        self.active = 0  # Initialize active attribute
        self.status = ""
        self.color = (0, 0, 0)
    
    def __del__(self):
        self.cap.release()

    def get_frame(self):
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        face_frame = frame.copy()
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = self.predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)
            
            left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                self.sleep += 1
                self.drowsy = 0
                self.active = 0
                if self.sleep > 6:
                    self.status = "SLEEPING !!!"
                    self.color = (255, 0, 0)

            elif left_blink == 1 or right_blink == 1:
                self.sleep = 0
                self.active = 0
                self.drowsy += 1
                if self.drowsy > 6:
                    self.status = "Drowsy !"
                    self.color = (0, 0, 255)

            else:
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                if self.active > 6:
                    self.status = "Active :)"
                    self.color = (0, 255, 0)

            cv2.putText(face_frame, self.status, (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color, 3)

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        return cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB), self.status

video_camera = VideoCamera()

# st.title('Realtime Drowsiness Detection System 📷')
st.markdown(
    "<h1 style='color: orange;'>Realtime Drowsiness Detection System 📷</h1>",
    unsafe_allow_html=True
)
st.markdown("**Computer Vision ESE - 2023**", unsafe_allow_html=True)

status_placeholder = st.empty()

run_camera = st.checkbox('Run Camera')

if not run_camera:
    st.image('cartoon.png', caption='', use_column_width=True)

else:
    stframe = st.empty()
    while run_camera:
        frame, status = video_camera.get_frame()
        stframe.image(frame, channels='RGB', use_column_width=True)

# if run_camera:
#     stframe = st.empty()
#     while run_camera:
#         frame, status = video_camera.get_frame()
#         stframe.image(frame, channels='RGB', use_column_width=True)

        status_placeholder.markdown(f"<h4>Status: {status}</h4>", unsafe_allow_html=True)
        #status_placeholder.text(f"Status: {status}")
