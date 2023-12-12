# # Importing OpenCV Library for basic image processing functions
# import cv2
# # Numpy for array related functions
# import numpy as np
# # Dlib for deep learning based Modules and face landmark detection
# import dlib
# # face_utils for basic operations of conversion
# from imutils import face_utils

# # Initializing the camera and taking the instance
# cap = cv2.VideoCapture(0)

# # Initializing the face detector and landmark detector
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # status marking for the current state
# sleep = 0
# drowsy = 0
# active = 0
# status = ""
# color = (0, 0, 0)

# # Initialize face_frame outside the loop
# face_frame = None

# def compute(ptA, ptB):
#     dist = np.linalg.norm(ptA - ptB)
#     return dist

# def blinked(a, b, c, d, e, f):
#     up = compute(b, d) + compute(c, e)
#     down = compute(a, f)
#     ratio = up / (2.0 * down)

#     # Checking if it is blinked
#     if ratio > 0.25:
#         return 2
#     elif 0.21 <= ratio <= 0.25:
#         return 1
#     else:
#         return 0

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = detector(gray)
#     # detected face in faces array
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()

#         face_frame = frame.copy()
#         cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         landmarks = predictor(gray, face)
#         landmarks = face_utils.shape_to_np(landmarks)

#         # The numbers are actually the landmarks that will show the eye
#         left_blink = blinked(landmarks[36], landmarks[37],
#                              landmarks[38], landmarks[41], landmarks[40], landmarks[39])
#         right_blink = blinked(landmarks[42], landmarks[43],
#                               landmarks[44], landmarks[47], landmarks[46], landmarks[45])

#         # Now judge what to do for the eye blinks
#         if left_blink == 0 or right_blink == 0:
#             sleep += 1
#             drowsy = 0
#             active = 0
#             if sleep > 6:
#                 status = "SLEEPING !!!"
#                 color = (255, 0, 0)

#         elif left_blink == 1 or right_blink == 1:
#             sleep = 0
#             active = 0
#             drowsy += 1
#             if drowsy > 6:
#                 status = "Drowsy !"
#                 color = (0, 0, 255)

#         else:
#             drowsy = 0
#             sleep = 0
#             active += 1
#             if active > 6:
#                 status = "Active :)"
#                 color = (0, 255, 0)

#         cv2.putText(frame, status, (100, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

#         for n in range(0, 68):
#             (x, y) = landmarks[n]
#             cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

#     cv2.imshow("Frame", frame)
#     if face_frame is not None:
#         cv2.imshow("Result of detector", face_frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()


from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils

app = Flask(__name__)

def compute(ptA, ptB):
        dist = np.linalg.norm(ptA - ptB)
        return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
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
        self.sleep = 0  # Declare sleep as an instance variable
        self.drowsy = 0
        self.active = 0
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

        _, jpeg = cv2.imencode('.jpg', face_frame)
        return jpeg.tobytes()

    def update_sleep(self, value):
        self.sleep = value

video_camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = video_camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
