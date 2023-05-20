import cv2
import mediapipe as mp
import time

camera = cv2.VideoCapture(0)
ptime = 0

#pendeklarasian
mpDrawing = mp.solutions.drawing_utils
mpfacemesh = mp.solutions.face_mesh

#hanya mendeteksi 2 wajah
facemesh = mpfacemesh.FaceMesh(max_num_faces= 2)

#spesifikasi sensor
drawspec = mpDrawing.DrawingSpec(thickness= 1, circle_radius= 2)


while True:
    start, frame = camera.read()
    gambarRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = facemesh.process(gambarRGB)

    if hasil.multi_face_landmarks:
        for facelandmark in hasil.multi_face_landmarks:
            mpDrawing.draw_landmarks(frame, facelandmark, mpfacemesh.FACEMESH_CONTOURS, drawspec, drawspec)

            for lm in facelandmark.landmark:
                #print(lm)

                h, w, c = frame.shape
                x, y = int(lm.x*w), int(lm.y*h)
                print(x, y)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame, f'{int(fps)}', (5, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow('Face mesh', frame)
    if cv2.waitKey(1) == ord('q'):
        break
