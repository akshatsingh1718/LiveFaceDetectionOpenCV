import cv2
import urllib.request
import numpy as np

# Ip of your device which you are using as input for video
# Only if you want to use camera of other device
# URL = 'http://<YOUR-IPV4>/shot.jpg'
URL = "http://192.168.0.15:8080/shot.jpg"
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# webcam = cv2.VideoCapture(0)

while True:
    # Read from current frame
    # successful_cam_read is boolean if frame is readed or not
    # frame is image or current frame
    # successful_cam_read, frame = webcam.read()

    # reading from urllib via phone or any device IP
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)

    # Image must convert to gray
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)
    # Printing face coordinates
    print(f'Face Cords  : {face_coordinates}')

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 4)

    # Resizing window size
    imS = cv2.resize(frame, (960, 540))
    cv2.imshow("Live Image Tracking", imS)
    key = cv2.waitKey(1)

    # Press Q or q for terminating program
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
# webcam.release()
print("Exiting..")
