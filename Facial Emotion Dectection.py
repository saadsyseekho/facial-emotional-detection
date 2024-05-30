import cv2

pip install deepface

from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('E:\Downloads\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change to 1, 2, etc. if you have multiple cameras

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret,frame = cap.read()
    result = DeepFace.analyze(img_path = frame , actions=['emotion'], enforce_detection=False )
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotion = result[0]["dominant_emotion"]
    txt = str(emotion)
    
    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('frame',frame)


    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object when done
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()