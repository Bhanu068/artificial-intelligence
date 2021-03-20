import cv2
import numpy as np
import speech_recognition as sr
from tensorflow import keras
from keras.models import model_from_json
from imutils.video import FileVideoStream
from PIL import Image, ImageFont, ImageDraw


save_model_path = 'C:\Personal Coding\Python Personal Developments\DL & ML Models\emotion_from_video.json'
save_model_weights = 'C:\Personal Coding\Python Personal Developments\DL & ML Models\emotion_from_video_weights.h5'
haar_cascade_path = 'C:\Personal Coding\Python Personal Developments\Open CV Haar Cascade files\haarcascade_frontalface_default.xml'




json_file = open(save_model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(save_model_weights)

cv2.ocl.setUseOpenCL(False)



EMOTIONS = ["angry" ,"disgust","scared", "happy", "neutral", "surprised",
 "sad"]


face_classifier = cv2.CascadeClassifier(haar_cascade_path)

r = sr.Recognizer()

cap = cv2.VideoCapture('C:/Users/91951/Pictures/Camera Roll/WIN_20210307_14_41_01_Pro.mp4')
        


while cap.isOpened():
    ret, frame = cap.read()
    
    
    results = face_classifier.detectMultiScale(frame, 1.3, 5)
    
    for (x, y, w, h) in results:
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_color = frame[y : y + h, x : x + w]
        
        cropped_img = cv2.resize(roi_color, (96, 96)) 
        cropped_img_expanded = np.expand_dims(cropped_img, axis = 0)
        cropped_img_float = cropped_img_expanded.astype(float)
        
        prediction = loaded_model.predict(cropped_img_float)
        
        maxindex = int(np.argmax(prediction))
        
        cv2.putText(frame, EMOTIONS[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Video', frame)
    try:
        cv2.imshow("frame", cropped_img)
    except:
        cv2.imshow("frame", black)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

with sr.AudioFile("converted.wav") as audio_file:
    
        audio_content = r.record(audio_file)
        detected_text = r.recognize_google(audio_content)
        print('\n\n')
        print('VOICE TO TEXT')
        print('\n')
        print(detected_text)
