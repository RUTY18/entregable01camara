import cv2
import os
from camera import getcamera

dataPath ='./data'
imagePaths =os.listdir(dataPath)
print('imagen=', imagePaths)

#crear el modelo
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('modelo.xml')

#crear el clasificador de rostros
faceClassif = cv2.CascadeClassifier('rostros.xml')

#abrir la camara web
camera = getcamera()
cap =cv2.VideoCapture(camera, cv2.CAP_DSHOW)

while True:
    ret, frame =cap.read()
    
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #copia de la imagen en blanco y negro
    auxFrame = gray.copy()
    
    face = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in face:
        
        #extraer rosytro de la ima original        
        rostro = auxFrame[y:y + h, x:x + w]
        
        rostro = cv2.resize(rostro, (150,150))
        result = face_recognizer.predict(rostro)
        
        if result[1]<75:
            cv2.putText(frame, '{}'.format(imagePaths [result[0]]), (x, y-25), 2, 1, (0, 255, 0))
            cv2.rectangle(frame, (x,y), (x + w, y + h), (8, 255, 0), 2) # rectangulo verde
        else:
            cv2.putText(frame, 'Desconocido', (x, y-25), 2, 1, (0, 255, 0))
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 0, 255), 2) # rectangulo rojo
            
    cv2.imshow('imagen', frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):

        break
#cierra camara y ventanas

cv2.destroyAllWindows()
cap.release()


