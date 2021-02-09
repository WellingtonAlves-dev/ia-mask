#adelia IA
from dataset_adelia import *
cap = cv2.VideoCapture(0)

cascade = "cascade/faces.xml"

cascades = cv2.CascadeClassifier(cascade)

while True:
    ret, frame = cap.read(cv2.IMREAD_COLOR)
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascades.detectMultiScale(frame_grey)
    for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0))
            frame_crop = frame[y:y+h, x:x+w]
            frame_crop = cv2.resize(frame_crop, (160, 160))
            frame_crop = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            frame_array = numpy.array(frame_crop).flatten()
            predicado = clf.predict([frame_array])
            if(predicado[0] == 0):
                cv2.putText(frame, "Sem mascara D:", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
            elif(predicado[0] ==1):
                cv2.putText(frame, "Com mascara :D", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
    try:
        cv2.imshow("Test", frame)
    except:
        pass
    if(cv2.waitKey(1) == ord("ç")):
        break

cap.release()
cv2.destroyAllWindows()
