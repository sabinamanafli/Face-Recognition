import face_recognition
import cv2
import os
from imutils import paths
import pickle



		
pathes=list(paths.list_images('dataset'))
knownNames=[]
knownEncodings=[]


for path in pathes:
 
    
    names=path.split(os.path.sep)[-2]

    image=cv2.imread(path)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    box=face_recognition.face_locations(rgb,model='hog')
    encodings=face_recognition.face_encodings(rgb,box)

    for encoding in encodings:
    	knownEncodings.append(encoding)

    for name in names:
    	knownNames.append(name)



data={'encodings': knownEncodings, 'names':knownNames}

file=open('encodings.pickle','wb')
file.write(pickle.dumps(data))
file.close()







#box=face_recognition.face_locations(rbg,method='method')
#encodings=face_recognition.face_encodings(rgb,boxes)





