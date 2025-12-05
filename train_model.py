from PIL import Image
import cv2
import numpy as np
import os

path = "dataset"


def getImageName(path):
    imagePath =[os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    labels = []



    for imagePaths in imagePath:
        facePic = Image.open(imagePaths).convert("L")  
        faceNp = np.array(facePic, 'uint8')

        faces.append(faceNp)
        labels.append(labels)


        cv2.imshow("testing capturing", faceNp)
        cv2.waitKey(5)
    return faces, labels


faces, labels = getImageName(path)
recognizer.train(faces, np.array(labels))
recognizer.write("model.yml")
cv2.destroyAllWindows()


print("Faces:", len(faces), "Labels:", labels)



