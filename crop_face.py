# -*- coding: utf-8 -*-
import configparser
import cv2
import sys
import os

config = configparser.ConfigParser()
config.read('./config.ini')
cascade_dir = config.get('default', 'cascade_dir')


# 顔を検出して切り抜く
def crop_face(file):
    image, faces = detect_face(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale

    cascade_e = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))

    output_dir = "face_images"
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(file))[0] + "_"

    #　検出した顔
    if len(faces) > 0:
        filenum = 0
        for rect in faces:
            x, y, w, h = rect
            eye_area = image_gray[y:y+h, x:x+w]

            eye = config['eye']
            eyes = cascade_e.detectMultiScale(
                    eye_area,
                    scaleFactor = float(eye['scaleFactor']),
                    minNeighbors = int(eye['minNeighbors']),
                )
            # eyes = filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes) # 目の位置がおかしいのを除外

            if len(eyes) > 0:
                image_face = image[y:y+h, x:x+h]
                file_name = output_dir + "/" + base + str("{0:02d}".format(filenum)) + ".jpg"
                cv2.imwrite(file_name, image_face)
                print("output: " + file_name)
                filenum += 1


def detect_face(file):
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_f = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_alt2.xml'))

    # 顔検出
    face = config['face']
    minSize = int(face['minSizeWidth'])
    faces = cascade_f.detectMultiScale(
            image_gray,
            scaleFactor = float(face['scaleFactor']),
            minNeighbors = int(face['minNeighbors']),
            minSize = (minSize, minSize),
        )

    return [image, faces]


def mark_face(file):
    image, faces = detect_face(file)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image


def show_image(file):
    image = mark_face(file)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    argvs = sys.argv
    if (len(argvs) != 2):
        print ("Usage: $ python " + argvs[0] + " image.jpg")
        quit()

    crop_face(argvs[1])
    show_image(argvs[1])
