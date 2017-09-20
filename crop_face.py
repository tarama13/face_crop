# -*- coding: utf-8 -*-
import cv2
import sys
import os

cascade_dir = './haarcascades/'

# 顔を検出して切り抜く
def crop_face(file):
    image = cv2.imread(file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale

    cascade_f = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_alt2.xml'))
    cascade_e = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))

    # 顔検出
    facerect = cascade_f.detectMultiScale(image_gray, scaleFactor=1.11, minNeighbors=2, minSize=(100, 100))

    output_dir = "face_images"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    base = os.path.splitext(os.path.basename(file))[0] + "_"

    #　検出した顔
    if len(facerect) > 0:
        filenum = 0
        for rect in facerect:
            x, y, w, h = rect
            eye_area = image_gray[y:y+h, x:x+w]
            eyes = cascade_e.detectMultiScale(eye_area, scaleFactor=1.11, minNeighbors=2)
            eyes = filter(lambda e: (e[0] > w / 2 or e[0] + e[2] < w / 2) and e[1] + e[3] < h / 2, eyes) # 目の位置がおかしいのを除外

            if len(eyes) > 0:
                image_face = image[y:y+h, x:x+h]
                file_name = output_dir + "/" + base + str("{0:02d}".format(filenum)) + ".jpg"
                cv2.imwrite(file_name, image_face)
                print("output: " + file_name)
                filenum += 1


if __name__ == '__main__':
    argvs = sys.argv
    if (len(argvs) != 2):
        print ("Usage: $ python " + argvs[0] + " image.jpg")
        quit()

    print(cascade_dir)
    crop_face(argvs[1])
