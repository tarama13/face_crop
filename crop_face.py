# -*- coding: utf-8 -*-
import configparser
import cv2
import sys
import os
import math
import numpy

parent_dir = os.path.abspath(os.path.dirname(__file__))
config = configparser.ConfigParser()
config.read(os.path.join(parent_dir, 'config.ini'))

xml_dir = os.path.join(parent_dir, 'haarcascades/')
cascade_face = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_frontalface_alt2.xml'))
cascade_eye = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_eye.xml'))
cascade_mouth = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_mcs_mouth.xml'))
cascade_nose = cv2.CascadeClassifier(os.path.join(xml_dir, 'haarcascade_nose.xml'))

# ファイルから顔を（検出して）切り抜いて保存
def crop_face(file):
    image = cv2.imread(file)
    image, faces = detect_face(image)

    output_dir = "face_images"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file))[0] + "_"

    #　顔を保存
    if len(faces) > 0:
        filenum = 0
        for rect in faces:
            x, y, w, h = rect
            image_face = image[y:y+h, x:x+h]
            file_name = output_dir + "/" + base + str("{0:02d}".format(filenum)) + ".jpg"
            cv2.imwrite(file_name, image_face)
            print("output: " + file_name)
            filenum += 1
    return

# 画像から顔をすべて検出する
def detect_face(image):
    # image = cv2.imread(file)

    # 顔検出
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = config['face']
    minSize = int(face['minSizeWidth']) #　顔の大きさ
    faces = cascade_face.detectMultiScale(
                image_gray,
                scaleFactor = float(face['scaleFactor']),
                minNeighbors = int(face['minNeighbors']),
                minSize = (minSize, minSize),
            )

    if len(faces) > 0:
        for rect in faces:
            x, y, w, h = rect
            eye_area = image_gray[y:y+h, x:x+w]
            eye = config['eye']
            eyes = cascade_eye.detectMultiScale(
                eye_area,
                scaleFactor = float(eye['scaleFactor']),
                minNeighbors = int(eye['minNeighbors']),
            )
            if len(eyes) > 0:
                image_face = image[y:y+h, x:x+h] # 正方形

    return [image, faces]


# 画像を大きさ(斜辺,斜辺)の枠に入れる
def image_extend(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    hypot = int(math.ceil(math.hypot(h, w))) # 斜辺の長さ
    y, x = int((hypot-h)*0.5), int((hypot-w)*0.5)
    frame = numpy.zeros( (hypot, hypot), numpy.uint8) # 枠
    frame[y:y+h, x:x+w] = image_gray # 枠の中心に画像を配置
    return frame

# 画像を中心を軸に時計回りに回転
def image_rotate(frame, deg):
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D( (h*0.5, w*0.5), -deg, 1.0)
    rotated = cv2.warpAffine(frame, M, (h, w))
    return rotated


def mark_face(image):
    image, faces = detect_face(image)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def show_image(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    argvs = sys.argv
    if (len(argvs) != 2):
        print ("Usage: $ python " + argvs[0] + " image.jpg")
        quit()

    image = cv2.imread(argvs[1])
    # frame = image_extend(image)
    # rotated = image_rotate(frame, 30)
    # show_image(rotated)
    # crop_face(argvs[1])
    image = mark_face(image)
    show_image(image)


