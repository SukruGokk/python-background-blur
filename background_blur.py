# @date: 23.03.2021
# @author: Şükrü Erdem Gök
# @version: Python 3.8
# @os: Windows 10
# @github: https://github.com/SukruGokk

# Background Blur

# Libs
from cv2 import imread, GaussianBlur, imshow, BORDER_DEFAULT, waitKey, CascadeClassifier, cvtColor, COLOR_BGR2GRAY, VideoCapture
from sys import argv

# Load the cascade
face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')

# Get image
if argv[1] == 'cam':
	src = VideoCapture(0)
	_, img = src.read()
else: 
	img = imread(argv[1])

# Convert to grayscale
gray = cvtColor(img, COLOR_BGR2GRAY)

# Detect the faces
face_coors = face_cascade.detectMultiScale(gray, 1.1, 4)

faces = []

# Blur
blurred_img = GaussianBlur(img,(7,7), BORDER_DEFAULT)

for face in face_coors:

    left = face[0]
    top = face[1]
    right = face[2] + left
    bottom = face[3] + top

    blurred_img[top:bottom, left:right] = img[top:bottom, left:right]

# Show
imshow('Blurred.jpg', blurred_img)

waitKey()