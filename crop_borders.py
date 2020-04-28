import cv2

img = cv2.imread('img/deneme3.jpg')
x = img.shape[1]
y = img.shape[0]


img_resized = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

y_list = []

for cy in range(img_resized.shape[0]):
    counter = 0
    for cx in range(img_resized.shape[1]):
        if (thresh[cy,cx] == 255):
            counter+=1

    if(counter == img_resized.shape[1]):
            y_list.append(cy)

crop = img_resized[min(y_list):max(y_list), 0:x]

cv2.imwrite('cropped_new2.jpg',crop)
