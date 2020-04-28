import io
import cv2
from flask import Flask, request, send_file
import numpy as np

app = Flask(__name__)

@app.route('/cropimage', methods=['POST'])
def crop_image():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    x = img_resized.shape[1]
    y = img_resized.shape[0]

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    y_list = []

    for cy in range(y):
        counter = 0
        for cx in range(x):
            if (thresh[cy, cx] == 255):
                counter += 1

        if (counter == x):
            y_list.append(cy)

    crop = img_resized[min(y_list):max(y_list), 0:x]

    cv2.imwrite('output/cropped.jpg', crop)

    with open("output/cropped.jpg",'rb') as bites:
        return send_file(
            io.BytesIO(bites.read()),
            mimetype='image/jpeg',
            as_attachment=True,
            attachment_filename='cropped.jpg')

if __name__ == '__main__':
    app.run(port=5000)
