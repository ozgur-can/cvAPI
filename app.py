import base64
import io
from wsgiref.headers import Headers

import cv2
from PIL.Image import Image
from flask import Flask, request, send_file, Response, make_response, jsonify
from flask_cors import CORS
import numpy as np
import requests
from werkzeug.datastructures import FileStorage, ImmutableMultiDict
from requests_toolbelt import MultipartEncoder

app = Flask(__name__)
CORS(app)

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

@app.route('/test', methods=['POST'])
def test():

    photo = request.files["photo"].read()

    # convert string of image data to uint8
    nparr = np.fromstring(photo, np.uint8)

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

    is_success, encoded = cv2.imencode('.jpg', crop)
    byte_img = io.BytesIO(encoded)
    img_to_search =  byte_img.getvalue()

    if(is_success):
        r = requests.post("http://localhost:8983/solr/lire/lireq?field=cl_ha&ms=false&accuracy=0&candidates=100000",data=img_to_search)
        return Response(response=r)
    else:
        return Response(response=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
