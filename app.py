import io
import cv2
from flask import Flask, request, send_file, Response, jsonify, make_response
from flask_cors import CORS
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

recursive_result = []


def recursive_search(img_to_search, accuracy):
    r = requests.post(
        f"http://localhost:8983/solr/lire/lireq?field=cl_ha&ms=false&accuracy={accuracy}&candidates=100000",
        data=img_to_search)

    data = r.json()['response']['docs']
    diff = data[0]['d']

    if diff > 10:
        for d in data:
            if d not in recursive_result:
                recursive_result.append(d)

        if accuracy < 6:
            accuracy += 1
            return recursive_search(img_to_search, accuracy=accuracy)

        else:
            recursive_result.sort(key=lambda x: x['d'])
            return recursive_result[:10]

    else:
        return r.json()['response']['docs'][:10]


@app.route('/cropimg', methods=['POST'])
def test():
    photo = request.files["photo"]

    # convert string of image data to uint8
    nparr = np.fromstring(photo.read(), np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img_grayscale = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if cv2.countNonZero(img_grayscale) == 0:
        return Response(response="black")

    img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    x = img_resized.shape[1]
    y = img_resized.shape[0]

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    y_list = []

    for cy in range(y):
        counter = 0
        for cx in range(x):
            if thresh[cy, cx] == 255:
                counter += 1

        if counter == x:
            y_list.append(cy)

    crop = img_resized[min(y_list):max(y_list), 0:x]

    is_success, encoded = cv2.imencode('.jpg', crop)
    byte_img = io.BytesIO(encoded)
    img_to_search = byte_img.getvalue()

    if is_success:
        r = recursive_search(img_to_search, accuracy=0)

    else:
        r = None

    return make_response({"search_results": r})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
