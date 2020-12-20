#main.py
from PIL import Image, ImageDraw
from flask import Flask, request
import cloud_activities
import numpy as np
import base64
import json
import math
import cv2
import io
from io import BytesIO

app = Flask(__name__)
# https://cloud.google.com/vision/automl/docs/tutorial


@app.route("/recognize", methods=["POST"])
def recognize():
    params = request.get_json()
    image_base64_string = params['image']
    image_bytes = base64.b64decode(image_base64_string)
    image_pillow = Image.open(io.BytesIO(image_bytes))
    data = np.asarray(image_pillow)
    results = cloud_activities.flask_face_recognize(data)
    return results


@app.route("/detect_mask", methods=["POST"])
def detect_mask():
    params = request.get_json()
    image_base64_string = params['image']
    image_bytes = base64.b64decode(image_base64_string)
    faces = cloud_activities.flask_detect_face(image_bytes)
    cropped_image_bytes, box = cloud_activities.flask_crop_face(image_bytes, faces)
    mask_detect_prediction = cloud_activities.flask_detect_mask(cropped_image_bytes,
                                                                "cse212",
                                                                "ICN3693103427048439808"
                                                                )
    for result in mask_detect_prediction.payload:
        if result.display_name == 'Mask':
            display_name = 'Y'
        else:
            display_name = 'N'
        score = result.classification.score
        break

    response = {
                    "mask": display_name,
                    "score": score,
                    "xmin": str(box[0]), "xmax": str(box[1]),
                    "ymin": str(box[2]), "ymax": str(box[3])
                }
    response = json.dumps(response)
    # print(response)
    return response


@app.route("/detect_mask_1", methods=["POST"])
def detect_mask_without_face_detection():
    params = request.get_json()
    image_base64_string = params['image']
    image_bytes = base64.b64decode(image_base64_string)
    mask_detect_prediction = cloud_activities.flask_detect_mask(image_bytes, "cse212", "ICN3693103427048439808")
    for result in mask_detect_prediction:
        display_name = result.display_name
        score = result.classification.score
        break

    response = {
                    "mask": display_name,
                    "score": score,
                }

    return json.dumps(response)


def calculate_distance(midpoints):
    distances = []
    boxes = {}
    visited = set([])
    for obj_a in midpoints:
        boxes[obj_a] = midpoints[obj_a][2]
        for obj_b in midpoints:
            if obj_a == obj_b : continue
            if (obj_a, obj_b) in visited : continue
            visited.add((obj_a, obj_b))
            visited.add((obj_b, obj_a))
            dist = math.sqrt((midpoints[obj_a][0]-midpoints[obj_b][0])**2+(midpoints[obj_a][1]-midpoints[obj_b][1])**2)
            distances.append([
                              [midpoints[obj_a][0], midpoints[obj_a][1]],
                              [midpoints[obj_b][0], midpoints[obj_b][1]],
                              dist/113.0
                            ])
    return distances, boxes


@app.route("/detect_mask_2", methods=["POST"])
def detect_mask_without_face_detection_2():
    params = request.get_json()
    image_base64_string = params['image']
    faces = params["faces"]
    image_bytes = base64.b64decode(image_base64_string)
    mask_detect_prediction, midpoints = cloud_activities.flask_detect_mask_2(image_bytes, faces,  "cse212", "ICN3693103427048439808")
    response = {}

    distances, boxes = calculate_distance(midpoints)

    k = 0
    for result in mask_detect_prediction:

        obj_id,  pred_obj, box, midpoints = result

        score = pred_obj.payload[0].classification.score

        if pred_obj.payload[0].display_name == 'Mask':
            display_name = 'Y'
        else:
            display_name = 'N'

        response[k] = {
                        "mask": display_name,
                        "score": score,
                        "box": box,
                        "mid_point": midpoints,
                    }
        k += 1
    response[k] = {"distances": distances}

    # print("#############")
    # print(response)
    # print("#############")

    return json.dumps(response)


@app.route("/detect_thermal", methods=["POST"])
def detect_thermal():
    params = request.get_json()
    image_base64_string = params['image']
    image_bytes = base64.b64decode(image_base64_string)
    faces = cloud_activities.flask_detect_face(image_bytes)
    image_pillow = Image.open(io.BytesIO(image_bytes))
    for face in faces:
        face_box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        image_pillow = image_pillow.crop([face_box[0][0]+5, face_box[0][1]+10, face_box[1][0]-5, face_box[2][1]])
        data = np.asarray(image_pillow)
        break
    image = cloud_activities.flask_detect_thermal(data)
    mean = np.mean(image)
    image_pillow = Image.fromarray(image)
    buffered = BytesIO()
    image_pillow.save(buffered, format="JPEG")
    # image_pillow.show()
    base64_encoded = base64.b64encode(buffered.getvalue())
    img_str = str(base64_encoded)
    img_str = img_str[2:len(img_str)-1]

    return json.dumps(
                        {
                            "image": str("data:image/png;base64,"+img_str),
                            "mean": mean
                        }
                    )


if __name__ == "__main__":
    app.run()
