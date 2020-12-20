from google.cloud import automl_v1
from google.cloud.vision_v1 import types
from google.cloud import vision
from PIL import Image, ImageDraw
import base64
import cv2
import io
import os


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sunhacks-292004-00c1fcf54694.json"


def flask_detect_face(image_content, max_results=1):
    client = vision.ImageAnnotatorClient()
    image = types.Image(content=image_content)
    return client.face_detection(image=image, max_results=max_results).face_annotations


def flask_detect_mask_2(image_bytes, faces, project_id, model_id):
    image_pillow = Image.open(io.BytesIO(image_bytes))
    response = []

    midpoints = {}
    for i, face in enumerate(faces):
        midpoint_x, midpoint_y = (face[0]+(face[0]+face[2]))/2.0, (face[1]+(face[1]+face[2]))/2.0
        midpoints[i] = [midpoint_x, midpoint_y, face]
        img_pil = image_pillow.crop([face[0], face[1], face[0]+face[2], face[1]+face[3]])
        image_bytes = io.BytesIO()
        img_pil.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
        prediction_client = automl_v1.PredictionServiceClient()
        payload = {'image': {'image_bytes': image_bytes}}
        prediction_object = prediction_client.predict(name=name, payload=payload)
        response.append([i, prediction_object, face, [midpoint_x, midpoint_y]])
    return response, midpoints


def flask_detect_mask(image_content, project_id, model_id):
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    prediction_client = automl_v1.PredictionServiceClient()
    payload = {'image': {'image_bytes': image_content}}
    response = prediction_client.predict(name=name, payload=payload)
    return response


def flask_crop_face(image_bytes, faces):
    image_pillow = Image.open(io.BytesIO(image_bytes))
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        image_pillow.crop([box[0][0], box[1][0], box[0][1], box[2][1]])
        image_bytes = io.BytesIO()
        image_pillow.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()
        return image_bytes, [box[0][0], box[1][0], box[0][1], box[2][1]]
    return None


def flask_detect_thermal(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def detect_face(face_file, max_results=1):
    client = vision.ImageAnnotatorClient()
    content = face_file.read()
    image = types.Image(content=content)
    return client.face_detection(image=image, max_results=max_results).face_annotations


def highlight_faces(image, faces, output_filename):
    im = Image.open(image)
    draw = ImageDraw.Draw(im)
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        draw.text((face.bounding_poly.vertices[0].x,
                   face.bounding_poly.vertices[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save(output_filename)


def detect_mask(content, project_id, model_id):
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    prediction_client = automl_v1.PredictionServiceClient()
    payload = {'image': {'image_bytes': content}}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request


def recognize_face():
    pass


def main(input_filename, output_filename, max_results):
    with open(input_filename, 'rb') as image:
        faces = detect_face(image, max_results)
        print('Found {} face{}'.format(len(faces), '' if len(faces) == 1 else 's'))
        print('Writing to file {}'.format(output_filename))
        # Reset the file pointer, so we can read the file again
        image.seek(0)
        highlight_faces(image, faces, output_filename)

    with io.open(input_filename, 'rb') as image_file:
        image_content = image_file.read()
        results = detect_mask(image_content, "cse212", "ICN3693103427048439808")


if __name__ == "__main__":
    main("fm.jpeg", "fm_op.jpg", 100)
