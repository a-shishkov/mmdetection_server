from ast import Num
import base64
from itertools import groupby
from flask import Flask, request
from flask_compress import Compress
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from object_detection.utils import ops as utils_ops

app = Flask(__name__)
Compress(app)

model_handle = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"
hub_model = hub.load(model_handle)


def run_length_encode(data: str) -> str:
    """Returns run length encoded string for data"""
    # A memory efficient (lazy) and pythonic solution using generators
    return "".join(f"{x}{sum(1 for _ in y)}" for x, y in groupby(data))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route("/")
def hello_world():
    return "qweqwe"


@app.route("/predict")
def predict():
    img = Image.open("image2.jpg")

    (im_width, im_height) = img.size
    img = np.array(img.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    results = hub_model(img)
    result = {key: value.numpy() for key, value in results.items()}

    keys = ("detection_classes", "detection_scores")
    output = {key: result[key][0] for key in keys}

    boxes = result["detection_boxes"][0]
    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        new_boxes.append(
            [
                int(x1 * im_width),
                int(y1 * im_height),
                int(x2 * im_width),
                int(y2 * im_height),
            ]
        )
    output["detection_boxes"] = new_boxes

    if "detection_masks" in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result["detection_masks"][0])
        detection_boxes = tf.convert_to_tensor(result["detection_boxes"][0])

        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, img.shape[1], img.shape[2]
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)

        # Binary masks to RLE
        masks = detection_masks_reframed.numpy()
        new_masks = []
        for mask in masks:
            byte_mask = np.packbits(mask.flatten()).tobytes()
            encoded_mask = base64.b64encode(byte_mask).decode("utf-8")
            encoded_mask = run_length_encode(encoded_mask)
            new_masks.append(encoded_mask)

        output["detection_masks_reframed"] = new_masks

    return json.dumps(output, cls=NumpyEncoder)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
