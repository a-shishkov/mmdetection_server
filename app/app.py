import base64
from itertools import groupby
from flask import Flask, request
from flask_compress import Compress
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
    return "Hello world"


@app.route("/detect", methods=["POST"])
def detect():
    encoded_image = request.json["image"]
    image_bytes = base64.b64decode(encoded_image)

    image_size = (request.json["width"], request.json["height"])
    image_h, image_w = image_size

    image = np.array(image_bytes)
    image = image.reshape((1, image_h, image_w, 3)).astype(np.uint8)

    results = hub_model(image)
    result = {key: value.numpy() for key, value in results.items()}

    keys = (
        "classes",
        "scores",
        "boxes",
        "masks",
    )
    output = {}

    output[keys[0]] = result[keys[0]][0].astype(np.int32)
    output[keys[1]] = result[keys[1]][0]

    output["width"] = image_w
    output["height"] = image_h

    boxes = result[keys[2]][0]
    new_boxes = []
    for box in boxes:
        y1, x1, y2, x2 = box
        new_boxes.append(
            [(y1 * image_h), (x1 * image_w), (y2 * image_h), (x2 * image_w)]
        )
    output[keys[2]] = new_boxes

    if keys[3] in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result[keys[3]][0])
        detection_boxes = tf.convert_to_tensor(result[keys[2]][0])

        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2]
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)

        # Binary masks to RLE
        masks = detection_masks_reframed.numpy()
        new_masks = []
        for mask in masks:
            byte_mask = np.packbits(mask.flatten()).tobytes()
            encoded_mask = base64.b64encode(byte_mask).decode("utf-8")
            # encoded_mask = run_length_encode(encoded_mask)
            new_masks.append(encoded_mask)

        output[keys[3]] = new_masks

    return json.dumps(output, cls=NumpyEncoder)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
