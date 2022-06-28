import base64
from itertools import groupby
from flask import Flask, request
from flask_compress import Compress
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from object_detection.utils import ops as utils_ops
import os
import imageio

app = Flask(__name__)
Compress(app)

# model_handle = "https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"
model_handle = "saved_model"
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

def save_anno(image):
    if "annotations" not in request.json:
        return

    annotations = request.json["annotations"]

    client_dir = os.path.join("results", request.remote_addr)
    images_dir = os.path.join(client_dir, "images")

    if not os.path.exists(client_dir):
        os.makedirs(client_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    image_id = len(os.listdir(images_dir))
    imageio.imwrite(os.path.join(images_dir, f"{image_id}.jpg"), image)

    anno_path = os.path.join(client_dir, "annotations.json")
    if os.path.exists(anno_path):
        with open(anno_path, "r") as anno_file:
            anno_data = json.load(anno_file)
    else:
        anno_data = {"annotations": [], "images": []}

    last_anno_id = len(anno_data["annotations"])
    for anno in annotations:
        anno["image_id"] = image_id
        anno["id"] = last_anno_id
        last_anno_id += 1
        anno_data["annotations"].append(anno)
    anno_data["images"].append({"file_name": f"images/{image_id}.jpg", "id": image_id})
    with open(anno_path, "w") as anno_file:
        json.dump(anno_data, anno_file, indent=4, sort_keys=True)

@app.route("/")
def hello_world():
    return "Hello world"


@app.route("/detect", methods=["POST"])
def detect():
    # Decode image from base64 and get raw pixels sequence
    encoded_image = request.json["image"]
    image_bytes = base64.b64decode(encoded_image)

    image_size = (request.json["width"], request.json["height"])
    image_h, image_w = image_size

    # Convert to usual image format
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((1, image_h, image_w, 3))

    save_anno(image)

    # Inference
    results = hub_model(image)
    result = {key: value.numpy() for key, value in results.items()}

    keys = (
        "classes",
        "scores",
        "boxes",
        "masks",
    )
    output = {}

    # Get list of classes
    output[keys[0]] = result["detection_" + keys[0]][0].astype(np.int32)
    # List of scores
    output[keys[1]] = result["detection_" + keys[1]][0]

    output["width"] = image_w
    output["height"] = image_h

    # Renormalize boxes
    boxes = result["detection_" + keys[2]][0]
    renorm_boxes = []
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        renorm_boxes.append(
            [(ymin * image_h), (xmin * image_w), (ymax * image_h), (xmax * image_w)]
        )
    output[keys[2]] = renorm_boxes

    # If inference model is Mask RCNN then build binary mask
    if "detection_" + keys[3] in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result["detection_" + keys[3]][0])
        detection_boxes = tf.convert_to_tensor(result["detection_" + keys[2]][0])

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
