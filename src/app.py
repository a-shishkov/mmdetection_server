from itertools import groupby
import os
from flask import Flask, request
import base64
from mmdet.apis import (
    inference_detector,
    init_detector,
)
import numpy as np
import cv2
import json
from flask_compress import Compress

app = Flask(__name__)
Compress(app)


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
    cv2.imwrite(os.path.join(images_dir, f"{image_id}.jpg"), image)

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
    return "hello"


@app.route("/predict", methods=["POST"])
def predict():
    encoded_image = request.json["image"]
    image_bytes = base64.b64decode(encoded_image)

    nparr = np.fromstring(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    save_anno(image)

    # build the model from a config file and a checkpoint file
    model = init_detector(
        "configs/car/car_parts_config.py",
        "checkpoints/car/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_cars-parts.pth",
        device="cuda:0",
    )

    # test a single image
    result = inference_detector(model, image)

    # show the results
    # show_result_pyplot(model, image, result)

    h, w = image.shape[:2]
    boxes, masks = result

    class_masks = []
    for instances in masks:
        encoded_masks = []
        for instance in instances:
            encoded_masks.append(instance.flatten())
            # byte_mask = np.packbits(instance.flatten()).tobytes()
            # encoded_mask = base64.b64encode(byte_mask).decode("utf-8")
            # encoded_masks.append(encoded_mask)
        class_masks.append(encoded_masks)

    output = {
        "width": w,
        "height": h,
        "damage": {"boxes": boxes, "masks": class_masks},
    }

    result_json = json.dumps(output, cls=NumpyEncoder)

    return result_json
