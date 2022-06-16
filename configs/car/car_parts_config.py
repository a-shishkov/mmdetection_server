# the new config inherits the base configs to highlight the necessary modification
# _base_ = 'mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('headlamp', 'rear_bumper', 'door', 'hood', 'front_bumper')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='datasets/car/train/COCO_mul_train_annos.json',
        img_prefix='datasets/car/img/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='datasets/car/val/COCO_mul_val_annos.json',
        img_prefix='datasets/car/img/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='datasets/car/val/COCO_mul_val_annos.json',
        img_prefix='datasets/car/img/'),
    )

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)))

load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_16324.pth'