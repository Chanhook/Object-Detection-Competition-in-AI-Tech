# dataset settings

dataset_type = "CocoDataset"
data_root = "dataset2/"

classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
min_values = (704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024)

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip', p=1.0),
            dict(type='RandomRotate90', p=1.0)
        ],
        p=0.1
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='RandomResizedCrop', height=1024,
         width=1024, scale=(0.5, 1.0), p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ChannelShuffle',
                p=1.0),
            dict(
                type='RandomGamma',
                p=1.0),
            dict(
                type='RGBShift',
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]
custom_import = dict(
    imports=['mmdet.datasets.pipelines.transforms'], allow_failed_imports=False)
train_pipeline = [
    dict(type='Mosaic_p', img_scale=(1024, 1024), pad_val=0, p=0.3),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(value, 1024) for value in min_values],
                multiscale_mode='value',
                keep_ratio=True)
        ],
            [
            dict(
                type='Resize',
                img_scale=[(512, 1024), (640, 1024), (768, 1024)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(492, 768),
                allow_negative_crop=True),
            dict(
                type='Resize',
                img_scale=[(value, 1024) for value in min_values],
                multiscale_mode='value',
                override=True,
                keep_ratio=True)
        ]]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + "stratified_kfold/cv_train_1.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "stratified_kfold/cv_val_1.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
    ),
)
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
