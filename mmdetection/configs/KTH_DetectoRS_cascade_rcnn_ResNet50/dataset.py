# dataset settings

dataset_type = 'CocoDataset'
data_root = 'dataset2/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

albu_train_transforms = [
    # Flipping and Rotating an image
    dict(
        type='HorizontalFlip',
        p = 0.5
    ),
    dict(
        type='VerticalFlip',
        p=0.5
    ),
    dict(
        type='RandomRotate90',
        p=0.5    
    ),
    # Adjusting a dynamic range of the image
    dict(
        type='OneOf',
        transforms = [
            dict(
                type = 'CLAHE',
                p = 0.2,
                clip_limit = (1,4),
                tile_grid_size=(8,8),
            ),
            dict(
                type = 'CLAHE',
                p = 0.2,
                clip_limit = (1,4),
                tile_grid_size=(16,16),
            ),
            dict(
                type = 'CLAHE',
                p = 0.2,
                clip_limit = (1,4),
                tile_grid_size=(64,64),
            ),
            dict(
                type = 'CLAHE',
                p = 0.2,
                clip_limit = (1,4),
                tile_grid_size=(128,128),
            ),
            dict(
                type = 'RandomBrightnessContrast',
                p=0.2,
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3]
            )
        ],
        p=0.33
    ),
    # Cropping and Resizing
    dict(
        type = 'RandomSizedBBoxSafeCrop',
        height = 512,
        width = 512,
        erosion_rate = 0.0,
        interpolation = 3,#3 = 'cv2.INTER_AREA', https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/enum_cv_InterpolationFlags.html
        p = 0.5
    ),
    # Adding noise or blur effect on the image
    dict(
        type = 'OneOf',
        transforms = [
            dict(
                type='ColorJitter',
                brightness = 0.2,
                contrast = 0.2,
                saturation = 0.2,
                hue = 0.2,
                p = 0.2
            ),
            dict(
                type = 'GaussNoise',
                var_limit=[5.0,15.0],
                mean = 0,
                per_channel = True,
                p = 0.2
            ),
            dict(
                type = 'MotionBlur',
                blur_limit = [3,9],
                p = 0.2
            ),
            dict(
                type = 'ChannelShuffle',
                p = 0.2
            ),
            dict(
                type = 'Blur',
                blur_limit = [3,9],
                p = 0.2
            )
        ],
        p = 0.5
    )
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'stratified_kfold/cv_train_1.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'stratified_kfold/cv_val_1.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
