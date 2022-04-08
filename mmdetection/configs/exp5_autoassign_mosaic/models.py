
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'  # noqa
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa

model = dict(
    type='AutoAssign',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')),
    bbox_head=dict(
        type='AutoAssignHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_bbox=dict(type='GIoULoss', loss_weight=5.0)),
    train_cfg=None,
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100),
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
