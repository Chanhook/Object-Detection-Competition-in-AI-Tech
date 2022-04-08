# optimizer
optimizer = dict(type="AdamW", lr=0.000025, weight_decay=0.0001, paramwise_cfg=dict(
    custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'relative_position_bias_table': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)
    }
))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1099,
    warmup_ratio=0.001,
    periods=[5495, 5495, 6594, 8792, 8792],
    restart_weights=[1, 0.85, 0.75, 0.7, 0.6],
    by_epoch=False,
    min_lr=5e-6
)
runner = dict(type='EpochBasedRunner', max_epochs=36)
