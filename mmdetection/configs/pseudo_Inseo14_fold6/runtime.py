checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='Inseo_Lee',
                 name='[Inseo9]cascadeRCNN_swin_b_ms'
             ),
             ),
        dict(type='MlflowLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
# './work_dirs/cascadeRCNN_swin_l_fpp_rpn_cosineRe_ms/best_bbox_mAP_50_epoch_11.pth'
load_from = './work_dirs/cascadeRCNN_swin_l_fpp_rpn_cosineRe_ms_fold6/best_bbox_mAP_50_epoch_13.pth'
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
