_base_ = [
    'dataset.py',
    'schedule.py',
    'runtime.py',
    'models.py'
]

# 총 epochs 사이즈
runner = dict(max_epochs=12)

# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4)

checkpoint_config = dict(interval=-1)

# 로그와 관련된 셋팅
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    # 각각 자신에 맞춰서 Project이름 설정
                    project= 'Taeha_Kim',
                    name = 'DetectoRS_cascade_rcnn_resnet50'
                ),
            ),
        dict(type='MlflowLoggerHook')
    ])