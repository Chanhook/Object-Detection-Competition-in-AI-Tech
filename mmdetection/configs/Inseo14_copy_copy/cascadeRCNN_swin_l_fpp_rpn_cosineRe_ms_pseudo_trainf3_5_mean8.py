_base_ = [
    'dataset.py',
    'schedule.py',
    'runtime.py',
    'models.py'
]

# 총 epochs 사이즈
runner = dict(max_epochs=24)

# samples_per_gpu -> batch size라 생각하면 됨
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2)

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
                 project='Inseo_Lee',
                 name='[Inseo13]cascadeRCNN_swin_l_ms_pseudo_trainf3_5_mean8'
             ),
             ),
        dict(type='MlflowLoggerHook')
    ])
