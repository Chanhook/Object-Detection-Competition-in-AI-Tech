https://github.com/open-mmlab/mmdetection/blob/master/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py

여기 파일을 참조했음.

이걸 바탕으로 하려니 train_cfg와 test_cfg는 남겼는데.

이게 이번 버전에서는 deprecated되었다고(반드시 모델 안에 선언해야한다고)...뜨네...?

음.. 넣었더니 또 안돼, cfg or default_args는 type이라는 key를 포함해야한다고 뜨는데. 모델에 key가 없어서 그렇다니..