# YOLOv9로 플라스틱 재활용 폐기물 분류하기

## 1. Setup

`pip install ultralytics`


## 2. 데이터셋 설정하기

YOLOv9 모델을 훈련시킬 데이터셋을 `ultralytics/cfg/`에 넣고 plastic.yaml에 경로를 지정합니다.


## 3. How to use

### Train

`hyperparameters.json` 파일을 통해 하이퍼 파라미터를 변경해 주고 `yolov9_train.py` 파일을 실행하여 훈련을 시작합니다.


### Test

`plastic_test.py` 에서 `model = YOLO("weight 경로 설정")` 을 통해 원하는 weight의 경로를 지정한 뒤 파일을 실행해 테스트를 진행합니다.

### Visualization

`visualization.py` 에서 훈련된 모델의 가중치와, 시각화를 해줄 이미지를 지정합니다.

```python
model = YOLO("훈련된 weight 지정")

image_paths = [
"이미지 경로1",
"이미지 경로2",
"이미지 경로3",
"이미지 경로4",
"이미지 경로5"
]
```

