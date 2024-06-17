from ultralytics import YOLO

model = YOLO("./YOLOv9_best.pt")

validatioin_results = model.val(
	data='./plastic.yaml',
	save=False,
	imgsz=640,
	conf=0.25,
	device="mps",
)