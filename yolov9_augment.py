from ultralytics import YOLO
import json

if __name__ == "__main__":
	model = YOLO('./yolov9c.pt')
	
	# JSON 파일 읽기
	with open('hyperparameters.json', 'r') as file:
		hyperparameters = json.load(file)
	experiment_name = 'plastic_train'
	results = model.train(data='./plastic.yaml',
			epochs = hyperparameters["epochs"], 
			imgsz = hyperparameters["imgsz"], 
			batch = hyperparameters["batch"], 
			device = hyperparameters["device"],
			patience = hyperparameters["patience"],
			lr0 = hyperparameters["lr0"],
			optimizer = hyperparameters["optimizer"],
			weight_decay = hyperparameters["weight_decay"],
			name = experiment_name,
			augment = True,
			hsv_h = 0.15,
			hsv_s = 0.4,
			hsv_v = 0.4,
			degrees = 0.0,
			translate = 0.1,
			scale = 0.5,
			shear = 0.0,
			perspective = 0.0,
			flipud = 0.5,
			fliplr = 0.5,
			bgr = 0.5,
			mosaic = 1.0
	)