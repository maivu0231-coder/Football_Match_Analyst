
from ultralytics import YOLO
model	= YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
#train the model
if __name__ == '__main__':
    model.train(
	data= "Personal_Project/Football_Analyst_Project/Training/Dataset/data.yaml",  # dataset path
	epochs=100,
	imgsz=640,
	device=0,
	batch=16,
	workers=0,
	project="Football_Analyst_Project",  # project name
	name="football_players_detection"
)