from ultralytics import YOLO
# Load a model
model = YOLO('yolo12m.pt')
# Train the model
results = model.train(data='dronesite.yaml', epochs=12, imgsz=640, batch=8)
# validate the model
metrics = model.val(data='dronesite.yaml', batch=8)
print(metrics)