from ultralytics import YOLO
# Load a model
model = YOLO('yolo11m.pt', verbose=False)
# Train the model
results = model.train(data='dronesite.yaml', epochs=100, imgsz=640, batch=32)
# validate the model
metrics = model.val(data='dronesite.yaml', batch=64)
print(metrics)