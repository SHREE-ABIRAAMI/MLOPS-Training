# YOLO Model

Place your converted YOLO model here as **`best.onnx`**.

## Convert best.pt → best.onnx

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=640, opset=12, dynamic=False, simplify=True)"
```

This produces `best.onnx`. Copy it to `public/models/best.onnx`.

The app expects a single-class (or multi-class) YOLOv5/v8/v11 model with input
size 640x640. The first detected class is treated as "phone".
