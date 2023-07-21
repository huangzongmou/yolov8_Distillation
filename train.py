import torch
import pdb
from ultralytics import YOLO

# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train32/weights/best.pt')
# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/coco_v8l/weights/best.pt')
# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('yolov8l.pt')
# success = modelL.export(format="onnx",device="cpu")

data = "/home/huangzm/code/mycode/pytorch/yolov8/Knowledge_Distillation/ultralytics/datasets/coco.yaml"
# model_t.model.model[-1].set_Distillation = True

# model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)

model_s = YOLO('yolov8n.pt')


# success = modeln.export(format="onnx")
# modelL.val(data=data)

# model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)
model_s.train(data=data, epochs=100, imgsz=640, Distillation = None)



