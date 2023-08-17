# 知识蒸馏

## 训练教师模型

    model_t = YOLO('yolov8l.pt')
    model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)

## 学生模型

    model_s = YOLO('yolov8s.pt')
    model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)

## 提示

    选择蒸馏方法施主自行到/ultralytics/yolo/engine/trainer.py：176行更改通道数以及CWDLoss 或者 MGDLoss。

## 效果

    炼丹需要看各位施主的缘分。一般提高0.2-0.5map。

## 心得

    大学教授不一定教好小学生。
