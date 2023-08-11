# 知识蒸馏

## 训练教师模型

    model_t = YOLO('yolov8l.pt')
    model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)

## 学生模型

    model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)

## 效果

    炼丹需要看各位施主的缘分。一般提高0.2-0.5map。

## 心得

    大学教授不一定教好小学生。