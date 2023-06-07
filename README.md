本项目基于YoloV5，添加了[GrooMeD-NMS](https://github.com/abhi1kumar/groomed_nms)。

## 项目已完成内容
1. 在yolov5的推理中可以替换为GrooMeD-NMS，推理结果与传统NMS一致。

## 待完成内容
1. 在yolov5的训练中修改为GrooMeD-NMS的方式，需要修改loss。
2. 训练流程尚未跑通。
3. 目前groomed_nms.py中的超参数尚未设置外部的接口

## 使用说明
与yolov5使用方法一致，在val中添加了一个参数use_differentiable_nms来选择NMS方式。

```bash
sh val.sh 
```

相比原始yolov5，目前添加的代码都在utils文件夹下，包括aploss.py、groomed_nms.py、loss_gnms.py。
