CUDA_VISIBLE_DEVICES=2,3 python3 ./train.py \
                                --data ./data/coco.yaml \
                                --epochs 300 \
                                --weights '' \
                                --cfg ./models/yolov5s.yaml \
                                --batch-size 64 \
                                --device 2,3 \
                                --name coco_debug \
                                --use_differentiable_nms
                                