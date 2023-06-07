CUDA_VISIBLE_DEVICES=3 python3 ./val.py \
                                --data ./data/coco.yaml \
                                --batch-size 32 \
                                --name val_coco_nms
                                # --use_differentiable_nms