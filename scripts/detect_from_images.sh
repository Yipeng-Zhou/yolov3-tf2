CUDA_VISIBLE_DEVICES=1 \
python ../detect.py \
    --tiny \
	--classes ../data/coco.names \
	--num_classes 80 \
	--weights ../checkpoints/yolov3-tiny.tf \
	--image ../data/street.jpg