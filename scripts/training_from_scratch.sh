python ../train/train_00021.py \
    --tiny \
    --dataset ../data/coco2voc2tfrecord/coco2017_train.tfrecord \
    --val_dataset ../data/coco2voc2tfrecord/coco2017_val.tfrecord \
    --classes ../data/coco.names \
    --num_classes 80 \
    --mode fit \
    --transfer none \
    --batch_size 32 \
    --epochs 12 