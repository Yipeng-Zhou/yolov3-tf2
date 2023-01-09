import os
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models_00000 import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './scripts/checkpoints/00000/yolov3_train_12.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', '/home/yipeng_zhou/yolov3-tf2/data/coco2voc2tfrecord/voc_coco2017_val/JPEGImages', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        for image in os.listdir(FLAGS.image):
            image_fullname = os.path.join(FLAGS.image, image) 
            image_id = image.split('.jpg')[0]
            img_raw = tf.image.decode_image(open(image_fullname, 'rb').read(), channels=3)

            img = tf.expand_dims(img_raw, 0)
            img = transform_images(img, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            result_path = "./results/{}.txt".format(image_id)
            img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
            wh = np.flip(img.shape[0:2])
            with open(result_path, 'w') as f:
                for i in range(nums[0]):
                    x1y1 = tuple((np.array(boxes[0][i][0:2]) * wh).astype(np.int32))
                    x2y2 = tuple((np.array(boxes[0][i][2:4]) * wh).astype(np.int32))
                    f.write(class_names[int(classes[0][i])])
                    f.write(' ')
                    f.write(np.array2string(np.array(scores[0][i])))
                    f.write(' '),
                    f.write(np.array2string(np.array(x1y1[0])))
                    f.write(' ')
                    f.write(np.array2string(np.array(x1y1[1])))
                    f.write(' ')
                    f.write(np.array2string(np.array(x2y2[0])))
                    f.write(' ')
                    f.write(np.array2string(np.array(x2y2[1])))
                    f.write('\n')
            f.close()
        logging.info('finish!')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
