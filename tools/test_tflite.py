import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf

from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

flags.DEFINE_string('model', './checkpoints/yolov3-tiny.tflite',
                    'path to saved_model')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('image', './data/000000000139.jpg', 'path to input image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('size', 416, 'image size')

def main(_argv):
    interpreter = tf.lite.Interpreter(model_path=FLAGS.model)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    print(img)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)
    print(img)

    t1 = time.time()
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    classes = interpreter.get_tensor(output_details[0]['index'])
    nums = interpreter.get_tensor(output_details[1]['index'])
    boxes = interpreter.get_tensor(output_details[2]['index'])
    scores = interpreter.get_tensor(output_details[3]['index'])

    print(nums)
    print(classes)
    print(scores)
    print(boxes)

    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

if __name__ == '__main__':
    app.run(main)