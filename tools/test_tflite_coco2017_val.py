import os
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
flags.DEFINE_string('image', '/home/yipeng_zhou/yolov3-tf2/data/coco2voc2tfrecord/voc_coco2017_val/JPEGImages', 'path to input image')
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

    for image in os.listdir(FLAGS.image):
        image_fullname = os.path.join(FLAGS.image, image) 
        image_id = image.split('.jpg')[0]
        img_raw = tf.image.decode_image(open(image_fullname, 'rb').read(), channels=3)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)

        t1 = time.time()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        classes = interpreter.get_tensor(output_details[0]['index'])
        nums = interpreter.get_tensor(output_details[1]['index'])
        boxes = interpreter.get_tensor(output_details[2]['index'])
        scores = interpreter.get_tensor(output_details[3]['index'])

        result_path = "./results/{}.txt".format(image_id)
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        wh = np.flip(img.shape[0:2]) # [width,height] e.g.[640,426]
        with open(result_path, 'w') as f:
            for i in range(nums[0]):
                if scores[0][i] == 0:
                    break
                else:
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
    app.run(main)