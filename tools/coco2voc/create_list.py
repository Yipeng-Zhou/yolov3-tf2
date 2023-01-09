# coding=utf-8
'''
create list of train2017 or val2017
the output is train2017.txt val2017.txt
在train2017.txt中保存的格式是每一行都为imagename
'''
import os

def create_list(img_path, outlist):
    # get image list
    imglist = []
    for file in os.listdir(img_path):
        if file.split('.')[-1] == 'jpg':
            imglist.append(str(file))

    # imglist loop
    for img in imglist:
        # print(img)
        name = img.split('.')[0]
        fd = open(outlist, 'a')
        img_dir = name
        line = img_dir
        fd.write(line)
        fd.write('\n')
    fd.close()


if __name__ == '__main__':
    img_path = '/home/yipeng_zhou/yolov3-tf2/data/coco2voc2tfrecord/val2017/'
    outlist = '/home/yipeng_zhou/yolov3-tf2/data/coco2voc2tfrecord/val.txt'
    create_list(img_path, outlist)