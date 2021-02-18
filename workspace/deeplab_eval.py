"""
deeplab Evaluation Script
"""
import os
import argparse
import sys
import tensorflow as tf
import tensorflow.contrib.decent_q
import cv2
import numpy

FLAGS = None

_R_MEAN = 128.00
_G_MEAN = 128.00
_B_MEAN = 128.00

MEANS = [_B_MEAN,_G_MEAN,_R_MEAN]


label_colours_AIedge_RGB = [(0, 0, 255), (193, 214, 0), (180, 0, 129), (255, 121, 166), (255, 0, 0),  # 0=Car,1=Bus, 2=Truck, 3=SVehicle, 4=Pedestrian
                        (65, 166, 1), (208, 149, 1), (255, 255, 0), (255, 134, 0), (0, 152, 225), # 5=Moterbike,  6=Bicycle, 7=Signal, 8=Signs, 9=Sky
                        (0, 203, 151), (85, 255, 50), (92, 136, 125), (69, 47, 142), (136, 45, 66), # 10=Building, 11=Natural, 12=Wall, 13=Lane, 14=Ground
                        (0, 255, 255), (215, 0, 255), (180, 131, 135), (81, 99, 0), (86, 62, 67), # 15=Sidewalk, 16=RoadShoulder, 17=Obstacle, 18=others, 19=own 
                        (0, 0, 0)] # 20=None

label_colours_AIedge_BGR = [(255, 0, 0), (0, 214, 193), (129, 0, 180), (166, 121, 255), (0, 0, 255),  # 0=Car,1=Bus, 2=Truck, 3=SVehicle, 4=Pedestrian
                        (1, 166, 65), (1, 149, 208), (0, 255, 255), (0, 134, 255), (255, 152, 0), # 5=Moterbike,  6=Bicycle, 7=Signal, 8=Signs, 9=Sky
                        (151, 203, 0), (50, 255, 85), (125, 136, 92), (142, 47, 69), (66, 45, 136), # 10=Building, 11=Natural, 12=Wall, 13=Lane, 14=Ground
                        (255, 255, 0), (255, 0, 215), (135, 131, 180), (0, 99, 81), (67, 62, 86), # 15=Sidewalk, 16=RoadShoulder, 17=Obstacle, 18=others, 19=own 
                        (0, 0, 0)] # 20=None

label_colours_AIedge_BGR_4cate = [(255, 0, 0), (0, 0, 255), (0, 255, 255), (142, 47, 69), (0, 0, 0)]  # 0=Car,1=Pedestrian, 2=Signal, 3=Lane, 4=else


def decode_labels(image, num_classes=4):
    h, w, c = image.shape
    outputs = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    for i in range(h):
        for j in range(w):
            if image[i, j, 0] < num_classes:
                outputs[i, j, :] = label_colours_AIedge_BGR_4cate[image[i, j, 0]]
    return outputs


def mean_image_subtraction(image, means):
  B, G, R = cv2.split(image)
  B = B - means[0]
  G = G - means[1]
  R = R - means[2]
  image = cv2.merge([R, G, B])
  return image


def normalize_deeplab(image):
  image=image/255.0
  image=image*2
  image=image-1.0
  return image


def eval_input(iter, eval_image_dir, eval_image_list):
    with open(eval_image_list, 'r') as f:
        line = f.read().splitlines()
    image_name = line[iter]
    print(image_name)

    image = cv2.imread(eval_image_dir + image_name)

    image1 = image[100:1124,0:1024,:]
    image1 = cv2.resize(image1, (384, 384))
    image1 = mean_image_subtraction(image1, MEANS)

    images1 = []
    images1.append(image1)

    image2 = image[100:1124,912:1936,:]
    image2 = cv2.resize(image2, (384, 384))
    image2 = mean_image_subtraction(image2, MEANS)

    images2 = []
    images2.append(image2)

    return {"input1": images1, "input2": images2}


def deeplab_eval(input_graph_def, input_node, output_node):

    tf.import_graph_def(input_graph_def,name = '')

    input_tensor = tf.get_default_graph().get_tensor_by_name(input_node+':0')
    output = tf.get_default_graph().get_tensor_by_name(output_node+':0')
    prediction = tf.argmax(output,3)
    prediction = tf.squeeze(prediction, [0])

    with tf.Session() as sess:

        with open(FLAGS.eval_image_list, 'r') as f:
            line = f.read().splitlines()

        for iter in range(0,FLAGS.eval_batches):
            input_data = eval_input(iter, FLAGS.eval_image_dir, FLAGS.eval_image_list)
            
            image1 = input_data['input1']
            feed_dict = {input_tensor: image1,}
            pred1 = sess.run(prediction,feed_dict)
            pred1_resize = cv2.resize(pred1, dsize=(1024, 1024) ,interpolation=cv2.INTER_NEAREST)

            image2 = input_data['input2']
            feed_dict = {input_tensor: image2,}
            pred2 = sess.run(prediction,feed_dict)
            pred2_resize = cv2.resize(pred2, dsize=(1024, 1024) ,interpolation=cv2.INTER_NEAREST)

            pred = 20*numpy.ones((1216, 1936), dtype=numpy.uint8) #20:None
            pred[100:1124,0:1024] = pred1_resize
            pred[100:1124,968:1936] = pred2_resize[:,56:1024]

            cv2.imwrite(FLAGS.save_dir + line[iter].replace('.jpg', '.png'), pred)
            print(FLAGS.save_dir + line[iter].replace('.jpg', '.png'))

            img = cv2.imread(FLAGS.save_dir + line[iter].replace('.jpg', '.png'))
            img_decoded = decode_labels(img)
            cv2.imwrite(FLAGS.save_dir + line[iter].replace('.jpg', '.png'), img_decoded)


def main(unused_argv):
    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.gfile.FastGFile(FLAGS.input_frozen_graph, "rb").read())
    deeplab_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_frozen_graph', type=str,
                        default='./deeplab/vai_q_output/quantize_eval_model.pb',
                        help='frozen pb file.')
    parser.add_argument('--input_node', type=str,
                        default='Input',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='output_logit',
                        help='output node.')
    parser.add_argument('--eval_batches', type=int,
                        default=649,
                        help='number of total batches for evaluation.')   
    parser.add_argument('--eval_image_dir', type=str,
                        default="seg_test_images/",
                        help='evaluation image directory.')   
    parser.add_argument('--save_dir', type=str,
                        default="result/",
                        help='evaluation image directory.')   
    parser.add_argument('--eval_image_list', type=str,
                        default="seg_test_images/test_list.txt",
                        help='evaluation image list file.')  
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
