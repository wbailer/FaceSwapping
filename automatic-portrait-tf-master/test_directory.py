import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2
import os
import sys

import slim_net


NUM_CLASSES = 2

COLOR_SET = [
    [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
    [187, 119, 132], [142, 6, 59], [74, 111, 227], [133, 149, 225],
    [181, 187, 227], [230, 175, 185], [224, 123, 145], [211, 63, 106],
    [17, 198, 56], [141, 213, 147], [198, 222, 199], [234, 211, 198],
    [240, 185, 141], [239, 151, 8], [15, 207, 192], [156, 222, 214],
    [213, 234, 231], [243, 225, 235], [246, 196, 225], [247, 156, 212]
]


def build_image(filename):
    MEAN_VALUES = np.array([104.00698793, 116.66876762, 122.67891434])
    MEAN_VALUES = MEAN_VALUES.reshape((1, 1, 1, 3))
    img = scipy.misc.imread(filename, mode='RGB')[:, :, ::-1]
    height, width, _ = img.shape
    img = np.reshape(img, (1, height, width, 3)) - MEAN_VALUES
    return img


def save_masked_image(result, srcfilename, filename):
    srcimg = scipy.misc.imread(srcfilename, mode='RGB')
    height, width, _ = srcimg.shape
    srcimg = np.reshape(srcimg, (height, width, 3))	
    srcimg = np.asarray(srcimg, np.float32)
    _, h, w = result.shape
    result = result.reshape(h * w)
    image = []
    for v in result:
        image.append(COLOR_SET[v])
    image = np.array(image)
    image = np.reshape(image, (h, w, 3))    
    image = np.asarray(image, np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)) #np.ones((50,50),np.float32)
    large_image = np.zeros((2*h,2*w,3),np.float32)
    large_image = np.reshape(large_image, (2*h, 2*w, 3))	
    large_image[int(h/2):int(h/2+h),int(w/2):int(w/2+w),:] = image
    closing = cv2.morphologyEx(large_image, cv2.MORPH_CLOSE, kernel)
    closing_center = closing[int(h/2):int(h/2+h),int(w/2):int(w/2+w),:]
    masked_image = cv2.multiply(closing_center, srcimg)
    scipy.misc.imsave(filename, masked_image)


def test(image_name):
    inputs = tf.placeholder(tf.float32, [1, None, None, 3])
    with slim.arg_scope(slim_net.fcn8s_arg_scope()):
        logits, _ = slim_net.fcn8s(inputs, NUM_CLASSES)

    image = build_image(image_name)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint('./model/')

        if model_file:
            saver.restore(sess, model_file)
        else:
            raise Exception('Testing needs pre-trained model!')

        feed_dict = {
            inputs: image,
        }
        result = sess.run(tf.argmax(logits, axis=-1), feed_dict=feed_dict)
	# added to enable second run, see stackoverflow.com/questions/44312985/how-to-reuse-slim-arg-scope-in-tensorflow
    tf.reset_default_graph()
    return result


if __name__ == '__main__':
    #result_image = test("D:/project/pp_training/celebA/000001.jpg")
    print("done")
    #save_masked_image(result_image, "image.jpg", "result.jpg")
	
    srcpath = sys.argv[1]
    dstpath = sys.argv[2]
	
    files = [i for i in os.listdir(srcpath) if i.endswith("jpg")]

    for file in files:
        print(file)
        full_src_name = srcpath + "/" + file
        #full_src_name = "D:/project/pp_training/celebA/000001.jpg"
        #full_src_name = full_src_name.replace("\\", "/")
        print(full_src_name)		
        
        resimg = test(full_src_name)
        full_dst_name = dstpath + "/" + file
        save_masked_image(resimg, full_src_name, full_dst_name)
