from flask import Flask, request
import cv2
import tensorflow as tf
import c3d_model
import os
import numpy as np
import pickle

app = Flask(__name__)

video_dir = '/data02/chengjian19/videos/'
feature_dir = os.path.abspath('./video_feature/')
clip_lens = [128, 64]
overlap = 0.5
model_name = "./sports1m_finetuning_ucf101.model"
batch_size = 10

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def clip_generator(clip_dic):
    for clip_name in clip_dic.keys():
        clip = clip_dic[clip_name]
        yield clip_name, clip
        
def construct_net():
    images_placeholder, labels_placeholder = placeholder_inputs(batch_size)
    with tf.variable_scope('var_name') as var_scope:
        weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
                }
        biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
                }
    with tf.device('/gpu:2'):
        dense1, logit = c3d_model.inference_c3d(images_placeholder, 1.0, weights, biases)
    return images_placeholder, labels_placeholder, dense1

images_placeholder, labels_placeholder, dense1 = construct_net()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
saver.restore(sess, model_name)

def get_clip(video_name):
    print 'Loading video.'
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    frames = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    cap.release()
    frames = np.asarray(frames)
    print 'Generating clip.'
    clip_dic = {}
    for clip_len in clip_lens:
        step = int(clip_len*(1-overlap))
        for i in np.arange(0, frames.shape[0], step):
            if i+clip_len>frames.shape[0]:
                break
            key = "%d_%d" % (i+1, i+clip_len+1)
            sample = frames[np.linspace(i, i+clip_len-1, 16, dtype=np.int32)]
            clip_dic[key] = sample
    return clip_dic

def get_feature(clip_dic):
    print 'Extracing feature.'
    generator = clip_generator(clip_dic)
    feature_list = []
    while True:
        batch_clip_name = []
        batch_clip_data = []
        for i in range(batch_size):
            try:
                clip_name, clip_data = next(generator)
            except StopIteration:
                break
            batch_clip_name.append(clip_name)
            batch_clip_data.append(clip_data)
        if len(batch_clip_name)==0:
            break
        batch_clip_data = np.asarray(batch_clip_data)
        batch_feature = sess.run(dense1, feed_dict={images_placeholder: batch_clip_data})
        for clip_name, feature in zip(batch_clip_name, batch_feature):
            feature_list.append((clip_name, feature))
    return feature_list

@app.route('/', methods=['POST'])
def extrac_video_feature():
    video_name = request.form['video_name']
    video_path = os.path.join(video_dir, video_name)
    if not os.path.isfile(video_path):
        return "No video!", 500
    feature_name = '.'.join(video_name.split('.')[:-1])
    feature_name = feature_name + '.pkl'
    feature_path = os.path.join(feature_dir, feature_name)
    if not os.path.isfile(feature_path):
        clip_dic = get_clip(video_name)
        feature_list = get_feature(clip_dic)
        with open(feature_path, 'w') as f:
            pickle.dump(feature_list, f)
    return feature_path, 200

if __name__ == '__main__':
    app.run(
      host='12.12.12.2',
      port= 7000,
      debug=True
    )