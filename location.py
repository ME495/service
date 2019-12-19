from flask import Flask, request
import requests, json
import numpy as np
import tensorflow as tf
import pickle
import ctrl_model
import os
import cv2
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = './ckpt/TALL_c3d.ckpt-12000'
video_dir = '/data02/chengjian19/videos/'

app = Flask(__name__)

model = ctrl_model.CTRL_Model()
loss_align_reg, vs_train_op, vs_eval_op, offset_pred, loss_reg = model.construct_model()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())
saver.restore(sess, model_name)

def get_visual_featmap(start_center, end_center, feature_center, feature_list):
    feature_pre = feature_center[:]
    feature_after = feature_center[:]
    start_pre, start_after = 0., 100000000.
    for clip_name, feature in feature_list:
        start = float(clip_name.split("_")[0])
        end = float(clip_name.split("_")[1])
        if end-start!=end_center-start_center:
            continue
        if end <= start_center:
            if start_pre < start:
                start_pre = start
                feature_pre = feature
        if start >= end_center:
            if start_after > start:
                start_after = start
                feature_after = feature
    featmap = np.concatenate((feature_pre, feature_center, feature_after))
    featmap = np.reshape(featmap, [1, -1])
    return featmap

def eval_clips(feature_list, vector):
    vector = np.reshape(vector, [1, -1])
    sentence_image_mat = np.zeros(len(feature_list))
    sentence_image_reg_mat = np.zeros((len(feature_list), 2))
    for i, (clip_name, feature) in enumerate(feature_list):
        start = float(clip_name.split("_")[0])
        end = float(clip_name.split("_")[1])
        
        featmap = get_visual_featmap(start, end, feature, feature_list)
        
        feed_dict = {
            model.visual_featmap_ph_test: featmap,
            model.sentence_ph_test:vector
        }
        outputs = sess.run(vs_eval_op, feed_dict=feed_dict)
        sentence_image_mat[i] = outputs[0]
        reg_start = start+outputs[1]
        reg_end = end+outputs[2]
        
        sentence_image_reg_mat[i,0] = reg_start
        sentence_image_reg_mat[i,1] = reg_end
    
    index = np.argsort(-sentence_image_mat)
    results = sentence_image_reg_mat[index[:5]]
    return results


@app.route('/', methods=['POST'])
def location():
    sentence = request.form['sentence'].encode()
    video_name = request.form['video_name'].encode()
    
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
#     print fps
    
    url = 'http://12.12.12.2:7000/'
    data = {'video_name': video_name}
    r = requests.post(url, data)
    if r.status_code==500:
        return 'No video!', 500
    with open(r.content, 'r') as f:
        feature_list = pickle.load(f)
    
    url = 'http://12.12.12.2:7005/'
    data = {'sentence': sentence}
    r = requests.post(url, data)
    vector = np.asarray(list(map(float, r.content.split(','))))
    
    results = eval_clips(feature_list, vector)
#     print(results)
    results = results/fps
    
    resp = {}
    for i in range(5):
        resp['start%d'%(i+1)] = results[i,0]
        resp['end%d'%(i+1)] = results[i,1]
    resp = json.dumps(resp)
    
    return resp, 200


if __name__ == '__main__':
    app.run(
      host='12.12.12.3',
      port= 7009,
      debug=True
    )