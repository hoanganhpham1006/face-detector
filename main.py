import time
import imutils
import dlib
import cv2
import pickle
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras import applications
import tensorflow as tf
from keras import backend as K
import h5py


def convnet_model_():
    vgg_model = applications.VGG16(weights=None, include_top=False, input_shape=(221, 221, 3))
    x = vgg_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Lambda(lambda x_: K.l2_normalize(x,axis=1))(x)
#     x = Lambda(K.l2_normalize)(x)
    convnet_model = Model(inputs=vgg_model.input, outputs=x)
    return convnet_model

def deep_rank_model():
    convnet_model = convnet_model_()

    first_input = Input(shape=(221, 221, 3))
    first_conv = Conv2D(96, kernel_size=(8,8), strides=(16,16), padding='same')(first_input)
    first_max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')(first_conv)
    first_max = Flatten()(first_max)
#     first_max = Lambda(K.l2_normalize)(first_max)
    first_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(first_max)

    second_input = Input(shape=(221, 221, 3))
    second_conv = Conv2D(96, kernel_size=(8,8), strides=(32,32), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7), strides=(4,4), padding='same')(second_conv)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda x: K.l2_normalize(x, axis=1))(second_max)
#     second_max = Lambda(K.l2_normalize)(second_max)
                       
    merge_one = concatenate([first_max, second_max])
    merge_two = concatenate([merge_one, convnet_model.output])
    emb = Dense(4096)(merge_two)
    emb = Dense(128)(emb)
    l2_norm_final = Lambda(lambda x: K.l2_normalize(x, axis=1))(emb)
#     l2_norm_final = Lambda(K.l2_normalize)(emb)
                        
    final_model = Model(inputs=[first_input, second_input, convnet_model.input], outputs=l2_norm_final)

    return final_model

#Load model
deep_rank_model = deep_rank_model()
deep_rank_model.load_weights('/home/pham.hoang.anh/prj/face_detect/triplet_weight.hdf5')

#Load all vector embedding LFW of my model
with open('/home/pham.hoang.anh/prj/face_detect/embs128.pkl', 'rb') as f:
    embs128 = pickle.load(f)
with open('/home/pham.hoang.anh/prj/face_detect/visualize/128D-Facenet-LFW-Embedding-Visualisation/oss_data/LFW_128_HA_labels.tsv', 'r') as f:
    names = f.readlines()

#Read image
image = cv2.imread('/home/pham.hoang.anh/Desktop/1.jpg')
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hog_face_detector = dlib.get_frontal_face_detector()
# cnn_face_detector = dlib.cnn_face_detection_model_v1('/Users/phamhoanganh/Desktop/mmod_human_face_detector.dat')

# start = time.time()
faces_hog = hog_face_detector(image, 1)
# end = time.time()
# print("Hog + SVM Execution time: " + str(end-start))

for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    # draw green box over face which detect by hog + svm
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

    #Get vector embeding
    frame = image[y:y+h, x:x+w]
    frame = cv2.resize(frame, (221, 221))
    frame = frame /255.
    frame = np.expand_dims(frame, axis=0)
    emb128 = deep_rank_model.predict([frame, frame, frame])
    minimum = 99999
    person = -1
    for k, e in enumerate(embs128):
        #Euler distance
        dist = np.linalg.norm(emb128-e)
        if dist < minimum:
            minimum = dist
            person = k
    cv2.putText(image, names[person], (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# start = time.time()
# faces_cnn = cnn_face_detector(image, 1)
# end = time.time()
# print("CNN Execution time: " + str(end-start))

# # loop over detected faces
# for face in faces_cnn:
#   x = face.rect.left()
#   y = face.rect.top()
#   w = face.rect.right() - x
#   h = face.rect.bottom() - y

#   # draw red box over face which detect by cnn
#   cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.imwrite("/home/pham.hoang.anh/prj/face_detect/img_res.jpg", image)
