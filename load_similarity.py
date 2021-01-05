import cv2
import tensorflow as tf
import numpy as np
from keras import Model

def load_similarity_model(modelpath,fclayer):
    model = tf.keras.models.load_model(modelpath)
    return Model(model.input, model.get_layer(fclayer).output)

def preprocess_for_predict(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    imgarr = img.reshape((1,) + img.shape)
    return imgarr

def getpred(img,model):
    preds = model.predict(img)
    return preds

def getCSVwithVector(from_csv_path,model,to_csv_path):
    df = pd.read_csv(from_csv_path)
    li = []
    for index,row in df.iterrows():
        path = 'ikeadata/'+row['item_cat']+'/'+str(index)+'.jpg'
        img_array = preprocess_for_predict(path)
        result = getpred(img_array,model)
        li.append(result)
        print('running'+str(index))

    vector = pd.DataFrame({'vector':li})

    newdf_1 = df.join(vector)
    newdf_1.to_pickle(to_csv_path)

