import streamlit as st
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import os
import cv2
import time
import os
import glob
import scipy.spatial.distance as distance
import re

import tensorflow as tf
from keras import Model


from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


target_classess=['Bed', 'Dresser', 'Chair', 'Sofa', 'Lamp', 'Table']
app_dir =  'appdata/'
save_img_dir = 'appdata/detect/'
dfpath = 'ikeadata/ikea_final_model0.csv'

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_ikeadf(path):
    return pd.read_pickle(path)
     
#Initialize the detectron model
@st.cache(allow_output_mutation=True,show_spinner=False)
def initialization():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # Set threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = "model/faster_rcnn_X_101_32x8d_FPN_3x_18000iter_0_0025lr_model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 
    predictor = DefaultPredictor(cfg)
    return cfg, predictor



#Get a list of bounding box item with class
def get_bbox_list(outputs):

    bbox_list=[]
    bbox_class_list=outputs["instances"].pred_classes     # get each predicted class from output, return a torch tensor
    bbox_cor_list=outputs["instances"].pred_boxes          # get each bounding box coordinate(xmin_ymin_xmax_ymax) from output, return a torch tensor
    bbox_class_list=bbox_class_list.cpu().numpy()         # convert class to numpy 
    
    #conver coordinate to numpy
    new_list=[]
    for i in bbox_cor_list:                               
        i=i.cpu().numpy()
        new_list.append(i)
    bbox_cor_list=new_list
    #combine to a new list with dict of class and coordinate
    for i in range(len(bbox_class_list)):                
        # store each class and corresponding coordinate to dict
        temp_dict={'class':bbox_class_list[i],'coordinate':bbox_cor_list[i]}  
        bbox_list.append(temp_dict)

    return bbox_list


def save_bbox_image(bgr_image,bbox_list,save_img_dir):
    img_dict=[]
    counter=1

    #Convert CV2 image to PIL Image for cropping
    img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    finalimg = Image.fromarray(img)

    for i in bbox_list:
        #name img file with index
        file_name=str(counter)+'.jpg'
        path=save_img_dir+'/'+file_name
        # get bounding box coordinate for corping
        coordinate=i.get('coordinate')  
        # bbox coordinate should be in list when out put from detectron and change to tuple
        coordinate=tuple(coordinate)
        #crop image and save
        crop_img=finalimg.crop(coordinate)
        crop_img.save(path)
        #store it in a dictionary with file name and class
        temp_dict={'File_name':file_name,'class':target_classess[int(i['class'])]}
        counter+=1
        img_dict.append(temp_dict)

    return img_dict

@st.cache(allow_output_mutation=True,show_spinner=False)
def load_similarity_model():
    model = tf.keras.models.load_model("model/multilabel0")
    model_new = Model(model.input, model.get_layer('dense_4').output)
    return model_new


def preprocess_for_predict(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    imgarr = img.reshape((1,) + img.shape)
    return imgarr

def getpred(img,pred_model):
    preds = pred_model.predict(img)
    return preds

def calculate_similarity(vector1,vector2):
    return 1-distance.cosine(vector1, vector2)


def compare_similarity(image_arr,model_dense):
    vector_obj = getpred(image_arr,model_dense)
    ikeadf['similarity'] = ikeadf['vector'].apply(lambda x:calculate_similarity(x,vector_obj))
    return ikeadf.sort_values(by=['similarity'],ascending=False).head(1000)

def clearold():
    files = glob.glob('appdata/'+'*.jpg')
    if files:
        for f in files:
            os.remove(f)

@st.cache(show_spinner=False)
def GetImage(user_img_path):
    # cfg, predictor = initialization()
    #delete unncessay history file
    files = glob.glob(save_img_dir+'*.jpg')
    if files:
        for f in files:
            os.remove(f)

    imagecv = cv2.imread(user_img_path)
    outputs = predictor(imagecv)
    bbox_list=get_bbox_list(outputs) # get each bbx info
    final_img_dict = save_bbox_image(imagecv, bbox_list,save_img_dir)
    #get a furniture option list which is well formatted for users to read
    

    return bbox_list,final_img_dict

def getfurnlist(img_dict):
    furniture_list = []
    for index, item in enumerate(img_dict):
        furniture_list.append(str(index+1)+' - '+item['class'])

    return furniture_list


#app start
st.set_page_config(layout="wide")
cfg, predictor = initialization()
model = load_similarity_model()

st.image(Image.open("idecor_logo.png"), width = 700)
st.write('**_Here is where you furnish your home with a Click, just from your couch._**')
st.sidebar.header("Choose a furniture image for recommendation.")

#load the model


clearold()

uploaded_file = st.file_uploader("Choose an image with .jpg format", type="jpg")

if uploaded_file is not None:
    #save user image and display success message
    ikeadf = load_ikeadf(dfpath)
    
    image = Image.open(uploaded_file)
    user_img_path = app_dir+uploaded_file.name
    image.save(user_img_path)
    
    st.sidebar.image(image,width = 250)

    st.sidebar.success('Upload Successful! Please wait for object detection.')

    #get image list from the detectron model
    with st.spinner('Working hard on finding furniture...'):
        #delete unncessay history file


        bbli,imgdict= GetImage(user_img_path)
        furniturelist = getfurnlist(imgdict)
        #open cropped image of furniture
        for i,file in enumerate(furniturelist):
            st.sidebar.write(file)
            d_image = save_img_dir+str(i+1)+'.jpg'
            st.sidebar.image(Image.open(d_image),width = 150)

    #provide select box for selection
    display = furniturelist
    options = list(range(len(furniturelist)))
    option = st.selectbox('Which furniture do you want to look for?', options, format_func=lambda x: display[x])
    
    if st.button('Confirm to select '+furniturelist[option]):
        
        pred_path = save_img_dir+str(option+1)+'.jpg'
        image_array = preprocess_for_predict(pred_path)
        # model = load_similarity_model()
        df = compare_similarity(image_array,model)
        obj_class = imgdict[option]['class'].lower()+'s'
        st.write("Recommendation for: "+imgdict[option]['class']+'s')

        c1, c2, c3, c4, c5 = st.beta_columns((1, 1, 1, 1, 1))
        columnli = [c1,c2,c3,c4,c5]

        for i,column in enumerate(columnli):
            coltitle = re.match(r"^([^,]*)",str(df[df['item_cat']==obj_class][i:i+1].item_name.values.astype(str)[0])).group()
            colcat = str(df[df['item_cat']==obj_class][i:i+1].item_cat.values.astype(str)[0])
            colpic = str(df[df['item_cat']==obj_class][i:i+1].index.values.astype(str)[0])
            colprice = '$' + str(df[df['item_cat']==obj_class][i:i+1].item_price.values.astype(str)[0])
            collink = str(df[df['item_cat']==obj_class][i:i+1].prod_url.values.astype(str)[0])
            colurl = 'ikeadata/'+colcat+'/'+colpic+'.jpg'
            column.image(Image.open(colurl),width=180)
            column.write('### '+colprice)  
            column.write('##### '+coltitle)
            column.write("##### "+"[View more product info]("+collink+")")
            # column.write("[!["+coltitle+"]("+colurl+")]("+collink+")")

        st.text("")
        st.write("Some other non-"+imgdict[option]['class']+"s items you may like: ")

        c6,c7,c8,c9,c10 = st.beta_columns((1, 1, 1, 1, 1))
        columnli2 = [ c6,c7,c8,c9,c10]
        for i,column in enumerate(columnli2):
            coltitle = re.match(r"^([^,]*)",str(df[df['item_cat']!=obj_class][i:i+1].item_name.values.astype(str)[0])).group()
            colcat = str(df[df['item_cat']!=obj_class][i:i+1].item_cat.values.astype(str)[0])
            colpic = str(df[df['item_cat']!=obj_class][i:i+1].index.values.astype(str)[0])
            colprice = '$' + str(df[df['item_cat']!=obj_class][i:i+1].item_price.values.astype(str)[0])
            colurl = 'ikeadata/'+colcat+'/'+colpic+'.jpg'
            column.image(Image.open(colurl),width=180)
            column.write('### '+colprice)  
            column.write('##### '+coltitle)
            column.write("##### "+"[View more product info]("+collink+")")
