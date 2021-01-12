# iDECOR - Furniture Detector & Recommender

<p>This repository contains all the codes to complete iDECOR, our Furniture Recommender that help users who have recently moved to explore IKEA products at ease.</p> 

#### What does iDECOR do?
<p>After uploading a room scene image, IDECOR returns users with similar-styled furnitures in favour from IKEA dataset.</p>




#### Table of Contents
- Project Overview
- Business Objectives
- Data Collection
- Data Preprocessing
- Architecture
- Modeling
  - Object Detection: Detectron2 & Faster R-CNN
  - Similarity Detection: VGG16 vs IncepctionV3
- Evaluation
- Deployment
  - Streamlit
- Recommendations
- Skills
- Pitch
- Credits
- Skills


#### Project Overview 
<p>iDECOR is a customized solution for furniture shoppers by wrapping furniture detection and recommendation system into a single, seamless process on Streamlit. This product leverages two deep learning technologies - 1) Detectron2 developed by Facebook to detect every possible furniture item in the user's uploaded image, and 2) VGG16 model to help narrow down most similar items to the detected item in terms of visual cues, and return them with top 5 items from our furniture product catalogue, in order to drive conversions to sales of furniture retailers by improving onsite recommendations. iDECOR is made available across six furniture products as the first step, and will expand to more brands/categories with API support in the future.</p>

#### Business Objectives
<li>To drive IKEA's (or other furntiure e-commerce stores) sales by providing better onsite recommendations. </li>

#### Prerequisites
1. CPU
2. GPU x 1

#### Installation
1. Detectron2 Library
- Built on PyTorch
- NVIDIA GPU requirement

2. Similarity Detection


#### Data Collection
1. Bonn Furniture Style Dataset:  Contains 90000+ furniture images with style labelled by interior design professionals. In the dataset, there are 6 different furniture types. The furniture in the dataset is categorized into beds, chairs, dressers, lamps, sofas and tables. [1]

2. Open Images Dataset V6: A dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. [2]

3. IKEA Product Catalogue Dataset: Web-scraped 1000+ products directly from IKEA site for recommendations to user later. [3]


#### Data Preprocessing:
- Balanced Dataset
- Detectron2 Formatting with Annotations
- Mutli-label Binarizers
- Annotations 
- Image resize before training
- Drop duplicates
- Augmentation



#### Architecture
<img src="https://github.com/sophiachann/ObjectDetectionProject-FurnitureRecommender/blob/main/images/system-architecture.png" alt="Use flow of Detection & Recommendation Models" width="800" height="370">

<p>This flowchart shows five main stages during the whole development of our product, with different focus, tools and technologies we have used within which we will explain further. </p>

1. Data Acquisition
- Collected 3 types of datasets for training of both Detectron2 and VGG16 models, and for recommendations as feedback, as mentioned in the above section.

2. Data Preprocessing
- Took advantage of the high computational power of Google Colab and Google Cloud Platform (GCP) to help filter to desired categories, annotate bouding box coordinates, resize images, encode multiple labels per instance, and augmentate the images, for model training.

3. Model Engineering
- Prepare the codes as the skeleton to train and make inference on the models.
- Built the configurations crucial for the training of Detectron2 model using built-in DefaultTrainer with passing annotation dicts and labelled images, and trained it using Pytorch.
- Compared similarity model between VGG16 and InceptionV3 using transfer learning and established our final model with VGG16 upon Keras and Tensorflow.

4. Evaluation
- 

#### References:
1. Bonn Furniture Style Dataset: https://cvml.comp.nus.edu.sg/furniture/index.html
2. Open Images Dataset V6: https://storage.googleapis.com/openimages/web/index.html
3. IKEA site: https://www.ikea.com.hk/en

