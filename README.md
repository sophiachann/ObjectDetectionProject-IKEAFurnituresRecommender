# Furniture Object Detector & Recommender using Detectron2 & VGG16

This repository contains all the code to complete iDECOR, a Furniture Recommender that help users who have recently moved to explore IKEA products at ease.

<img src="https://github.com/sophiachann/ObjectDetectionProject-FurnitureRecommender/blob/main/img/intro.png" width="800"/>

After uploading a room scene image, IDECOR returns users with similar-styled furnitures from IKEA product catalog.

### Business Values of this Project
Amazon is a giant in ecommerce which has its [35% revenue generated by its recommendation engine](https://rejoiner.com/resources/amazon-recommendations-secret-selling-online/). This proportion has potential to increase up to 60% based off its competitors.

This project is to drive conversions-to-sales of IKEA's and other furniture ecommerce stores, by providing **better onsite recommendations**.

### Project Overview
<img src="https://github.com/sophiachann/ObjectDetectionProject-FurnitureRecommender/blob/main/img/sys-architecture2.png" width="800"/>

- Developed a customized solution for furniture shoppers, by wrapping furniture detection and recommendation system into a seamless process on [Streamlit](https://www.streamlit.io/). 

- Leveraged 2 Deep Learning technologies: 
	- [Detectron2](https://ai.facebook.com/tools/detectron2/) (developed by Facebook AI): Detect every furniture item in user's uploaded image
	- VGG16 model: Detect furniture features and generate vector representations for respective furniture items

- Returned users with 5 most-similar furniture items, from product catalogue web-scraped from [IKEA Hong Kong](https://www.ikea.com.hk/en) using Selenium

- Made iDECOR available across 6 furniture products, and will expand to more brands/ categories with API support in the future.

## What is in this repo?

- `app` are some functions to deploy iDECOR with Streamlit.
- `ikea` contains programs to scrape IKEA product images and stores scraped data.
- `img` contains illustrations for this README.md
- `model-evaluation` are some plots made upon model inference.
- `model` contains the program and notebook that perform model training of Detectron2 and VGG16 models.
- `open-images` contains the codes to download and annotate images downloaded from ‘Open Images Dataset V6’
	- save annotation files here to run the program: https://drive.google.com/file/d/11dKuACP-K8PWYrQX-2br3Xj83ryQKbt4/view?usp=sharing
- `iDECOR.pdf` is a powerpoint which illustrates the complete framework of this project. 
- `load_similarity.py` (TBC)

## Table of Contents
- Data Collection
- Data Preprocessing
- System Architecture
- Modeling
  - Object Detection: Detectron2 & Faster R-CNN
  - Similarity Detection: VGG16 vs IncepctionV3
- Evaluation
- Deployment
  - Streamlit
- References

## Data Collection

01. Annotated Furniture Images
	- Used in **Object Detection Model** training
	- Data Source: [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)
	- Obtained around 10,000 images across 6 categories
	- Images annotated with image-level labels, object bounding boxes

02. Style-labelled Furniture Images
	- Used in **Similarity Detection Model** training
	- Data Source: [Bonn Furniture Style Dataset](https://cvml.comp.nus.edu.sg/furniture/index.html)
	- Obtained around 13,000 images across 6 categories and 15 styles
	- Style labelled by interior design professionals

03. IKEA Product Catalogue Dataset
	- Used in **Recommendation**
	- Data Source: [IKEA Hong Kong](https://www.ikea.com.hk/en)
	- Web-scraped around 1,400 products in any category

## Data Preprocessing
- Balancing Dataset
- Detectron2 Formatting with Annotations
- Mutli-label Binarizers
- Image resizing
- Drop duplicated images
- Image Augmentation

### System Architecture
<img src="https://github.com/sophiachann/ObjectDetectionProject-FurnitureRecommender/blob/main/images/sys-architecture2.png" width="800"/>

This flowchart shows five main stages during the whole development of our product, with different focus, tools and technologies we have used within which we will explain further.

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


#### Prerequisites 
- CPU
- GPU x 1

#### Installation
- Detectron2 Library
	- Built on PyTorch
	- NVIDIA GPU required
- Similarity Detection


#### References:
1. Bonn Furniture Style Dataset: https://cvml.comp.nus.edu.sg/furniture/index.html
2. Open Images Dataset V6: https://storage.googleapis.com/openimages/web/index.html
3. IKEA site: https://www.ikea.com.hk/en

