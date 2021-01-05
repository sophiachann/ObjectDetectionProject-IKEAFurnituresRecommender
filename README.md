# iDECOR - Furniture Recommender

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
