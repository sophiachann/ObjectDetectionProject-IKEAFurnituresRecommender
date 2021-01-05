# iDECOR - Furniture Recommender

<p>This repository contains all the codes to complete iDECOR, our Furniture Recommender that help users who have recently moved to explore IKEA products at ease.</p>

#### What does iDECOR do?
<p>After uploading a room scene image, IDECOR returns users with similar-styled furnitures in favour from IKEA dataset.</p>

#### What is in this repo?

<ul>
<li>preprocessing.py contains the preprocessing functions for turning Open Images images & labels into Detectron2 style.</li>
<li>downloadOI.py is a slightly modified downloader script from LearnOpenCV which downloads only certain classes of images from Open Images, example:</li>
</ul>

<code># Download all Coffeemaker images for train & test & validation
!python3 downloadOI.py --classes 'Bed, Cabinetry, Chair, Couch, Lamp, Table' --dataset train, validation, test</code>
