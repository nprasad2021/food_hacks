# food_hacks
Computer Vision and Grocery Stores: A Deep Learning Methodology to Solve World Hunger

* **T-1 Best Hack at SAP iXP Hackathon**
* **"Most Impact" Hack**

Here, we use computer vision techniques to assess food quality, and dynamically assess price
from a number of sources of data, including color, food type, growth stage, season, and location.

## How to Run
Clone this repository

```
git clone git@github.com:nprasad2021/food_hacks.git
```

Create file in format of 'fruit_image_index.csv' with your desired search queries in repo
Scrape images from Google

```
python scraper.py
```

Follow instructions on interactive console to download (100) images per search query indicated in your csv file.
Either restart (r) or continue (c), based on your progress.

To train the model, run:

```
python main.py $VAR
```
where $VAR is an integer between 0 and 11 indicated type of net and dataset to use.

To open the web-app, run:

```
cd keras-flask-webapp
python app.py
```

Navigate to the web-page, and play around!

## Image Scraping

There are no datasets available on Kaggle or on the web that fit our specification:
map food item to quality, color, growth stage, etc. Therefore, we scrape images from google, 
designing a robust set of text queries based on current research that allow us to scrape the
best results.

## Deep Learning

We design, train, and test neural networks with TensorFlow and its high-end framework, Keras. 
We test several classical networks, augmented with several extra layers to account for class imbalance.
We use weights from imagenet to speed up training. A custom learning rate scheduler (SGD) is applied. To account
for the large amount of training data, the input pipeline is optimized for maximum performance. CPU and GPU work together
to streamline training.

Validation Accuracies were achieved for the following categories with Deep Neural Nets:
* Color: 70%
* Food: 96%
* Quality: 70%

Best results were achieved by building on top of Inception and VGG models.

## Live Model Demonstration

Build and deployed machine learning models with Flask Backend. 

## User Interface

Built demo for live user interface, showing interactive toolbar, and dynamic, relevant content.

Contact @nprasad2021 for further details of deep learning models built, as well as image scraping algorithm.