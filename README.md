# Display Detector

Graduation Work Project by Paulo de Oliveira Guedes, from the Electronic Engineering course, at the Federal University of Pernambuco (UFPE).

This work creates a screen detector for the brazilian electronic voting machine, using AlexNet, Resnet50 and a modified AlexNet called Displaynet to predict the 4 corners of the screen:

![](https://i.imgur.com/nN97rzN.png)

![](https://i.imgur.com/CBb51dJ.png)


# Prerequisites

```
$ pip install -r requirements.txt
```

# How to label

To label make sure you have images in ascending order (0.jpg, 1.jpg, ... , n.jpg).

In "data manipulation" folder use a csv with the fallowing colunms name;x1;y1;x2;y2;x3;y3;x4;y4, put the dataset directory to be labeling and the number of the fisrt image to be read. Then run:

```
$ python labeling.py
```

The fallowing commands can be used:

*    mouse_click = put point
*    f = fullscreen
*    e = erase point
*    backspace = go to previus image
*    space = go to next image (if have 4 points in the screen)
*    t = fine tuning
*    w a s d = fine tuning controll
*    esq = quit

# How to train

The training code ```train.py``` uses Neptune as a experiment tracking tool, to better understanging read the [basics](https://docs.neptune.ai/getting-started/installation).

To train a network use one of the networks provided, or one of your interest, commenting as needed, as explained in the code, then run:

```
$ python train.py
```


# How to use a trained net

The code ```load.py```[](https://) takes a trained network ("model"), a folder with photos (in "test_data") and predicts the points in the photos, saving the images with the points. It is necessary to change the "resize_vector" depending on the chosen network

Change the quoted variables as needed and run:

```
$ python load.py
```

Link to [trained nets](https://drive.google.com/drive/folders/1F4uiQCDq5op0oz3OQVBEJeWcxTUkXfHe?usp=sharing).

# Results on Neptune and Monography

Link to my results on [Neptune](https://app.neptune.ai/pog/TCC/experiments?split=tbl&dash=charts&viewId=standard-view).

Link to my [Monography](https://drive.google.com/drive/folders/1i5zsXuNKxO4-fdKlH1H2Cz_U20uploOW?usp=sharing) in portuguese.