# freshness-detector
Detect freshness of fruits

The model has been trained on this [kaggle dataset](https://www.kaggle.com/sriramr/fruits-fresh-and-rotten-for-classification). 
The project has been hosted on [heroku](https://freshness-detector.herokuapp.com/).

To get the best predictions use the test images from the kaggle dataset. In case you want to test it on other images then make sure you tick these all conditions:-
- The image is .jpeg or .png.
- The image is of a Apple, Banana or Orange.
- The image has a white background.

These many constraints are because of the limited training data and the model is as good as the data on which it is trained.
