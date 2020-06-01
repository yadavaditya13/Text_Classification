# Text_Classification

Text_classification is one of the tasks you could perform under the NLP (Natural Language Processing) section of AI.
Now the project I made is a Movie_Review_Classifier which uses RNN model with LSTM network to learn and later classify the texts as positive or negative.

Also I used pyttsx3 for Text-To-Speech operation in the project just to make it a little bit more interesting.

We have an "*.ipynb" file which loads dataset and preprocesses it and the model is created there and saved in the disk at last.
I wasn't able to upload the model here because I am having LFS issues for this repository, so if you wish to use this project or run it... I would ask to first setup a local feasible environment for tensorflow 2.1 with cuda 10.1 else run this on google colab.

You will get your trained model weights as output which will be loaded in file "review_classifier.py" and you will be able to use it on new reviews provided by you.

I wasn't able to upload few large files here for the project but I will provide the link so that you can directly download it from source.

Dataset Link If needed : https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews          

Global Vector link (GloVe) : http://nlp.stanford.edu/data/glove.6B.zip
           
Drive link to my model if needed : https://drive.google.com/file/d/1SXxt8u29o3lbC7d1Bx_eVlKDnfXke39Z/view?usp=sharing
       
Note: You need to download GloVe word vectors and store it in the "data" folder of the repository / project directory if not present create one.
