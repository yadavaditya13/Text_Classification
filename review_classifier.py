"""

This script will be used to classify reviews given by the user via command line argument as positive or negative using
our Trained Classifier Model!!!

"""

# importing required packages

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

import pyttsx3 as tts
import pandas as pd
import numpy as np


# let's create a function for TTS
def text_to_speech(text):
    # initializing the Engine
    print("\n[INFO] Initializing the Engine...")
    engine = tts.init()

    # selecting the female voice
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)

    # setting a decent rate of speech and volume
    engine.setProperty("rate", 115)
    engine.setProperty("volume", 1.0)

    # let the engine speak and then wait
    print("\n[INFO] Engine begins Speaking...")
    engine.say(text)
    #print("\n[INFO] Engine stops Speaking...")
    engine.runAndWait()


# let's load the model
print("\n[INFO] Loading model from disk...")
model = load_model("classifierModel.h5")

# for the model to perform well we will need the review data-set in token format on which we trained it
# let's load the data-set and pre-process it and make it ready for use
df = pd.read_csv("IMDB Dataset.csv")

# Initializing empty lists for reviews
reviews = []

# iterating through the entire data-set one-by-one
for index in df.index:
    # grabbing reviews
    review = df.loc[index, 'review'].strip()
    # appending these value to respective list
    reviews.append(review)

# Initializing Tokenizer
print("\n[INFO] Initializing Tokenizer...")
tokenizer = Tokenizer(num_words=10000, oov_token=None)
tokenizer.fit_on_texts(reviews)

# declaring the classes
label = {1: "positive", 0: "negative"}

# let's run a loop for grabbing reviews
while True:

    # let's ask for review
    textInit = "Hello There, Kindly enter the review."
    text_to_speech(textInit)

    textReview = input("\n[INFO] Enter Review here: ")
    textContent = "The review reads as follows", textReview, "Please wait for a moment."
    # passing it to engine
    text_to_speech(textContent)

    # let's begin pre-processing and passing the input to model
    textReview = textReview.strip()         # optional steps but for better performance we should pre-process it
    tokenizer.fit_on_texts(textReview)      # optional steps but for better performance we should pre-process it
    sequence = tokenizer.texts_to_sequences([textReview])

    # add padding after converting to numpy array
    sequence = np.array(sequence)           # optional steps but for better performance we should pre-process it
    sequence = pad_sequences(sequence, maxlen=300)

    # let's begin predictions
    prediction = model.predict(sequence)[0]
    print("\n[INFO] Prediction Vector: ", prediction)
    predictionIdx = np.argmax(prediction)
    print("\n[INFO] Prediction Index: ", predictionIdx)

    review = label[predictionIdx]
    print("\n[INFO] FeedBack is: {}".format(review))

    textFeedBack = "The Review is ", review
    accuracy = "Accuracy of prediction is {:.2f} %".format(prediction[predictionIdx] * 100)
    finalResult = textFeedBack, "and", accuracy
    text_to_speech(finalResult)

    # ask if the user want's to continue or exit
    text_to_speech("Kindly type yes is you wish to continue and no to exit")

    if input("\nType 'yes' is you wish to continue and 'no' to exit:") == "no":
        text_to_speech("Good Bye")
        break