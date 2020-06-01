{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "# importing required packages\n",
    "\n",
    "from keras.layers import Dense, Dropout, LSTM, Embedding\n",
    "from keras.preprocessing.sequence import pad_sequences\n",
    "from keras.preprocessing.text import Tokenizer\n",
    "from keras.callbacks import EarlyStopping\n",
    "from keras.utils import to_categorical\n",
    "from keras.models import Sequential\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report\n",
    "from matplotlib import pyplot as plt\n",
    "from tqdm import tqdm\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] Dataset Rows: 50000\n",
      "[INFO] Dataset Columns: 2\n"
     ]
    }
   ],
   "source": [
    "# loading the IMDB-Review dataset\n",
    "df = pd.read_csv(\"IMDB Dataset.csv\")\n",
    "\n",
    "# let's have a look at the shape of the given dataset\n",
    "print(\"[INFO] Dataset Rows: {}\\n[INFO] Dataset Columns: {}\".format(df.shape[0], df.shape[1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>Basically there's a family where a little boy ...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Petter Mattei's \"Love in the Time of Money\" is...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>Probably my all-time favorite movie, a story o...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>I sure would like to see a resurrection of a u...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>This show was an amazing, fresh &amp; innovative i...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>Encouraged by the positive comments about this...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>If you like original gut wrenching laughter yo...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  One of the other reviewers has mentioned that ...  positive\n",
       "1  A wonderful little production. <br /><br />The...  positive\n",
       "2  I thought this was a wonderful way to spend ti...  positive\n",
       "3  Basically there's a family where a little boy ...  negative\n",
       "4  Petter Mattei's \"Love in the Time of Money\" is...  positive\n",
       "5  Probably my all-time favorite movie, a story o...  positive\n",
       "6  I sure would like to see a resurrection of a u...  positive\n",
       "7  This show was an amazing, fresh & innovative i...  negative\n",
       "8  Encouraged by the positive comments about this...  negative\n",
       "9  If you like original gut wrenching laughter yo...  positive"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# let's have a look at the data\n",
    "df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] Let's have a look at first 5 reviews:\n",
      "\n",
      "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side. \n",
      "\n",
      "A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />The actors are extremely well chosen- Michael Sheen not only \"has got all the polari\" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great master's of comedy and his life. <br /><br />The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional 'dream' techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets (particularly of their flat with Halliwell's murals decorating every surface) are terribly well done. \n",
      "\n",
      "I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching a light-hearted comedy. The plot is simplistic, but the dialogue is witty and the characters are likable (even the well bread suspected serial killer). While some may be disappointed when they realize this is not Match Point 2: Risk Addiction, I thought it was proof that Woody Allen is still fully in control of the style many of us have grown to love.<br /><br />This was the most I'd laughed at one of Woody's comedies in years (dare I say a decade?). While I've never been impressed with Scarlet Johanson, in this she managed to tone down her \"sexy\" image and jumped right into a average, but spirited young woman.<br /><br />This may not be the crown jewel of his career, but it was wittier than \"Devil Wears Prada\" and more interesting than \"Superman\" a great comedy to go see with friends. \n",
      "\n",
      "Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them. \n",
      "\n",
      "Petter Mattei's \"Love in the Time of Money\" is a visually stunning film to watch. Mr. Mattei offers us a vivid portrait about human relations. This is a movie that seems to be telling us what money, power and success do to people in the different situations we encounter. <br /><br />This being a variation on the Arthur Schnitzler's play about the same theme, the director transfers the action to the present time New York where all these different characters meet and connect. Each one is connected in one way, or another to the next person, but no one seems to know the previous point of contact. Stylishly, the film has a sophisticated luxurious look. We are taken to see how these people live and the world they live in their own habitat.<br /><br />The only thing one gets out of all these souls in the picture is the different stages of loneliness each one inhabits. A big city is not exactly the best place in which human relations find sincere fulfillment, as one discerns is the case with most of the people we encounter.<br /><br />The acting is good under Mr. Mattei's direction. Steve Buscemi, Rosario Dawson, Carol Kane, Michael Imperioli, Adrian Grenier, and the rest of the talented cast, make these characters come alive.<br /><br />We wish Mr. Mattei good luck and await anxiously for his next work. \n",
      "\n"
     ]
    }
   ],
   "source": [
    "# let's have a look at top 5 of the reviews\n",
    "\n",
    "print(\"[INFO] Let's have a look at first 5 reviews:\\n\")\n",
    "for i in range(5):\n",
    "    print(df['review'][i],\"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# let's check if some values are missing\n",
    "df.isnull().values.any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>Basically there's a family where a little boy ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>Petter Mattei's \"Love in the Time of Money\" is...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>Probably my all-time favorite movie, a story o...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>I sure would like to see a resurrection of a u...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>This show was an amazing, fresh &amp; innovative i...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>Encouraged by the positive comments about this...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>If you like original gut wrenching laughter yo...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  sentiment\n",
       "0  One of the other reviewers has mentioned that ...          1\n",
       "1  A wonderful little production. <br /><br />The...          1\n",
       "2  I thought this was a wonderful way to spend ti...          1\n",
       "3  Basically there's a family where a little boy ...          0\n",
       "4  Petter Mattei's \"Love in the Time of Money\" is...          1\n",
       "5  Probably my all-time favorite movie, a story o...          1\n",
       "6  I sure would like to see a resurrection of a u...          1\n",
       "7  This show was an amazing, fresh & innovative i...          0\n",
       "8  Encouraged by the positive comments about this...          0\n",
       "9  If you like original gut wrenching laughter yo...          1"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# let's begin pre-processing the dataset\n",
    "# we will firstly replace the setiment column with number 1 or 0\n",
    "# initializing a new dataframe and replacing sentiments\n",
    "\n",
    "dfX = df\n",
    "dfX.replace('positive', 1, inplace=True)\n",
    "dfX.replace('negative', 0, inplace=True)\n",
    "\n",
    "# let's have a look\n",
    "dfX.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's try to plot some graphs\n",
    "\n",
    "plt.figure()\n",
    "\n",
    "plt.hist(dfX['sentiment'])\n",
    "plt.title(\"Movie Review\")\n",
    "plt.xlabel(\"Sentiment\")\n",
    "plt.ylabel(\"Number of Votes\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# defining parameters\n",
    "\n",
    "numWords = 10000\n",
    "sequenceLength = 300\n",
    "testSize = 0.4\n",
    "oovToken = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████████████████████████████████| 50000/50000 [00:00<00:00, 57074.96it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[INFO] Reviews # : 50000\n",
      "[INFO] Sentiments # : 50000\n"
     ]
    }
   ],
   "source": [
    "# let's begin pre-processing and loading the dataset\n",
    "\n",
    "# Initializing empty lists for reviews and labels\n",
    "reviews = []\n",
    "labels = []\n",
    "\n",
    "# iterating through the entire dataset one-by-one\n",
    "for index in tqdm(dfX.index):\n",
    "    \n",
    "    # grabbing reviews and labels\n",
    "    review = dfX.loc[index, 'review'].strip()\n",
    "    label = dfX.loc[index, 'sentiment']\n",
    "    \n",
    "    # appending these value to respective lists\n",
    "    reviews.append(review)\n",
    "    labels.append(label)\n",
    "\n",
    "# let's print the length for validation\n",
    "print(\"\\n[INFO] Reviews # : {}\\n[INFO] Sentiments # : {}\".format(len(reviews), len(labels)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] trainX: 30000 :: trainY: 30000\n",
      "[INFO] testX: 10000 :: testY: 10000\n",
      "[INFO] valX: 10000 :: valY: 10000\n"
     ]
    }
   ],
   "source": [
    "# now let's tokenize the reviews, delete uncommon words such as names, etc.\n",
    "\n",
    "# initializing the Tokenizer\n",
    "tokenizer = Tokenizer(num_words=numWords, oov_token=oovToken)\n",
    "tokenizer.fit_on_texts(reviews)\n",
    "\n",
    "# creating a sequence\n",
    "X = tokenizer.texts_to_sequences(reviews)\n",
    "\n",
    "# converting it to numpy arrays\n",
    "X, y = np.array(X), np.array(labels)\n",
    "\n",
    "# applying padding to X to make sure the length remains consistent\n",
    "X = pad_sequences(X, maxlen=sequenceLength)\n",
    "# converting labels to one-hot-encodings\n",
    "y = to_categorical(y)\n",
    "\n",
    "# splitting the dataset for training and testing\n",
    "trainX, dataX, trainY, dataY = train_test_split(X, y, test_size=testSize, random_state=2)\n",
    "testX, valX, testY, valY = train_test_split(dataX, dataY, test_size=0.5, random_state=2)\n",
    "\n",
    "print(\"[INFO] trainX: {} :: trainY: {}\\n[INFO] testX: {} :: testY: {}\\n[INFO] valX: {} :: valY: {}\".format(len(trainX), len(trainY), len(testX), len(testY), len(valX), len(valY)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<keras_preprocessing.text.Tokenizer object at 0x00000248EFAFF108>\n"
     ]
    }
   ],
   "source": [
    "# let's create a dictionary and store our data as follows\n",
    "data = {}\n",
    "\n",
    "# appending values\n",
    "data[\"trainX\"] = trainX\n",
    "data[\"testX\"] = testX\n",
    "data[\"valX\"] = valX\n",
    "data[\"trainY\"] = trainY\n",
    "data[\"testY\"] = testY\n",
    "data[\"valY\"] = valY\n",
    "data[\"tokenizer\"] = tokenizer\n",
    "data[\"int2label\"] = {0:\"negative\", 1:\"positive\"}\n",
    "data[\"label2int\"] = {\"negative\":0, \"positive\":1}\n",
    "\n",
    "print(data[\"tokenizer\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# defining some parameters for model creation\n",
    "\n",
    "embeddingSize = 300\n",
    "units = 32\n",
    "dropout = 0.4\n",
    "loss = \"categorical_crossentropy\"\n",
    "optimizer = \"adam\"\n",
    "batchSize = 300\n",
    "epochs = 30\n",
    "wordIndex = data[\"tokenizer\"].word_index\n",
    "modelName = \"classifierModel.h5\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Reading GloVe: 400000it [00:15, 25236.32it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " [[ 0.          0.          0.         ...  0.          0.\n",
      "   0.        ]\n",
      " [ 0.04656     0.21318001 -0.0074364  ...  0.0090611  -0.20988999\n",
      "   0.053913  ]\n",
      " [ 0.038466   -0.039792    0.082747   ... -0.33427     0.011807\n",
      "   0.059703  ]\n",
      " ...\n",
      " [ 0.0566     -0.29016     0.47273001 ... -0.1166      0.21393\n",
      "  -0.61741   ]\n",
      " [ 0.          0.          0.         ...  0.          0.\n",
      "   0.        ]\n",
      " [-0.14168     0.35080999  0.44791001 ...  0.14529    -0.013969\n",
      "   0.40158999]]\n"
     ]
    }
   ],
   "source": [
    "# loading the pre-trained word vectors i.e. GloVe\n",
    "\n",
    "embeddingMatrix = np.zeros((len(wordIndex) + 1, embeddingSize))\n",
    "# opening the gloveFile\n",
    "with open(f\"data/glove.6B.{embeddingSize}d.txt\", encoding=\"utf8\") as f:\n",
    "    \n",
    "    for line in tqdm(f, \"Reading GloVe\"):\n",
    "        values = line.split()\n",
    "        \n",
    "        # get the word as the first word in the line\n",
    "        word = values[0]\n",
    "        \n",
    "        # see if the word is found in the wordIndex\n",
    "        if word in wordIndex:\n",
    "            # grab it's index\n",
    "            idx = wordIndex[word]\n",
    "            \n",
    "            # get the vectors as the remaining values in the line\n",
    "            embeddingMatrix[idx] = np.array(values[1:], dtype=\"float32\")\n",
    "\n",
    "# let's have a look at embedding matrix\n",
    "print(\"\\n\", embeddingMatrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# let's construct the model\n",
    "\n",
    "def Classifier(wordIndex, units, embeddingSize, sequenceLength, dropout, loss, optimizer, outputLength):\n",
    "    \n",
    "    # initializing the network\n",
    "    model = Sequential()\n",
    "\n",
    "    # adding layers to our model\n",
    "\n",
    "    # first embedding layer\n",
    "    model.add(Embedding(len(wordIndex) + 1, embeddingSize, weights=[embeddingMatrix],\n",
    "                        trainable=False, input_length=sequenceLength))\n",
    "\n",
    "    # adding LSTM (Hidden) layers to our model\n",
    "\n",
    "    # first hidden layer\n",
    "    model.add(LSTM(2*2*units, return_sequences=True))\n",
    "    model.add(Dropout(rate=dropout))\n",
    "\n",
    "    # second hidden layer\n",
    "    model.add(LSTM(2*units, return_sequences=True))\n",
    "    model.add(Dropout(rate=dropout))\n",
    "\n",
    "    # third hidden layer\n",
    "    model.add(LSTM(units, return_sequences=False))\n",
    "    model.add(Dropout(rate=dropout))\n",
    "\n",
    "    # softmax classifier\n",
    "    model.add(Dense(outputLength, activation=\"softmax\"))\n",
    "    \n",
    "    # let's return the model\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] Initializing the Model...\n",
      "[INFO] Compiling the Model...\n"
     ]
    }
   ],
   "source": [
    "# Initializing the model\n",
    "\n",
    "print(\"[INFO] Initializing the Model...\")\n",
    "model = Classifier(wordIndex, units, embeddingSize, \n",
    "                   sequenceLength, dropout, loss, optimizer, outputLength=2)\n",
    "\n",
    "# compiling the model\n",
    "print(\"[INFO] Compiling the Model...\")\n",
    "model.compile(optimizer=optimizer, loss=loss, metrics=[\"accuracy\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "embedding_1 (Embedding)      (None, 300, 300)          37267200  \n",
      "_________________________________________________________________\n",
      "lstm_1 (LSTM)                (None, 300, 128)          219648    \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 300, 128)          0         \n",
      "_________________________________________________________________\n",
      "lstm_2 (LSTM)                (None, 300, 64)           49408     \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 300, 64)           0         \n",
      "_________________________________________________________________\n",
      "lstm_3 (LSTM)                (None, 32)                12416     \n",
      "_________________________________________________________________\n",
      "dropout_3 (Dropout)          (None, 32)                0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 2)                 66        \n",
      "=================================================================\n",
      "Total params: 37,548,738\n",
      "Trainable params: 281,538\n",
      "Non-trainable params: 37,267,200\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# let's look at model summary\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# let's define EarlyStopping\n",
    "ES = EarlyStopping(monitor=\"val_loss\", patience=7, mode=\"min\", restore_best_weights=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 30000 samples, validate on 10000 samples\n",
      "Epoch 1/30\n",
      "30000/30000 [==============================] - ETA: 4:32 - loss: 0.7077 - accuracy: 0.46 - ETA: 2:56 - loss: 0.7053 - accuracy: 0.47 - ETA: 2:24 - loss: 0.6997 - accuracy: 0.48 - ETA: 2:07 - loss: 0.6999 - accuracy: 0.48 - ETA: 1:57 - loss: 0.6961 - accuracy: 0.49 - ETA: 1:50 - loss: 0.6945 - accuracy: 0.50 - ETA: 1:44 - loss: 0.6933 - accuracy: 0.51 - ETA: 1:39 - loss: 0.6911 - accuracy: 0.52 - ETA: 1:36 - loss: 0.6889 - accuracy: 0.52 - ETA: 1:33 - loss: 0.6859 - accuracy: 0.53 - ETA: 1:30 - loss: 0.6818 - accuracy: 0.54 - ETA: 1:28 - loss: 0.6785 - accuracy: 0.55 - ETA: 1:25 - loss: 0.6759 - accuracy: 0.55 - ETA: 1:23 - loss: 0.6715 - accuracy: 0.56 - ETA: 1:21 - loss: 0.6699 - accuracy: 0.57 - ETA: 1:19 - loss: 0.6668 - accuracy: 0.57 - ETA: 1:18 - loss: 0.6637 - accuracy: 0.58 - ETA: 1:16 - loss: 0.6622 - accuracy: 0.58 - ETA: 1:15 - loss: 0.6588 - accuracy: 0.59 - ETA: 1:13 - loss: 0.6542 - accuracy: 0.59 - ETA: 1:12 - loss: 0.6513 - accuracy: 0.60 - ETA: 1:10 - loss: 0.6488 - accuracy: 0.60 - ETA: 1:09 - loss: 0.6489 - accuracy: 0.60 - ETA: 1:08 - loss: 0.6432 - accuracy: 0.61 - ETA: 1:07 - loss: 0.6422 - accuracy: 0.61 - ETA: 1:06 - loss: 0.6413 - accuracy: 0.61 - ETA: 1:04 - loss: 0.6407 - accuracy: 0.62 - ETA: 1:03 - loss: 0.6372 - accuracy: 0.62 - ETA: 1:02 - loss: 0.6334 - accuracy: 0.63 - ETA: 1:01 - loss: 0.6305 - accuracy: 0.63 - ETA: 1:00 - loss: 0.6282 - accuracy: 0.63 - ETA: 59s - loss: 0.6242 - accuracy: 0.6396 - ETA: 57s - loss: 0.6224 - accuracy: 0.641 - ETA: 57s - loss: 0.6199 - accuracy: 0.644 - ETA: 56s - loss: 0.6162 - accuracy: 0.648 - ETA: 55s - loss: 0.6140 - accuracy: 0.651 - ETA: 54s - loss: 0.6102 - accuracy: 0.655 - ETA: 53s - loss: 0.6100 - accuracy: 0.655 - ETA: 52s - loss: 0.6086 - accuracy: 0.657 - ETA: 51s - loss: 0.6070 - accuracy: 0.659 - ETA: 50s - loss: 0.6059 - accuracy: 0.661 - ETA: 49s - loss: 0.6036 - accuracy: 0.662 - ETA: 48s - loss: 0.6018 - accuracy: 0.664 - ETA: 47s - loss: 0.5996 - accuracy: 0.666 - ETA: 46s - loss: 0.5986 - accuracy: 0.667 - ETA: 46s - loss: 0.5965 - accuracy: 0.669 - ETA: 45s - loss: 0.5941 - accuracy: 0.671 - ETA: 44s - loss: 0.5929 - accuracy: 0.673 - ETA: 43s - loss: 0.5927 - accuracy: 0.674 - ETA: 42s - loss: 0.5909 - accuracy: 0.676 - ETA: 41s - loss: 0.5897 - accuracy: 0.677 - ETA: 40s - loss: 0.5880 - accuracy: 0.679 - ETA: 40s - loss: 0.5859 - accuracy: 0.681 - ETA: 39s - loss: 0.5852 - accuracy: 0.682 - ETA: 38s - loss: 0.5845 - accuracy: 0.683 - ETA: 37s - loss: 0.5818 - accuracy: 0.686 - ETA: 36s - loss: 0.5791 - accuracy: 0.688 - ETA: 35s - loss: 0.5786 - accuracy: 0.689 - ETA: 34s - loss: 0.5771 - accuracy: 0.690 - ETA: 33s - loss: 0.5759 - accuracy: 0.691 - ETA: 32s - loss: 0.5738 - accuracy: 0.693 - ETA: 32s - loss: 0.5720 - accuracy: 0.695 - ETA: 31s - loss: 0.5700 - accuracy: 0.696 - ETA: 30s - loss: 0.5681 - accuracy: 0.698 - ETA: 29s - loss: 0.5661 - accuracy: 0.700 - ETA: 28s - loss: 0.5640 - accuracy: 0.701 - ETA: 27s - loss: 0.5624 - accuracy: 0.703 - ETA: 26s - loss: 0.5610 - accuracy: 0.705 - ETA: 25s - loss: 0.5592 - accuracy: 0.706 - ETA: 25s - loss: 0.5584 - accuracy: 0.706 - ETA: 24s - loss: 0.5575 - accuracy: 0.707 - ETA: 23s - loss: 0.5573 - accuracy: 0.707 - ETA: 22s - loss: 0.5575 - accuracy: 0.707 - ETA: 21s - loss: 0.5572 - accuracy: 0.708 - ETA: 20s - loss: 0.5563 - accuracy: 0.708 - ETA: 20s - loss: 0.5553 - accuracy: 0.710 - ETA: 19s - loss: 0.5544 - accuracy: 0.710 - ETA: 18s - loss: 0.5530 - accuracy: 0.711 - ETA: 17s - loss: 0.5514 - accuracy: 0.713 - ETA: 16s - loss: 0.5491 - accuracy: 0.714 - ETA: 15s - loss: 0.5474 - accuracy: 0.716 - ETA: 14s - loss: 0.5457 - accuracy: 0.717 - ETA: 14s - loss: 0.5452 - accuracy: 0.718 - ETA: 13s - loss: 0.5444 - accuracy: 0.719 - ETA: 12s - loss: 0.5430 - accuracy: 0.720 - ETA: 11s - loss: 0.5419 - accuracy: 0.721 - ETA: 10s - loss: 0.5410 - accuracy: 0.722 - ETA: 9s - loss: 0.5396 - accuracy: 0.723 - ETA: 9s - loss: 0.5383 - accuracy: 0.72 - ETA: 8s - loss: 0.5379 - accuracy: 0.72 - ETA: 7s - loss: 0.5362 - accuracy: 0.72 - ETA: 6s - loss: 0.5352 - accuracy: 0.72 - ETA: 5s - loss: 0.5338 - accuracy: 0.72 - ETA: 4s - loss: 0.5332 - accuracy: 0.72 - ETA: 4s - loss: 0.5329 - accuracy: 0.72 - ETA: 3s - loss: 0.5319 - accuracy: 0.73 - ETA: 2s - loss: 0.5307 - accuracy: 0.73 - ETA: 1s - loss: 0.5305 - accuracy: 0.73 - ETA: 0s - loss: 0.5291 - accuracy: 0.73 - 88s 3ms/step - loss: 0.5282 - accuracy: 0.7337 - val_loss: 0.5010 - val_accuracy: 0.7509\n",
      "Epoch 2/30\n",
      "30000/30000 [==============================] - ETA: 1:31 - loss: 0.4608 - accuracy: 0.77 - ETA: 1:22 - loss: 0.4965 - accuracy: 0.76 - ETA: 1:19 - loss: 0.4761 - accuracy: 0.78 - ETA: 1:18 - loss: 0.4822 - accuracy: 0.77 - ETA: 1:18 - loss: 0.4779 - accuracy: 0.78 - ETA: 1:17 - loss: 0.4751 - accuracy: 0.78 - ETA: 1:15 - loss: 0.4728 - accuracy: 0.78 - ETA: 1:14 - loss: 0.4624 - accuracy: 0.79 - ETA: 1:12 - loss: 0.4545 - accuracy: 0.79 - ETA: 1:12 - loss: 0.4567 - accuracy: 0.79 - ETA: 1:11 - loss: 0.4571 - accuracy: 0.79 - ETA: 1:10 - loss: 0.4487 - accuracy: 0.79 - ETA: 1:09 - loss: 0.4412 - accuracy: 0.80 - ETA: 1:08 - loss: 0.4410 - accuracy: 0.80 - ETA: 1:07 - loss: 0.4414 - accuracy: 0.80 - ETA: 1:06 - loss: 0.4370 - accuracy: 0.80 - ETA: 1:05 - loss: 0.4350 - accuracy: 0.80 - ETA: 1:04 - loss: 0.4380 - accuracy: 0.80 - ETA: 1:04 - loss: 0.4372 - accuracy: 0.80 - ETA: 1:03 - loss: 0.4374 - accuracy: 0.80 - ETA: 1:02 - loss: 0.4384 - accuracy: 0.80 - ETA: 1:01 - loss: 0.4377 - accuracy: 0.80 - ETA: 1:00 - loss: 0.4360 - accuracy: 0.80 - ETA: 1:00 - loss: 0.4378 - accuracy: 0.80 - ETA: 59s - loss: 0.4371 - accuracy: 0.8047 - ETA: 58s - loss: 0.4355 - accuracy: 0.805 - ETA: 57s - loss: 0.4329 - accuracy: 0.807 - ETA: 56s - loss: 0.4326 - accuracy: 0.807 - ETA: 55s - loss: 0.4324 - accuracy: 0.806 - ETA: 55s - loss: 0.4312 - accuracy: 0.807 - ETA: 54s - loss: 0.4316 - accuracy: 0.807 - ETA: 53s - loss: 0.4315 - accuracy: 0.807 - ETA: 52s - loss: 0.4305 - accuracy: 0.807 - ETA: 51s - loss: 0.4308 - accuracy: 0.806 - ETA: 51s - loss: 0.4304 - accuracy: 0.806 - ETA: 50s - loss: 0.4300 - accuracy: 0.806 - ETA: 49s - loss: 0.4291 - accuracy: 0.807 - ETA: 48s - loss: 0.4279 - accuracy: 0.808 - ETA: 47s - loss: 0.4273 - accuracy: 0.809 - ETA: 47s - loss: 0.4291 - accuracy: 0.807 - ETA: 46s - loss: 0.4275 - accuracy: 0.808 - ETA: 45s - loss: 0.4261 - accuracy: 0.809 - ETA: 44s - loss: 0.4264 - accuracy: 0.809 - ETA: 43s - loss: 0.4257 - accuracy: 0.810 - ETA: 43s - loss: 0.4260 - accuracy: 0.809 - ETA: 42s - loss: 0.4269 - accuracy: 0.808 - ETA: 41s - loss: 0.4270 - accuracy: 0.808 - ETA: 40s - loss: 0.4278 - accuracy: 0.808 - ETA: 39s - loss: 0.4272 - accuracy: 0.808 - ETA: 39s - loss: 0.4262 - accuracy: 0.808 - ETA: 38s - loss: 0.4246 - accuracy: 0.809 - ETA: 37s - loss: 0.4240 - accuracy: 0.809 - ETA: 36s - loss: 0.4235 - accuracy: 0.810 - ETA: 36s - loss: 0.4224 - accuracy: 0.810 - ETA: 35s - loss: 0.4211 - accuracy: 0.812 - ETA: 34s - loss: 0.4212 - accuracy: 0.812 - ETA: 33s - loss: 0.4204 - accuracy: 0.812 - ETA: 32s - loss: 0.4203 - accuracy: 0.812 - ETA: 32s - loss: 0.4192 - accuracy: 0.813 - ETA: 31s - loss: 0.4181 - accuracy: 0.813 - ETA: 30s - loss: 0.4185 - accuracy: 0.813 - ETA: 29s - loss: 0.4184 - accuracy: 0.813 - ETA: 29s - loss: 0.4180 - accuracy: 0.813 - ETA: 28s - loss: 0.4179 - accuracy: 0.813 - ETA: 27s - loss: 0.4175 - accuracy: 0.814 - ETA: 26s - loss: 0.4171 - accuracy: 0.814 - ETA: 25s - loss: 0.4166 - accuracy: 0.814 - ETA: 25s - loss: 0.4157 - accuracy: 0.815 - ETA: 24s - loss: 0.4150 - accuracy: 0.815 - ETA: 23s - loss: 0.4153 - accuracy: 0.815 - ETA: 22s - loss: 0.4149 - accuracy: 0.815 - ETA: 21s - loss: 0.4141 - accuracy: 0.815 - ETA: 21s - loss: 0.4135 - accuracy: 0.816 - ETA: 20s - loss: 0.4132 - accuracy: 0.816 - ETA: 19s - loss: 0.4125 - accuracy: 0.816 - ETA: 18s - loss: 0.4122 - accuracy: 0.817 - ETA: 18s - loss: 0.4117 - accuracy: 0.817 - ETA: 17s - loss: 0.4109 - accuracy: 0.818 - ETA: 16s - loss: 0.4104 - accuracy: 0.818 - ETA: 15s - loss: 0.4101 - accuracy: 0.818 - ETA: 14s - loss: 0.4099 - accuracy: 0.818 - ETA: 14s - loss: 0.4098 - accuracy: 0.818 - ETA: 13s - loss: 0.4089 - accuracy: 0.819 - ETA: 12s - loss: 0.4095 - accuracy: 0.819 - ETA: 11s - loss: 0.4098 - accuracy: 0.819 - ETA: 11s - loss: 0.4108 - accuracy: 0.818 - ETA: 10s - loss: 0.4115 - accuracy: 0.818 - ETA: 9s - loss: 0.4111 - accuracy: 0.818 - ETA: 8s - loss: 0.4118 - accuracy: 0.81 - ETA: 7s - loss: 0.4113 - accuracy: 0.81 - ETA: 7s - loss: 0.4117 - accuracy: 0.81 - ETA: 6s - loss: 0.4117 - accuracy: 0.81 - ETA: 5s - loss: 0.4116 - accuracy: 0.81 - ETA: 4s - loss: 0.4115 - accuracy: 0.81 - ETA: 3s - loss: 0.4110 - accuracy: 0.81 - ETA: 3s - loss: 0.4107 - accuracy: 0.81 - ETA: 2s - loss: 0.4102 - accuracy: 0.81 - ETA: 1s - loss: 0.4105 - accuracy: 0.81 - ETA: 0s - loss: 0.4107 - accuracy: 0.81 - 84s 3ms/step - loss: 0.4104 - accuracy: 0.8200 - val_loss: 0.3727 - val_accuracy: 0.8371\n",
      "Epoch 3/30\n",
      "30000/30000 [==============================] - ETA: 1:19 - loss: 0.4355 - accuracy: 0.79 - ETA: 1:17 - loss: 0.4282 - accuracy: 0.80 - ETA: 1:16 - loss: 0.4111 - accuracy: 0.82 - ETA: 1:14 - loss: 0.4146 - accuracy: 0.82 - ETA: 1:13 - loss: 0.4103 - accuracy: 0.82 - ETA: 1:12 - loss: 0.4092 - accuracy: 0.82 - ETA: 1:11 - loss: 0.4063 - accuracy: 0.82 - ETA: 1:10 - loss: 0.3988 - accuracy: 0.83 - ETA: 1:09 - loss: 0.3875 - accuracy: 0.83 - ETA: 1:08 - loss: 0.3995 - accuracy: 0.83 - ETA: 1:07 - loss: 0.3946 - accuracy: 0.83 - ETA: 1:07 - loss: 0.3888 - accuracy: 0.83 - ETA: 1:06 - loss: 0.3931 - accuracy: 0.83 - ETA: 1:06 - loss: 0.3920 - accuracy: 0.83 - ETA: 1:05 - loss: 0.3930 - accuracy: 0.83 - ETA: 1:05 - loss: 0.3950 - accuracy: 0.83 - ETA: 1:04 - loss: 0.4003 - accuracy: 0.82 - ETA: 1:03 - loss: 0.4030 - accuracy: 0.82 - ETA: 1:03 - loss: 0.4027 - accuracy: 0.82 - ETA: 1:02 - loss: 0.4037 - accuracy: 0.82 - ETA: 1:01 - loss: 0.4035 - accuracy: 0.82 - ETA: 1:00 - loss: 0.4054 - accuracy: 0.82 - ETA: 59s - loss: 0.4027 - accuracy: 0.8255 - ETA: 59s - loss: 0.4009 - accuracy: 0.827 - ETA: 58s - loss: 0.4010 - accuracy: 0.827 - ETA: 57s - loss: 0.4012 - accuracy: 0.827 - ETA: 57s - loss: 0.3995 - accuracy: 0.828 - ETA: 56s - loss: 0.4027 - accuracy: 0.825 - ETA: 55s - loss: 0.4014 - accuracy: 0.826 - ETA: 54s - loss: 0.4010 - accuracy: 0.827 - ETA: 54s - loss: 0.4001 - accuracy: 0.827 - ETA: 53s - loss: 0.3989 - accuracy: 0.829 - ETA: 52s - loss: 0.3969 - accuracy: 0.830 - ETA: 51s - loss: 0.3965 - accuracy: 0.829 - ETA: 50s - loss: 0.3954 - accuracy: 0.830 - ETA: 50s - loss: 0.3956 - accuracy: 0.829 - ETA: 49s - loss: 0.3944 - accuracy: 0.830 - ETA: 48s - loss: 0.3928 - accuracy: 0.831 - ETA: 47s - loss: 0.3925 - accuracy: 0.831 - ETA: 46s - loss: 0.3938 - accuracy: 0.831 - ETA: 46s - loss: 0.3931 - accuracy: 0.831 - ETA: 45s - loss: 0.3929 - accuracy: 0.831 - ETA: 44s - loss: 0.3927 - accuracy: 0.831 - ETA: 43s - loss: 0.3913 - accuracy: 0.832 - ETA: 42s - loss: 0.3900 - accuracy: 0.833 - ETA: 41s - loss: 0.3899 - accuracy: 0.833 - ETA: 41s - loss: 0.3904 - accuracy: 0.833 - ETA: 40s - loss: 0.3887 - accuracy: 0.833 - ETA: 39s - loss: 0.3878 - accuracy: 0.833 - ETA: 39s - loss: 0.3873 - accuracy: 0.834 - ETA: 38s - loss: 0.3871 - accuracy: 0.833 - ETA: 37s - loss: 0.3851 - accuracy: 0.834 - ETA: 36s - loss: 0.3845 - accuracy: 0.835 - ETA: 35s - loss: 0.3832 - accuracy: 0.835 - ETA: 35s - loss: 0.3828 - accuracy: 0.835 - ETA: 34s - loss: 0.3816 - accuracy: 0.836 - ETA: 33s - loss: 0.3811 - accuracy: 0.837 - ETA: 32s - loss: 0.3792 - accuracy: 0.838 - ETA: 31s - loss: 0.3793 - accuracy: 0.837 - ETA: 31s - loss: 0.3793 - accuracy: 0.837 - ETA: 30s - loss: 0.3790 - accuracy: 0.838 - ETA: 29s - loss: 0.3777 - accuracy: 0.838 - ETA: 28s - loss: 0.3770 - accuracy: 0.838 - ETA: 28s - loss: 0.3761 - accuracy: 0.839 - ETA: 27s - loss: 0.3750 - accuracy: 0.839 - ETA: 26s - loss: 0.3748 - accuracy: 0.839 - ETA: 25s - loss: 0.3746 - accuracy: 0.840 - ETA: 24s - loss: 0.3742 - accuracy: 0.840 - ETA: 24s - loss: 0.3734 - accuracy: 0.840 - ETA: 23s - loss: 0.3738 - accuracy: 0.840 - ETA: 22s - loss: 0.3740 - accuracy: 0.840 - ETA: 21s - loss: 0.3739 - accuracy: 0.840 - ETA: 21s - loss: 0.3740 - accuracy: 0.840 - ETA: 20s - loss: 0.3739 - accuracy: 0.840 - ETA: 19s - loss: 0.3734 - accuracy: 0.840 - ETA: 18s - loss: 0.3730 - accuracy: 0.841 - ETA: 17s - loss: 0.3725 - accuracy: 0.841 - ETA: 17s - loss: 0.3719 - accuracy: 0.841 - ETA: 16s - loss: 0.3713 - accuracy: 0.841 - ETA: 15s - loss: 0.3703 - accuracy: 0.842 - ETA: 14s - loss: 0.3698 - accuracy: 0.842 - ETA: 14s - loss: 0.3695 - accuracy: 0.842 - ETA: 13s - loss: 0.3690 - accuracy: 0.842 - ETA: 12s - loss: 0.3681 - accuracy: 0.843 - ETA: 11s - loss: 0.3677 - accuracy: 0.843 - ETA: 10s - loss: 0.3675 - accuracy: 0.843 - ETA: 10s - loss: 0.3671 - accuracy: 0.843 - ETA: 9s - loss: 0.3666 - accuracy: 0.843 - ETA: 8s - loss: 0.3668 - accuracy: 0.84 - ETA: 7s - loss: 0.3660 - accuracy: 0.84 - ETA: 7s - loss: 0.3655 - accuracy: 0.84 - ETA: 6s - loss: 0.3649 - accuracy: 0.84 - ETA: 5s - loss: 0.3644 - accuracy: 0.84 - ETA: 4s - loss: 0.3647 - accuracy: 0.84 - ETA: 3s - loss: 0.3644 - accuracy: 0.84 - ETA: 3s - loss: 0.3643 - accuracy: 0.84 - ETA: 2s - loss: 0.3642 - accuracy: 0.84 - ETA: 1s - loss: 0.3636 - accuracy: 0.84 - ETA: 0s - loss: 0.3633 - accuracy: 0.84 - 84s 3ms/step - loss: 0.3626 - accuracy: 0.8448 - val_loss: 0.3131 - val_accuracy: 0.8679\n",
      "Epoch 4/30\n",
      "30000/30000 [==============================] - ETA: 1:19 - loss: 0.3108 - accuracy: 0.89 - ETA: 1:18 - loss: 0.3136 - accuracy: 0.86 - ETA: 1:18 - loss: 0.3248 - accuracy: 0.86 - ETA: 1:17 - loss: 0.3062 - accuracy: 0.86 - ETA: 1:16 - loss: 0.3136 - accuracy: 0.86 - ETA: 1:15 - loss: 0.3147 - accuracy: 0.86 - ETA: 1:14 - loss: 0.3126 - accuracy: 0.87 - ETA: 1:13 - loss: 0.3189 - accuracy: 0.86 - ETA: 1:12 - loss: 0.3214 - accuracy: 0.86 - ETA: 1:11 - loss: 0.3204 - accuracy: 0.86 - ETA: 1:10 - loss: 0.3217 - accuracy: 0.86 - ETA: 1:09 - loss: 0.3229 - accuracy: 0.86 - ETA: 1:08 - loss: 0.3233 - accuracy: 0.86 - ETA: 1:07 - loss: 0.3288 - accuracy: 0.86 - ETA: 1:06 - loss: 0.3289 - accuracy: 0.86 - ETA: 1:06 - loss: 0.3317 - accuracy: 0.86 - ETA: 1:05 - loss: 0.3318 - accuracy: 0.86 - ETA: 1:04 - loss: 0.3347 - accuracy: 0.86 - ETA: 1:03 - loss: 0.3328 - accuracy: 0.86 - ETA: 1:02 - loss: 0.3327 - accuracy: 0.86 - ETA: 1:01 - loss: 0.3312 - accuracy: 0.86 - ETA: 1:00 - loss: 0.3297 - accuracy: 0.86 - ETA: 59s - loss: 0.3331 - accuracy: 0.8651 - ETA: 58s - loss: 0.3314 - accuracy: 0.865 - ETA: 58s - loss: 0.3346 - accuracy: 0.864 - ETA: 57s - loss: 0.3341 - accuracy: 0.863 - ETA: 56s - loss: 0.3341 - accuracy: 0.863 - ETA: 55s - loss: 0.3336 - accuracy: 0.863 - ETA: 54s - loss: 0.3311 - accuracy: 0.863 - ETA: 54s - loss: 0.3304 - accuracy: 0.864 - ETA: 53s - loss: 0.3299 - accuracy: 0.864 - ETA: 52s - loss: 0.3295 - accuracy: 0.863 - ETA: 51s - loss: 0.3271 - accuracy: 0.864 - ETA: 50s - loss: 0.3276 - accuracy: 0.864 - ETA: 50s - loss: 0.3276 - accuracy: 0.864 - ETA: 49s - loss: 0.3285 - accuracy: 0.863 - ETA: 48s - loss: 0.3280 - accuracy: 0.864 - ETA: 47s - loss: 0.3264 - accuracy: 0.864 - ETA: 47s - loss: 0.3279 - accuracy: 0.864 - ETA: 46s - loss: 0.3285 - accuracy: 0.864 - ETA: 45s - loss: 0.3285 - accuracy: 0.864 - ETA: 44s - loss: 0.3271 - accuracy: 0.864 - ETA: 43s - loss: 0.3267 - accuracy: 0.865 - ETA: 43s - loss: 0.3267 - accuracy: 0.864 - ETA: 42s - loss: 0.3262 - accuracy: 0.865 - ETA: 41s - loss: 0.3249 - accuracy: 0.865 - ETA: 40s - loss: 0.3237 - accuracy: 0.866 - ETA: 40s - loss: 0.3229 - accuracy: 0.867 - ETA: 39s - loss: 0.3222 - accuracy: 0.867 - ETA: 38s - loss: 0.3215 - accuracy: 0.867 - ETA: 38s - loss: 0.3211 - accuracy: 0.867 - ETA: 37s - loss: 0.3213 - accuracy: 0.867 - ETA: 36s - loss: 0.3219 - accuracy: 0.867 - ETA: 35s - loss: 0.3228 - accuracy: 0.866 - ETA: 34s - loss: 0.3232 - accuracy: 0.867 - ETA: 34s - loss: 0.3233 - accuracy: 0.866 - ETA: 33s - loss: 0.3229 - accuracy: 0.867 - ETA: 32s - loss: 0.3226 - accuracy: 0.867 - ETA: 31s - loss: 0.3216 - accuracy: 0.868 - ETA: 31s - loss: 0.3215 - accuracy: 0.868 - ETA: 30s - loss: 0.3203 - accuracy: 0.868 - ETA: 29s - loss: 0.3210 - accuracy: 0.868 - ETA: 28s - loss: 0.3199 - accuracy: 0.869 - ETA: 28s - loss: 0.3209 - accuracy: 0.868 - ETA: 27s - loss: 0.3209 - accuracy: 0.868 - ETA: 26s - loss: 0.3216 - accuracy: 0.868 - ETA: 25s - loss: 0.3222 - accuracy: 0.868 - ETA: 25s - loss: 0.3228 - accuracy: 0.867 - ETA: 24s - loss: 0.3220 - accuracy: 0.867 - ETA: 23s - loss: 0.3222 - accuracy: 0.867 - ETA: 22s - loss: 0.3220 - accuracy: 0.867 - ETA: 22s - loss: 0.3210 - accuracy: 0.868 - ETA: 21s - loss: 0.3217 - accuracy: 0.868 - ETA: 20s - loss: 0.3224 - accuracy: 0.867 - ETA: 19s - loss: 0.3225 - accuracy: 0.867 - ETA: 18s - loss: 0.3228 - accuracy: 0.867 - ETA: 18s - loss: 0.3227 - accuracy: 0.867 - ETA: 17s - loss: 0.3230 - accuracy: 0.867 - ETA: 16s - loss: 0.3236 - accuracy: 0.867 - ETA: 15s - loss: 0.3238 - accuracy: 0.867 - ETA: 14s - loss: 0.3239 - accuracy: 0.867 - ETA: 14s - loss: 0.3242 - accuracy: 0.866 - ETA: 13s - loss: 0.3241 - accuracy: 0.866 - ETA: 12s - loss: 0.3243 - accuracy: 0.866 - ETA: 11s - loss: 0.3241 - accuracy: 0.866 - ETA: 11s - loss: 0.3239 - accuracy: 0.866 - ETA: 10s - loss: 0.3243 - accuracy: 0.866 - ETA: 9s - loss: 0.3257 - accuracy: 0.866 - ETA: 8s - loss: 0.3253 - accuracy: 0.86 - ETA: 7s - loss: 0.3253 - accuracy: 0.86 - ETA: 7s - loss: 0.3249 - accuracy: 0.86 - ETA: 6s - loss: 0.3245 - accuracy: 0.86 - ETA: 5s - loss: 0.3244 - accuracy: 0.86 - ETA: 4s - loss: 0.3245 - accuracy: 0.86 - ETA: 3s - loss: 0.3246 - accuracy: 0.86 - ETA: 3s - loss: 0.3245 - accuracy: 0.86 - ETA: 2s - loss: 0.3249 - accuracy: 0.86 - ETA: 1s - loss: 0.3245 - accuracy: 0.86 - ETA: 0s - loss: 0.3243 - accuracy: 0.86 - 84s 3ms/step - loss: 0.3240 - accuracy: 0.8670 - val_loss: 0.3076 - val_accuracy: 0.8727\n",
      "Epoch 5/30\n",
      "30000/30000 [==============================] - ETA: 1:18 - loss: 0.2795 - accuracy: 0.87 - ETA: 1:19 - loss: 0.2931 - accuracy: 0.86 - ETA: 1:17 - loss: 0.2873 - accuracy: 0.87 - ETA: 1:16 - loss: 0.2876 - accuracy: 0.87 - ETA: 1:15 - loss: 0.2835 - accuracy: 0.88 - ETA: 1:14 - loss: 0.2819 - accuracy: 0.88 - ETA: 1:13 - loss: 0.2715 - accuracy: 0.88 - ETA: 1:12 - loss: 0.2777 - accuracy: 0.88 - ETA: 1:11 - loss: 0.2804 - accuracy: 0.88 - ETA: 1:10 - loss: 0.2816 - accuracy: 0.88 - ETA: 1:08 - loss: 0.2828 - accuracy: 0.88 - ETA: 1:07 - loss: 0.2841 - accuracy: 0.88 - ETA: 1:07 - loss: 0.2868 - accuracy: 0.88 - ETA: 1:06 - loss: 0.2885 - accuracy: 0.87 - ETA: 1:06 - loss: 0.2888 - accuracy: 0.88 - ETA: 1:05 - loss: 0.2889 - accuracy: 0.88 - ETA: 1:04 - loss: 0.2890 - accuracy: 0.88 - ETA: 1:03 - loss: 0.2903 - accuracy: 0.88 - ETA: 1:02 - loss: 0.2873 - accuracy: 0.88 - ETA: 1:01 - loss: 0.2912 - accuracy: 0.87 - ETA: 1:01 - loss: 0.2922 - accuracy: 0.87 - ETA: 1:00 - loss: 0.2938 - accuracy: 0.87 - ETA: 59s - loss: 0.2913 - accuracy: 0.8804 - ETA: 58s - loss: 0.2909 - accuracy: 0.880 - ETA: 57s - loss: 0.2909 - accuracy: 0.880 - ETA: 57s - loss: 0.2900 - accuracy: 0.881 - ETA: 56s - loss: 0.2885 - accuracy: 0.882 - ETA: 55s - loss: 0.2870 - accuracy: 0.883 - ETA: 54s - loss: 0.2856 - accuracy: 0.883 - ETA: 54s - loss: 0.2860 - accuracy: 0.883 - ETA: 53s - loss: 0.2833 - accuracy: 0.884 - ETA: 52s - loss: 0.2822 - accuracy: 0.885 - ETA: 51s - loss: 0.2831 - accuracy: 0.884 - ETA: 51s - loss: 0.2831 - accuracy: 0.884 - ETA: 50s - loss: 0.2838 - accuracy: 0.884 - ETA: 49s - loss: 0.2831 - accuracy: 0.884 - ETA: 48s - loss: 0.2836 - accuracy: 0.884 - ETA: 47s - loss: 0.2863 - accuracy: 0.882 - ETA: 47s - loss: 0.2876 - accuracy: 0.882 - ETA: 46s - loss: 0.2875 - accuracy: 0.882 - ETA: 45s - loss: 0.2885 - accuracy: 0.882 - ETA: 44s - loss: 0.2904 - accuracy: 0.881 - ETA: 44s - loss: 0.2899 - accuracy: 0.882 - ETA: 43s - loss: 0.2907 - accuracy: 0.881 - ETA: 42s - loss: 0.2915 - accuracy: 0.880 - ETA: 41s - loss: 0.2916 - accuracy: 0.880 - ETA: 40s - loss: 0.2917 - accuracy: 0.880 - ETA: 40s - loss: 0.2920 - accuracy: 0.879 - ETA: 39s - loss: 0.2923 - accuracy: 0.879 - ETA: 38s - loss: 0.2919 - accuracy: 0.879 - ETA: 37s - loss: 0.2922 - accuracy: 0.879 - ETA: 36s - loss: 0.2924 - accuracy: 0.879 - ETA: 36s - loss: 0.2931 - accuracy: 0.879 - ETA: 35s - loss: 0.2942 - accuracy: 0.879 - ETA: 34s - loss: 0.2945 - accuracy: 0.878 - ETA: 33s - loss: 0.2945 - accuracy: 0.878 - ETA: 33s - loss: 0.2949 - accuracy: 0.878 - ETA: 32s - loss: 0.2954 - accuracy: 0.878 - ETA: 31s - loss: 0.2951 - accuracy: 0.879 - ETA: 30s - loss: 0.2958 - accuracy: 0.878 - ETA: 30s - loss: 0.2955 - accuracy: 0.878 - ETA: 29s - loss: 0.2961 - accuracy: 0.878 - ETA: 28s - loss: 0.2964 - accuracy: 0.878 - ETA: 27s - loss: 0.2970 - accuracy: 0.877 - ETA: 26s - loss: 0.2961 - accuracy: 0.877 - ETA: 26s - loss: 0.2959 - accuracy: 0.878 - ETA: 25s - loss: 0.2953 - accuracy: 0.878 - ETA: 24s - loss: 0.2955 - accuracy: 0.878 - ETA: 23s - loss: 0.2949 - accuracy: 0.878 - ETA: 23s - loss: 0.2948 - accuracy: 0.877 - ETA: 22s - loss: 0.2952 - accuracy: 0.877 - ETA: 21s - loss: 0.2952 - accuracy: 0.877 - ETA: 20s - loss: 0.2954 - accuracy: 0.877 - ETA: 19s - loss: 0.2957 - accuracy: 0.877 - ETA: 19s - loss: 0.2954 - accuracy: 0.877 - ETA: 18s - loss: 0.2948 - accuracy: 0.877 - ETA: 17s - loss: 0.2942 - accuracy: 0.878 - ETA: 16s - loss: 0.2941 - accuracy: 0.878 - ETA: 16s - loss: 0.2935 - accuracy: 0.878 - ETA: 15s - loss: 0.2929 - accuracy: 0.879 - ETA: 14s - loss: 0.2932 - accuracy: 0.879 - ETA: 13s - loss: 0.2930 - accuracy: 0.878 - ETA: 13s - loss: 0.2932 - accuracy: 0.879 - ETA: 12s - loss: 0.2932 - accuracy: 0.879 - ETA: 11s - loss: 0.2929 - accuracy: 0.879 - ETA: 10s - loss: 0.2935 - accuracy: 0.878 - ETA: 10s - loss: 0.2932 - accuracy: 0.879 - ETA: 9s - loss: 0.2929 - accuracy: 0.879 - ETA: 8s - loss: 0.2933 - accuracy: 0.87 - ETA: 7s - loss: 0.2932 - accuracy: 0.87 - ETA: 6s - loss: 0.2930 - accuracy: 0.87 - ETA: 6s - loss: 0.2932 - accuracy: 0.87 - ETA: 5s - loss: 0.2929 - accuracy: 0.87 - ETA: 4s - loss: 0.2925 - accuracy: 0.87 - ETA: 3s - loss: 0.2923 - accuracy: 0.87 - ETA: 3s - loss: 0.2924 - accuracy: 0.87 - ETA: 2s - loss: 0.2918 - accuracy: 0.87 - ETA: 1s - loss: 0.2922 - accuracy: 0.87 - ETA: 0s - loss: 0.2920 - accuracy: 0.87 - 83s 3ms/step - loss: 0.2927 - accuracy: 0.8792 - val_loss: 0.2809 - val_accuracy: 0.8834\n",
      "Epoch 6/30\n",
      "30000/30000 [==============================] - ETA: 1:20 - loss: 0.2492 - accuracy: 0.91 - ETA: 1:17 - loss: 0.2679 - accuracy: 0.90 - ETA: 1:16 - loss: 0.2577 - accuracy: 0.90 - ETA: 1:14 - loss: 0.2642 - accuracy: 0.90 - ETA: 1:13 - loss: 0.2623 - accuracy: 0.90 - ETA: 1:12 - loss: 0.2721 - accuracy: 0.89 - ETA: 1:12 - loss: 0.2739 - accuracy: 0.89 - ETA: 1:12 - loss: 0.2775 - accuracy: 0.88 - ETA: 1:11 - loss: 0.2744 - accuracy: 0.89 - ETA: 1:10 - loss: 0.2698 - accuracy: 0.89 - ETA: 1:09 - loss: 0.2691 - accuracy: 0.89 - ETA: 1:08 - loss: 0.2691 - accuracy: 0.89 - ETA: 1:07 - loss: 0.2709 - accuracy: 0.88 - ETA: 1:06 - loss: 0.2710 - accuracy: 0.88 - ETA: 1:05 - loss: 0.2681 - accuracy: 0.89 - ETA: 1:05 - loss: 0.2716 - accuracy: 0.89 - ETA: 1:04 - loss: 0.2716 - accuracy: 0.89 - ETA: 1:03 - loss: 0.2729 - accuracy: 0.89 - ETA: 1:02 - loss: 0.2734 - accuracy: 0.89 - ETA: 1:02 - loss: 0.2717 - accuracy: 0.89 - ETA: 1:01 - loss: 0.2719 - accuracy: 0.89 - ETA: 1:00 - loss: 0.2733 - accuracy: 0.89 - ETA: 59s - loss: 0.2713 - accuracy: 0.8916 - ETA: 58s - loss: 0.2704 - accuracy: 0.891 - ETA: 58s - loss: 0.2693 - accuracy: 0.891 - ETA: 57s - loss: 0.2691 - accuracy: 0.891 - ETA: 56s - loss: 0.2678 - accuracy: 0.892 - ETA: 55s - loss: 0.2667 - accuracy: 0.892 - ETA: 55s - loss: 0.2668 - accuracy: 0.892 - ETA: 54s - loss: 0.2669 - accuracy: 0.892 - ETA: 53s - loss: 0.2680 - accuracy: 0.891 - ETA: 52s - loss: 0.2681 - accuracy: 0.890 - ETA: 51s - loss: 0.2693 - accuracy: 0.890 - ETA: 51s - loss: 0.2695 - accuracy: 0.889 - ETA: 50s - loss: 0.2688 - accuracy: 0.890 - ETA: 49s - loss: 0.2684 - accuracy: 0.890 - ETA: 48s - loss: 0.2684 - accuracy: 0.890 - ETA: 48s - loss: 0.2705 - accuracy: 0.890 - ETA: 47s - loss: 0.2699 - accuracy: 0.890 - ETA: 46s - loss: 0.2697 - accuracy: 0.890 - ETA: 45s - loss: 0.2700 - accuracy: 0.890 - ETA: 45s - loss: 0.2698 - accuracy: 0.890 - ETA: 44s - loss: 0.2699 - accuracy: 0.890 - ETA: 43s - loss: 0.2693 - accuracy: 0.891 - ETA: 42s - loss: 0.2685 - accuracy: 0.891 - ETA: 41s - loss: 0.2689 - accuracy: 0.891 - ETA: 41s - loss: 0.2686 - accuracy: 0.892 - ETA: 40s - loss: 0.2684 - accuracy: 0.892 - ETA: 39s - loss: 0.2683 - accuracy: 0.892 - ETA: 38s - loss: 0.2689 - accuracy: 0.891 - ETA: 38s - loss: 0.2689 - accuracy: 0.892 - ETA: 37s - loss: 0.2690 - accuracy: 0.892 - ETA: 36s - loss: 0.2678 - accuracy: 0.892 - ETA: 35s - loss: 0.2677 - accuracy: 0.892 - ETA: 35s - loss: 0.2676 - accuracy: 0.893 - ETA: 34s - loss: 0.2679 - accuracy: 0.892 - ETA: 33s - loss: 0.2685 - accuracy: 0.892 - ETA: 32s - loss: 0.2687 - accuracy: 0.892 - ETA: 32s - loss: 0.2678 - accuracy: 0.893 - ETA: 31s - loss: 0.2681 - accuracy: 0.892 - ETA: 30s - loss: 0.2679 - accuracy: 0.892 - ETA: 29s - loss: 0.2681 - accuracy: 0.892 - ETA: 28s - loss: 0.2675 - accuracy: 0.892 - ETA: 28s - loss: 0.2678 - accuracy: 0.892 - ETA: 27s - loss: 0.2679 - accuracy: 0.892 - ETA: 26s - loss: 0.2680 - accuracy: 0.892 - ETA: 25s - loss: 0.2679 - accuracy: 0.892 - ETA: 25s - loss: 0.2672 - accuracy: 0.892 - ETA: 24s - loss: 0.2667 - accuracy: 0.892 - ETA: 23s - loss: 0.2662 - accuracy: 0.892 - ETA: 22s - loss: 0.2657 - accuracy: 0.893 - ETA: 21s - loss: 0.2656 - accuracy: 0.892 - ETA: 21s - loss: 0.2656 - accuracy: 0.892 - ETA: 20s - loss: 0.2657 - accuracy: 0.893 - ETA: 19s - loss: 0.2648 - accuracy: 0.893 - ETA: 18s - loss: 0.2650 - accuracy: 0.893 - ETA: 17s - loss: 0.2650 - accuracy: 0.893 - ETA: 17s - loss: 0.2654 - accuracy: 0.893 - ETA: 16s - loss: 0.2653 - accuracy: 0.893 - ETA: 15s - loss: 0.2654 - accuracy: 0.893 - ETA: 14s - loss: 0.2650 - accuracy: 0.893 - ETA: 14s - loss: 0.2650 - accuracy: 0.893 - ETA: 13s - loss: 0.2648 - accuracy: 0.894 - ETA: 12s - loss: 0.2656 - accuracy: 0.893 - ETA: 11s - loss: 0.2654 - accuracy: 0.893 - ETA: 10s - loss: 0.2653 - accuracy: 0.893 - ETA: 10s - loss: 0.2658 - accuracy: 0.893 - ETA: 9s - loss: 0.2663 - accuracy: 0.892 - ETA: 8s - loss: 0.2667 - accuracy: 0.89 - ETA: 7s - loss: 0.2669 - accuracy: 0.89 - ETA: 7s - loss: 0.2680 - accuracy: 0.89 - ETA: 6s - loss: 0.2684 - accuracy: 0.89 - ETA: 5s - loss: 0.2684 - accuracy: 0.89 - ETA: 4s - loss: 0.2685 - accuracy: 0.89 - ETA: 3s - loss: 0.2685 - accuracy: 0.89 - ETA: 3s - loss: 0.2687 - accuracy: 0.89 - ETA: 2s - loss: 0.2686 - accuracy: 0.89 - ETA: 1s - loss: 0.2682 - accuracy: 0.89 - ETA: 0s - loss: 0.2684 - accuracy: 0.89 - 84s 3ms/step - loss: 0.2682 - accuracy: 0.8920 - val_loss: 0.2867 - val_accuracy: 0.8868\n",
      "Epoch 7/30\n",
      "30000/30000 [==============================] - ETA: 1:15 - loss: 0.2549 - accuracy: 0.91 - ETA: 1:18 - loss: 0.2328 - accuracy: 0.91 - ETA: 1:18 - loss: 0.2325 - accuracy: 0.90 - ETA: 1:18 - loss: 0.2474 - accuracy: 0.90 - ETA: 1:16 - loss: 0.2551 - accuracy: 0.89 - ETA: 1:15 - loss: 0.2579 - accuracy: 0.89 - ETA: 1:14 - loss: 0.2572 - accuracy: 0.89 - ETA: 1:13 - loss: 0.2589 - accuracy: 0.89 - ETA: 1:12 - loss: 0.2566 - accuracy: 0.89 - ETA: 1:11 - loss: 0.2555 - accuracy: 0.89 - ETA: 1:10 - loss: 0.2587 - accuracy: 0.89 - ETA: 1:09 - loss: 0.2599 - accuracy: 0.89 - ETA: 1:08 - loss: 0.2595 - accuracy: 0.89 - ETA: 1:07 - loss: 0.2618 - accuracy: 0.89 - ETA: 1:07 - loss: 0.2643 - accuracy: 0.89 - ETA: 1:06 - loss: 0.2627 - accuracy: 0.89 - ETA: 1:05 - loss: 0.2612 - accuracy: 0.89 - ETA: 1:04 - loss: 0.2605 - accuracy: 0.89 - ETA: 1:03 - loss: 0.2595 - accuracy: 0.89 - ETA: 1:02 - loss: 0.2565 - accuracy: 0.89 - ETA: 1:01 - loss: 0.2583 - accuracy: 0.89 - ETA: 1:00 - loss: 0.2558 - accuracy: 0.89 - ETA: 1:00 - loss: 0.2538 - accuracy: 0.89 - ETA: 59s - loss: 0.2550 - accuracy: 0.8989 - ETA: 58s - loss: 0.2560 - accuracy: 0.898 - ETA: 57s - loss: 0.2572 - accuracy: 0.898 - ETA: 57s - loss: 0.2563 - accuracy: 0.897 - ETA: 56s - loss: 0.2572 - accuracy: 0.897 - ETA: 55s - loss: 0.2570 - accuracy: 0.897 - ETA: 54s - loss: 0.2561 - accuracy: 0.897 - ETA: 53s - loss: 0.2582 - accuracy: 0.896 - ETA: 52s - loss: 0.2585 - accuracy: 0.896 - ETA: 52s - loss: 0.2600 - accuracy: 0.895 - ETA: 51s - loss: 0.2612 - accuracy: 0.895 - ETA: 50s - loss: 0.2613 - accuracy: 0.895 - ETA: 49s - loss: 0.2625 - accuracy: 0.894 - ETA: 49s - loss: 0.2640 - accuracy: 0.894 - ETA: 48s - loss: 0.2638 - accuracy: 0.894 - ETA: 47s - loss: 0.2634 - accuracy: 0.895 - ETA: 46s - loss: 0.2632 - accuracy: 0.895 - ETA: 45s - loss: 0.2627 - accuracy: 0.895 - ETA: 45s - loss: 0.2637 - accuracy: 0.895 - ETA: 44s - loss: 0.2654 - accuracy: 0.894 - ETA: 43s - loss: 0.2653 - accuracy: 0.894 - ETA: 42s - loss: 0.2639 - accuracy: 0.895 - ETA: 42s - loss: 0.2642 - accuracy: 0.895 - ETA: 41s - loss: 0.2642 - accuracy: 0.895 - ETA: 40s - loss: 0.2638 - accuracy: 0.895 - ETA: 39s - loss: 0.2629 - accuracy: 0.895 - ETA: 39s - loss: 0.2623 - accuracy: 0.896 - ETA: 38s - loss: 0.2630 - accuracy: 0.896 - ETA: 37s - loss: 0.2619 - accuracy: 0.896 - ETA: 36s - loss: 0.2612 - accuracy: 0.897 - ETA: 35s - loss: 0.2620 - accuracy: 0.896 - ETA: 35s - loss: 0.2611 - accuracy: 0.897 - ETA: 34s - loss: 0.2606 - accuracy: 0.897 - ETA: 33s - loss: 0.2597 - accuracy: 0.897 - ETA: 32s - loss: 0.2601 - accuracy: 0.897 - ETA: 32s - loss: 0.2600 - accuracy: 0.897 - ETA: 31s - loss: 0.2596 - accuracy: 0.897 - ETA: 30s - loss: 0.2588 - accuracy: 0.898 - ETA: 29s - loss: 0.2595 - accuracy: 0.898 - ETA: 28s - loss: 0.2591 - accuracy: 0.898 - ETA: 28s - loss: 0.2588 - accuracy: 0.898 - ETA: 27s - loss: 0.2581 - accuracy: 0.899 - ETA: 26s - loss: 0.2584 - accuracy: 0.898 - ETA: 25s - loss: 0.2580 - accuracy: 0.899 - ETA: 25s - loss: 0.2576 - accuracy: 0.899 - ETA: 24s - loss: 0.2575 - accuracy: 0.899 - ETA: 23s - loss: 0.2569 - accuracy: 0.899 - ETA: 22s - loss: 0.2578 - accuracy: 0.899 - ETA: 21s - loss: 0.2573 - accuracy: 0.899 - ETA: 21s - loss: 0.2575 - accuracy: 0.899 - ETA: 20s - loss: 0.2567 - accuracy: 0.899 - ETA: 19s - loss: 0.2567 - accuracy: 0.899 - ETA: 18s - loss: 0.2570 - accuracy: 0.899 - ETA: 18s - loss: 0.2576 - accuracy: 0.899 - ETA: 17s - loss: 0.2574 - accuracy: 0.899 - ETA: 16s - loss: 0.2574 - accuracy: 0.899 - ETA: 15s - loss: 0.2568 - accuracy: 0.899 - ETA: 14s - loss: 0.2567 - accuracy: 0.899 - ETA: 14s - loss: 0.2563 - accuracy: 0.899 - ETA: 13s - loss: 0.2560 - accuracy: 0.899 - ETA: 12s - loss: 0.2557 - accuracy: 0.899 - ETA: 11s - loss: 0.2555 - accuracy: 0.899 - ETA: 10s - loss: 0.2552 - accuracy: 0.899 - ETA: 10s - loss: 0.2544 - accuracy: 0.900 - ETA: 9s - loss: 0.2542 - accuracy: 0.900 - ETA: 8s - loss: 0.2539 - accuracy: 0.90 - ETA: 7s - loss: 0.2533 - accuracy: 0.90 - ETA: 7s - loss: 0.2528 - accuracy: 0.90 - ETA: 6s - loss: 0.2533 - accuracy: 0.90 - ETA: 5s - loss: 0.2533 - accuracy: 0.90 - ETA: 4s - loss: 0.2537 - accuracy: 0.90 - ETA: 3s - loss: 0.2537 - accuracy: 0.90 - ETA: 3s - loss: 0.2537 - accuracy: 0.90 - ETA: 2s - loss: 0.2537 - accuracy: 0.90 - ETA: 1s - loss: 0.2538 - accuracy: 0.90 - ETA: 0s - loss: 0.2542 - accuracy: 0.90 - 84s 3ms/step - loss: 0.2539 - accuracy: 0.9004 - val_loss: 0.2775 - val_accuracy: 0.8842\n",
      "Epoch 8/30\n",
      "30000/30000 [==============================] - ETA: 1:16 - loss: 0.2216 - accuracy: 0.90 - ETA: 1:16 - loss: 0.2129 - accuracy: 0.90 - ETA: 1:16 - loss: 0.2183 - accuracy: 0.90 - ETA: 1:16 - loss: 0.2118 - accuracy: 0.90 - ETA: 1:14 - loss: 0.2109 - accuracy: 0.90 - ETA: 1:14 - loss: 0.2083 - accuracy: 0.91 - ETA: 1:14 - loss: 0.2162 - accuracy: 0.90 - ETA: 1:12 - loss: 0.2241 - accuracy: 0.90 - ETA: 1:12 - loss: 0.2295 - accuracy: 0.90 - ETA: 1:11 - loss: 0.2248 - accuracy: 0.90 - ETA: 1:10 - loss: 0.2285 - accuracy: 0.90 - ETA: 1:09 - loss: 0.2244 - accuracy: 0.90 - ETA: 1:08 - loss: 0.2310 - accuracy: 0.90 - ETA: 1:08 - loss: 0.2298 - accuracy: 0.90 - ETA: 1:07 - loss: 0.2257 - accuracy: 0.90 - ETA: 1:06 - loss: 0.2243 - accuracy: 0.90 - ETA: 1:05 - loss: 0.2258 - accuracy: 0.90 - ETA: 1:04 - loss: 0.2252 - accuracy: 0.90 - ETA: 1:03 - loss: 0.2266 - accuracy: 0.90 - ETA: 1:03 - loss: 0.2267 - accuracy: 0.90 - ETA: 1:02 - loss: 0.2269 - accuracy: 0.90 - ETA: 1:01 - loss: 0.2275 - accuracy: 0.90 - ETA: 1:00 - loss: 0.2281 - accuracy: 0.90 - ETA: 59s - loss: 0.2283 - accuracy: 0.9072 - ETA: 59s - loss: 0.2273 - accuracy: 0.907 - ETA: 58s - loss: 0.2306 - accuracy: 0.906 - ETA: 57s - loss: 0.2295 - accuracy: 0.906 - ETA: 56s - loss: 0.2305 - accuracy: 0.906 - ETA: 55s - loss: 0.2308 - accuracy: 0.906 - ETA: 55s - loss: 0.2328 - accuracy: 0.905 - ETA: 54s - loss: 0.2324 - accuracy: 0.905 - ETA: 53s - loss: 0.2334 - accuracy: 0.904 - ETA: 52s - loss: 0.2342 - accuracy: 0.904 - ETA: 51s - loss: 0.2334 - accuracy: 0.905 - ETA: 51s - loss: 0.2333 - accuracy: 0.905 - ETA: 50s - loss: 0.2360 - accuracy: 0.904 - ETA: 49s - loss: 0.2374 - accuracy: 0.903 - ETA: 48s - loss: 0.2372 - accuracy: 0.903 - ETA: 47s - loss: 0.2382 - accuracy: 0.903 - ETA: 46s - loss: 0.2370 - accuracy: 0.904 - ETA: 46s - loss: 0.2370 - accuracy: 0.904 - ETA: 45s - loss: 0.2366 - accuracy: 0.904 - ETA: 44s - loss: 0.2363 - accuracy: 0.905 - ETA: 43s - loss: 0.2369 - accuracy: 0.904 - ETA: 42s - loss: 0.2373 - accuracy: 0.904 - ETA: 42s - loss: 0.2385 - accuracy: 0.904 - ETA: 41s - loss: 0.2381 - accuracy: 0.904 - ETA: 40s - loss: 0.2373 - accuracy: 0.905 - ETA: 39s - loss: 0.2370 - accuracy: 0.905 - ETA: 38s - loss: 0.2379 - accuracy: 0.904 - ETA: 38s - loss: 0.2363 - accuracy: 0.905 - ETA: 37s - loss: 0.2358 - accuracy: 0.905 - ETA: 36s - loss: 0.2362 - accuracy: 0.905 - ETA: 35s - loss: 0.2353 - accuracy: 0.906 - ETA: 35s - loss: 0.2364 - accuracy: 0.906 - ETA: 34s - loss: 0.2360 - accuracy: 0.906 - ETA: 33s - loss: 0.2356 - accuracy: 0.906 - ETA: 32s - loss: 0.2353 - accuracy: 0.906 - ETA: 31s - loss: 0.2349 - accuracy: 0.906 - ETA: 31s - loss: 0.2351 - accuracy: 0.906 - ETA: 30s - loss: 0.2359 - accuracy: 0.906 - ETA: 29s - loss: 0.2362 - accuracy: 0.906 - ETA: 28s - loss: 0.2359 - accuracy: 0.906 - ETA: 28s - loss: 0.2367 - accuracy: 0.905 - ETA: 27s - loss: 0.2368 - accuracy: 0.906 - ETA: 26s - loss: 0.2369 - accuracy: 0.906 - ETA: 25s - loss: 0.2361 - accuracy: 0.906 - ETA: 24s - loss: 0.2359 - accuracy: 0.906 - ETA: 24s - loss: 0.2362 - accuracy: 0.906 - ETA: 23s - loss: 0.2354 - accuracy: 0.906 - ETA: 22s - loss: 0.2350 - accuracy: 0.907 - ETA: 21s - loss: 0.2351 - accuracy: 0.907 - ETA: 21s - loss: 0.2358 - accuracy: 0.906 - ETA: 20s - loss: 0.2365 - accuracy: 0.906 - ETA: 19s - loss: 0.2362 - accuracy: 0.906 - ETA: 18s - loss: 0.2364 - accuracy: 0.906 - ETA: 17s - loss: 0.2368 - accuracy: 0.906 - ETA: 17s - loss: 0.2371 - accuracy: 0.905 - ETA: 16s - loss: 0.2371 - accuracy: 0.905 - ETA: 15s - loss: 0.2367 - accuracy: 0.906 - ETA: 14s - loss: 0.2369 - accuracy: 0.906 - ETA: 14s - loss: 0.2369 - accuracy: 0.906 - ETA: 13s - loss: 0.2370 - accuracy: 0.906 - ETA: 12s - loss: 0.2367 - accuracy: 0.906 - ETA: 11s - loss: 0.2372 - accuracy: 0.906 - ETA: 10s - loss: 0.2368 - accuracy: 0.906 - ETA: 10s - loss: 0.2367 - accuracy: 0.906 - ETA: 9s - loss: 0.2361 - accuracy: 0.906 - ETA: 8s - loss: 0.2363 - accuracy: 0.90 - ETA: 7s - loss: 0.2369 - accuracy: 0.90 - ETA: 7s - loss: 0.2364 - accuracy: 0.90 - ETA: 6s - loss: 0.2375 - accuracy: 0.90 - ETA: 5s - loss: 0.2370 - accuracy: 0.90 - ETA: 4s - loss: 0.2375 - accuracy: 0.90 - ETA: 3s - loss: 0.2376 - accuracy: 0.90 - ETA: 3s - loss: 0.2373 - accuracy: 0.90 - ETA: 2s - loss: 0.2370 - accuracy: 0.90 - ETA: 1s - loss: 0.2376 - accuracy: 0.90 - ETA: 0s - loss: 0.2384 - accuracy: 0.90 - 83s 3ms/step - loss: 0.2383 - accuracy: 0.9062 - val_loss: 0.2862 - val_accuracy: 0.8870\n",
      "Epoch 9/30\n",
      "30000/30000 [==============================] - ETA: 1:19 - loss: 0.2159 - accuracy: 0.93 - ETA: 1:14 - loss: 0.2681 - accuracy: 0.90 - ETA: 1:14 - loss: 0.2497 - accuracy: 0.90 - ETA: 1:14 - loss: 0.2531 - accuracy: 0.90 - ETA: 1:13 - loss: 0.2697 - accuracy: 0.89 - ETA: 1:12 - loss: 0.2595 - accuracy: 0.89 - ETA: 1:11 - loss: 0.2516 - accuracy: 0.90 - ETA: 1:11 - loss: 0.2609 - accuracy: 0.89 - ETA: 1:10 - loss: 0.2619 - accuracy: 0.89 - ETA: 1:09 - loss: 0.2579 - accuracy: 0.89 - ETA: 1:09 - loss: 0.2600 - accuracy: 0.89 - ETA: 1:08 - loss: 0.2581 - accuracy: 0.89 - ETA: 1:07 - loss: 0.2647 - accuracy: 0.89 - ETA: 1:07 - loss: 0.2638 - accuracy: 0.89 - ETA: 1:06 - loss: 0.2623 - accuracy: 0.89 - ETA: 1:05 - loss: 0.2649 - accuracy: 0.89 - ETA: 1:04 - loss: 0.2635 - accuracy: 0.89 - ETA: 1:03 - loss: 0.2607 - accuracy: 0.89 - ETA: 1:02 - loss: 0.2601 - accuracy: 0.89 - ETA: 1:02 - loss: 0.2586 - accuracy: 0.89 - ETA: 1:01 - loss: 0.2594 - accuracy: 0.89 - ETA: 1:00 - loss: 0.2618 - accuracy: 0.89 - ETA: 59s - loss: 0.2603 - accuracy: 0.8952 - ETA: 59s - loss: 0.2602 - accuracy: 0.894 - ETA: 58s - loss: 0.2596 - accuracy: 0.895 - ETA: 57s - loss: 0.2598 - accuracy: 0.895 - ETA: 56s - loss: 0.2574 - accuracy: 0.897 - ETA: 56s - loss: 0.2564 - accuracy: 0.897 - ETA: 55s - loss: 0.2539 - accuracy: 0.898 - ETA: 54s - loss: 0.2510 - accuracy: 0.899 - ETA: 53s - loss: 0.2490 - accuracy: 0.900 - ETA: 53s - loss: 0.2494 - accuracy: 0.901 - ETA: 52s - loss: 0.2500 - accuracy: 0.901 - ETA: 51s - loss: 0.2484 - accuracy: 0.901 - ETA: 50s - loss: 0.2488 - accuracy: 0.901 - ETA: 49s - loss: 0.2481 - accuracy: 0.901 - ETA: 48s - loss: 0.2480 - accuracy: 0.901 - ETA: 48s - loss: 0.2478 - accuracy: 0.901 - ETA: 47s - loss: 0.2484 - accuracy: 0.900 - ETA: 46s - loss: 0.2480 - accuracy: 0.901 - ETA: 45s - loss: 0.2468 - accuracy: 0.901 - ETA: 45s - loss: 0.2454 - accuracy: 0.902 - ETA: 44s - loss: 0.2436 - accuracy: 0.903 - ETA: 43s - loss: 0.2433 - accuracy: 0.903 - ETA: 42s - loss: 0.2438 - accuracy: 0.902 - ETA: 42s - loss: 0.2438 - accuracy: 0.902 - ETA: 41s - loss: 0.2443 - accuracy: 0.902 - ETA: 40s - loss: 0.2438 - accuracy: 0.902 - ETA: 40s - loss: 0.2437 - accuracy: 0.902 - ETA: 39s - loss: 0.2432 - accuracy: 0.903 - ETA: 38s - loss: 0.2422 - accuracy: 0.903 - ETA: 37s - loss: 0.2427 - accuracy: 0.902 - ETA: 36s - loss: 0.2441 - accuracy: 0.902 - ETA: 36s - loss: 0.2433 - accuracy: 0.903 - ETA: 35s - loss: 0.2430 - accuracy: 0.903 - ETA: 34s - loss: 0.2424 - accuracy: 0.903 - ETA: 33s - loss: 0.2416 - accuracy: 0.904 - ETA: 33s - loss: 0.2420 - accuracy: 0.903 - ETA: 32s - loss: 0.2422 - accuracy: 0.903 - ETA: 31s - loss: 0.2417 - accuracy: 0.904 - ETA: 30s - loss: 0.2410 - accuracy: 0.904 - ETA: 29s - loss: 0.2405 - accuracy: 0.904 - ETA: 29s - loss: 0.2394 - accuracy: 0.905 - ETA: 28s - loss: 0.2398 - accuracy: 0.905 - ETA: 27s - loss: 0.2395 - accuracy: 0.905 - ETA: 26s - loss: 0.2389 - accuracy: 0.906 - ETA: 26s - loss: 0.2387 - accuracy: 0.906 - ETA: 25s - loss: 0.2376 - accuracy: 0.906 - ETA: 24s - loss: 0.2366 - accuracy: 0.907 - ETA: 23s - loss: 0.2368 - accuracy: 0.907 - ETA: 23s - loss: 0.2360 - accuracy: 0.907 - ETA: 22s - loss: 0.2360 - accuracy: 0.907 - ETA: 21s - loss: 0.2359 - accuracy: 0.907 - ETA: 20s - loss: 0.2356 - accuracy: 0.907 - ETA: 19s - loss: 0.2355 - accuracy: 0.907 - ETA: 19s - loss: 0.2350 - accuracy: 0.908 - ETA: 18s - loss: 0.2342 - accuracy: 0.908 - ETA: 17s - loss: 0.2340 - accuracy: 0.908 - ETA: 16s - loss: 0.2326 - accuracy: 0.908 - ETA: 15s - loss: 0.2320 - accuracy: 0.909 - ETA: 15s - loss: 0.2326 - accuracy: 0.909 - ETA: 14s - loss: 0.2317 - accuracy: 0.909 - ETA: 13s - loss: 0.2318 - accuracy: 0.909 - ETA: 12s - loss: 0.2315 - accuracy: 0.909 - ETA: 11s - loss: 0.2312 - accuracy: 0.909 - ETA: 11s - loss: 0.2308 - accuracy: 0.910 - ETA: 10s - loss: 0.2308 - accuracy: 0.909 - ETA: 9s - loss: 0.2305 - accuracy: 0.910 - ETA: 8s - loss: 0.2303 - accuracy: 0.91 - ETA: 7s - loss: 0.2305 - accuracy: 0.91 - ETA: 7s - loss: 0.2307 - accuracy: 0.91 - ETA: 6s - loss: 0.2304 - accuracy: 0.90 - ETA: 5s - loss: 0.2303 - accuracy: 0.91 - ETA: 4s - loss: 0.2302 - accuracy: 0.91 - ETA: 3s - loss: 0.2298 - accuracy: 0.91 - ETA: 3s - loss: 0.2295 - accuracy: 0.91 - ETA: 2s - loss: 0.2298 - accuracy: 0.91 - ETA: 1s - loss: 0.2296 - accuracy: 0.91 - ETA: 0s - loss: 0.2296 - accuracy: 0.91 - 85s 3ms/step - loss: 0.2304 - accuracy: 0.9101 - val_loss: 0.2704 - val_accuracy: 0.8943\n",
      "Epoch 10/30\n",
      "30000/30000 [==============================] - ETA: 1:16 - loss: 0.2576 - accuracy: 0.91 - ETA: 1:18 - loss: 0.2437 - accuracy: 0.91 - ETA: 1:15 - loss: 0.2133 - accuracy: 0.92 - ETA: 1:14 - loss: 0.2164 - accuracy: 0.91 - ETA: 1:13 - loss: 0.2240 - accuracy: 0.91 - ETA: 1:13 - loss: 0.2158 - accuracy: 0.91 - ETA: 1:11 - loss: 0.2183 - accuracy: 0.91 - ETA: 1:11 - loss: 0.2169 - accuracy: 0.91 - ETA: 1:10 - loss: 0.2099 - accuracy: 0.91 - ETA: 1:09 - loss: 0.2047 - accuracy: 0.92 - ETA: 1:08 - loss: 0.2022 - accuracy: 0.92 - ETA: 1:07 - loss: 0.2038 - accuracy: 0.92 - ETA: 1:06 - loss: 0.1997 - accuracy: 0.92 - ETA: 1:05 - loss: 0.1966 - accuracy: 0.92 - ETA: 1:05 - loss: 0.1927 - accuracy: 0.92 - ETA: 1:04 - loss: 0.1915 - accuracy: 0.92 - ETA: 1:04 - loss: 0.1925 - accuracy: 0.92 - ETA: 1:03 - loss: 0.1921 - accuracy: 0.92 - ETA: 1:02 - loss: 0.1922 - accuracy: 0.92 - ETA: 1:01 - loss: 0.1967 - accuracy: 0.92 - ETA: 1:00 - loss: 0.1979 - accuracy: 0.92 - ETA: 1:00 - loss: 0.2017 - accuracy: 0.92 - ETA: 59s - loss: 0.2042 - accuracy: 0.9204 - ETA: 58s - loss: 0.2068 - accuracy: 0.919 - ETA: 58s - loss: 0.2065 - accuracy: 0.919 - ETA: 57s - loss: 0.2069 - accuracy: 0.919 - ETA: 56s - loss: 0.2086 - accuracy: 0.918 - ETA: 56s - loss: 0.2077 - accuracy: 0.919 - ETA: 55s - loss: 0.2069 - accuracy: 0.918 - ETA: 54s - loss: 0.2052 - accuracy: 0.919 - ETA: 53s - loss: 0.2055 - accuracy: 0.919 - ETA: 52s - loss: 0.2078 - accuracy: 0.918 - ETA: 52s - loss: 0.2090 - accuracy: 0.918 - ETA: 51s - loss: 0.2090 - accuracy: 0.918 - ETA: 50s - loss: 0.2080 - accuracy: 0.918 - ETA: 49s - loss: 0.2062 - accuracy: 0.919 - ETA: 49s - loss: 0.2062 - accuracy: 0.919 - ETA: 48s - loss: 0.2073 - accuracy: 0.918 - ETA: 47s - loss: 0.2070 - accuracy: 0.918 - ETA: 46s - loss: 0.2081 - accuracy: 0.918 - ETA: 46s - loss: 0.2082 - accuracy: 0.918 - ETA: 45s - loss: 0.2086 - accuracy: 0.917 - ETA: 44s - loss: 0.2081 - accuracy: 0.918 - ETA: 43s - loss: 0.2098 - accuracy: 0.917 - ETA: 42s - loss: 0.2096 - accuracy: 0.917 - ETA: 42s - loss: 0.2092 - accuracy: 0.917 - ETA: 41s - loss: 0.2101 - accuracy: 0.917 - ETA: 40s - loss: 0.2098 - accuracy: 0.917 - ETA: 39s - loss: 0.2101 - accuracy: 0.917 - ETA: 39s - loss: 0.2096 - accuracy: 0.918 - ETA: 38s - loss: 0.2096 - accuracy: 0.918 - ETA: 37s - loss: 0.2088 - accuracy: 0.918 - ETA: 36s - loss: 0.2081 - accuracy: 0.918 - ETA: 35s - loss: 0.2081 - accuracy: 0.918 - ETA: 35s - loss: 0.2080 - accuracy: 0.918 - ETA: 34s - loss: 0.2070 - accuracy: 0.919 - ETA: 33s - loss: 0.2063 - accuracy: 0.919 - ETA: 32s - loss: 0.2058 - accuracy: 0.919 - ETA: 31s - loss: 0.2057 - accuracy: 0.919 - ETA: 31s - loss: 0.2060 - accuracy: 0.919 - ETA: 30s - loss: 0.2053 - accuracy: 0.920 - ETA: 29s - loss: 0.2055 - accuracy: 0.920 - ETA: 28s - loss: 0.2051 - accuracy: 0.920 - ETA: 28s - loss: 0.2054 - accuracy: 0.920 - ETA: 27s - loss: 0.2049 - accuracy: 0.921 - ETA: 26s - loss: 0.2048 - accuracy: 0.921 - ETA: 25s - loss: 0.2048 - accuracy: 0.921 - ETA: 25s - loss: 0.2056 - accuracy: 0.921 - ETA: 24s - loss: 0.2057 - accuracy: 0.921 - ETA: 23s - loss: 0.2054 - accuracy: 0.921 - ETA: 22s - loss: 0.2058 - accuracy: 0.920 - ETA: 21s - loss: 0.2062 - accuracy: 0.920 - ETA: 21s - loss: 0.2054 - accuracy: 0.920 - ETA: 20s - loss: 0.2058 - accuracy: 0.920 - ETA: 19s - loss: 0.2053 - accuracy: 0.920 - ETA: 18s - loss: 0.2048 - accuracy: 0.921 - ETA: 17s - loss: 0.2053 - accuracy: 0.920 - ETA: 17s - loss: 0.2058 - accuracy: 0.920 - ETA: 16s - loss: 0.2055 - accuracy: 0.920 - ETA: 15s - loss: 0.2053 - accuracy: 0.920 - ETA: 14s - loss: 0.2059 - accuracy: 0.920 - ETA: 14s - loss: 0.2063 - accuracy: 0.919 - ETA: 13s - loss: 0.2061 - accuracy: 0.920 - ETA: 12s - loss: 0.2061 - accuracy: 0.920 - ETA: 11s - loss: 0.2067 - accuracy: 0.919 - ETA: 10s - loss: 0.2067 - accuracy: 0.920 - ETA: 10s - loss: 0.2066 - accuracy: 0.919 - ETA: 9s - loss: 0.2061 - accuracy: 0.920 - ETA: 8s - loss: 0.2062 - accuracy: 0.92 - ETA: 7s - loss: 0.2063 - accuracy: 0.91 - ETA: 7s - loss: 0.2065 - accuracy: 0.91 - ETA: 6s - loss: 0.2063 - accuracy: 0.92 - ETA: 5s - loss: 0.2064 - accuracy: 0.92 - ETA: 4s - loss: 0.2060 - accuracy: 0.92 - ETA: 3s - loss: 0.2060 - accuracy: 0.92 - ETA: 3s - loss: 0.2071 - accuracy: 0.92 - ETA: 2s - loss: 0.2063 - accuracy: 0.92 - ETA: 1s - loss: 0.2063 - accuracy: 0.92 - ETA: 0s - loss: 0.2062 - accuracy: 0.92 - 84s 3ms/step - loss: 0.2056 - accuracy: 0.9206 - val_loss: 0.2589 - val_accuracy: 0.8965\n",
      "Epoch 11/30\n",
      "30000/30000 [==============================] - ETA: 1:15 - loss: 0.1873 - accuracy: 0.92 - ETA: 1:14 - loss: 0.1772 - accuracy: 0.93 - ETA: 1:15 - loss: 0.1938 - accuracy: 0.92 - ETA: 1:15 - loss: 0.1778 - accuracy: 0.93 - ETA: 1:14 - loss: 0.1870 - accuracy: 0.93 - ETA: 1:13 - loss: 0.1912 - accuracy: 0.92 - ETA: 1:12 - loss: 0.1891 - accuracy: 0.93 - ETA: 1:11 - loss: 0.1800 - accuracy: 0.93 - ETA: 1:10 - loss: 0.1818 - accuracy: 0.93 - ETA: 1:09 - loss: 0.1810 - accuracy: 0.93 - ETA: 1:08 - loss: 0.1803 - accuracy: 0.93 - ETA: 1:08 - loss: 0.1825 - accuracy: 0.93 - ETA: 1:07 - loss: 0.1828 - accuracy: 0.93 - ETA: 1:06 - loss: 0.1856 - accuracy: 0.93 - ETA: 1:06 - loss: 0.1844 - accuracy: 0.93 - ETA: 1:05 - loss: 0.1848 - accuracy: 0.93 - ETA: 1:04 - loss: 0.1852 - accuracy: 0.93 - ETA: 1:03 - loss: 0.1871 - accuracy: 0.93 - ETA: 1:02 - loss: 0.1900 - accuracy: 0.93 - ETA: 1:02 - loss: 0.1912 - accuracy: 0.93 - ETA: 1:01 - loss: 0.1897 - accuracy: 0.93 - ETA: 1:00 - loss: 0.1913 - accuracy: 0.93 - ETA: 1:00 - loss: 0.1895 - accuracy: 0.93 - ETA: 59s - loss: 0.1889 - accuracy: 0.9311 - ETA: 58s - loss: 0.1876 - accuracy: 0.931 - ETA: 57s - loss: 0.1900 - accuracy: 0.930 - ETA: 57s - loss: 0.1891 - accuracy: 0.931 - ETA: 56s - loss: 0.1874 - accuracy: 0.932 - ETA: 55s - loss: 0.1880 - accuracy: 0.932 - ETA: 54s - loss: 0.1886 - accuracy: 0.931 - ETA: 54s - loss: 0.1892 - accuracy: 0.930 - ETA: 53s - loss: 0.1905 - accuracy: 0.930 - ETA: 52s - loss: 0.1911 - accuracy: 0.929 - ETA: 51s - loss: 0.1921 - accuracy: 0.929 - ETA: 50s - loss: 0.1918 - accuracy: 0.929 - ETA: 50s - loss: 0.1908 - accuracy: 0.929 - ETA: 49s - loss: 0.1908 - accuracy: 0.929 - ETA: 48s - loss: 0.1899 - accuracy: 0.929 - ETA: 47s - loss: 0.1900 - accuracy: 0.929 - ETA: 46s - loss: 0.1903 - accuracy: 0.928 - ETA: 46s - loss: 0.1908 - accuracy: 0.928 - ETA: 45s - loss: 0.1929 - accuracy: 0.928 - ETA: 44s - loss: 0.1935 - accuracy: 0.927 - ETA: 43s - loss: 0.1938 - accuracy: 0.927 - ETA: 43s - loss: 0.1940 - accuracy: 0.927 - ETA: 42s - loss: 0.1941 - accuracy: 0.927 - ETA: 41s - loss: 0.1940 - accuracy: 0.927 - ETA: 40s - loss: 0.1942 - accuracy: 0.927 - ETA: 39s - loss: 0.1954 - accuracy: 0.927 - ETA: 39s - loss: 0.1954 - accuracy: 0.927 - ETA: 38s - loss: 0.1953 - accuracy: 0.927 - ETA: 37s - loss: 0.1944 - accuracy: 0.927 - ETA: 36s - loss: 0.1942 - accuracy: 0.927 - ETA: 36s - loss: 0.1943 - accuracy: 0.927 - ETA: 35s - loss: 0.1941 - accuracy: 0.927 - ETA: 34s - loss: 0.1936 - accuracy: 0.927 - ETA: 33s - loss: 0.1934 - accuracy: 0.927 - ETA: 33s - loss: 0.1925 - accuracy: 0.927 - ETA: 32s - loss: 0.1932 - accuracy: 0.926 - ETA: 31s - loss: 0.1933 - accuracy: 0.926 - ETA: 30s - loss: 0.1929 - accuracy: 0.926 - ETA: 29s - loss: 0.1934 - accuracy: 0.926 - ETA: 29s - loss: 0.1926 - accuracy: 0.926 - ETA: 28s - loss: 0.1931 - accuracy: 0.926 - ETA: 27s - loss: 0.1934 - accuracy: 0.925 - ETA: 26s - loss: 0.1929 - accuracy: 0.926 - ETA: 25s - loss: 0.1924 - accuracy: 0.926 - ETA: 25s - loss: 0.1924 - accuracy: 0.926 - ETA: 24s - loss: 0.1923 - accuracy: 0.926 - ETA: 23s - loss: 0.1921 - accuracy: 0.926 - ETA: 22s - loss: 0.1920 - accuracy: 0.926 - ETA: 22s - loss: 0.1925 - accuracy: 0.926 - ETA: 21s - loss: 0.1925 - accuracy: 0.926 - ETA: 20s - loss: 0.1931 - accuracy: 0.926 - ETA: 19s - loss: 0.1935 - accuracy: 0.926 - ETA: 18s - loss: 0.1931 - accuracy: 0.926 - ETA: 18s - loss: 0.1937 - accuracy: 0.926 - ETA: 17s - loss: 0.1941 - accuracy: 0.926 - ETA: 16s - loss: 0.1941 - accuracy: 0.926 - ETA: 15s - loss: 0.1944 - accuracy: 0.926 - ETA: 14s - loss: 0.1944 - accuracy: 0.926 - ETA: 14s - loss: 0.1945 - accuracy: 0.926 - ETA: 13s - loss: 0.1940 - accuracy: 0.926 - ETA: 12s - loss: 0.1935 - accuracy: 0.926 - ETA: 11s - loss: 0.1932 - accuracy: 0.926 - ETA: 11s - loss: 0.1932 - accuracy: 0.926 - ETA: 10s - loss: 0.1931 - accuracy: 0.926 - ETA: 9s - loss: 0.1937 - accuracy: 0.926 - ETA: 8s - loss: 0.1938 - accuracy: 0.92 - ETA: 7s - loss: 0.1943 - accuracy: 0.92 - ETA: 7s - loss: 0.1951 - accuracy: 0.92 - ETA: 6s - loss: 0.1955 - accuracy: 0.92 - ETA: 5s - loss: 0.1962 - accuracy: 0.92 - ETA: 4s - loss: 0.1974 - accuracy: 0.92 - ETA: 3s - loss: 0.1979 - accuracy: 0.92 - ETA: 3s - loss: 0.1982 - accuracy: 0.92 - ETA: 2s - loss: 0.1981 - accuracy: 0.92 - ETA: 1s - loss: 0.1982 - accuracy: 0.92 - ETA: 0s - loss: 0.1982 - accuracy: 0.92 - 85s 3ms/step - loss: 0.1989 - accuracy: 0.9241 - val_loss: 0.3447 - val_accuracy: 0.8611\n",
      "Epoch 12/30\n",
      "30000/30000 [==============================] - ETA: 1:18 - loss: 0.3340 - accuracy: 0.84 - ETA: 1:16 - loss: 0.3089 - accuracy: 0.86 - ETA: 1:15 - loss: 0.2718 - accuracy: 0.87 - ETA: 1:14 - loss: 0.2602 - accuracy: 0.88 - ETA: 1:13 - loss: 0.2526 - accuracy: 0.89 - ETA: 1:13 - loss: 0.2534 - accuracy: 0.89 - ETA: 1:12 - loss: 0.2459 - accuracy: 0.90 - ETA: 1:11 - loss: 0.2471 - accuracy: 0.89 - ETA: 1:10 - loss: 0.2422 - accuracy: 0.89 - ETA: 1:10 - loss: 0.2363 - accuracy: 0.90 - ETA: 1:08 - loss: 0.2281 - accuracy: 0.90 - ETA: 1:08 - loss: 0.2293 - accuracy: 0.90 - ETA: 1:07 - loss: 0.2274 - accuracy: 0.90 - ETA: 1:06 - loss: 0.2256 - accuracy: 0.90 - ETA: 1:06 - loss: 0.2268 - accuracy: 0.90 - ETA: 1:05 - loss: 0.2233 - accuracy: 0.90 - ETA: 1:05 - loss: 0.2220 - accuracy: 0.91 - ETA: 1:04 - loss: 0.2195 - accuracy: 0.91 - ETA: 1:04 - loss: 0.2170 - accuracy: 0.91 - ETA: 1:03 - loss: 0.2171 - accuracy: 0.91 - ETA: 1:02 - loss: 0.2140 - accuracy: 0.91 - ETA: 1:01 - loss: 0.2130 - accuracy: 0.91 - ETA: 1:00 - loss: 0.2122 - accuracy: 0.91 - ETA: 59s - loss: 0.2091 - accuracy: 0.9161 - ETA: 59s - loss: 0.2069 - accuracy: 0.917 - ETA: 58s - loss: 0.2051 - accuracy: 0.918 - ETA: 57s - loss: 0.2040 - accuracy: 0.919 - ETA: 56s - loss: 0.2021 - accuracy: 0.919 - ETA: 55s - loss: 0.2015 - accuracy: 0.920 - ETA: 55s - loss: 0.2000 - accuracy: 0.920 - ETA: 54s - loss: 0.1984 - accuracy: 0.921 - ETA: 53s - loss: 0.1973 - accuracy: 0.922 - ETA: 52s - loss: 0.1960 - accuracy: 0.923 - ETA: 51s - loss: 0.1970 - accuracy: 0.922 - ETA: 51s - loss: 0.1972 - accuracy: 0.922 - ETA: 50s - loss: 0.1983 - accuracy: 0.922 - ETA: 49s - loss: 0.1981 - accuracy: 0.921 - ETA: 48s - loss: 0.1987 - accuracy: 0.921 - ETA: 47s - loss: 0.1979 - accuracy: 0.922 - ETA: 47s - loss: 0.2001 - accuracy: 0.921 - ETA: 46s - loss: 0.1998 - accuracy: 0.921 - ETA: 45s - loss: 0.1998 - accuracy: 0.921 - ETA: 44s - loss: 0.1997 - accuracy: 0.921 - ETA: 43s - loss: 0.1987 - accuracy: 0.922 - ETA: 43s - loss: 0.1978 - accuracy: 0.922 - ETA: 42s - loss: 0.1975 - accuracy: 0.922 - ETA: 41s - loss: 0.1962 - accuracy: 0.922 - ETA: 40s - loss: 0.1959 - accuracy: 0.922 - ETA: 39s - loss: 0.1954 - accuracy: 0.923 - ETA: 39s - loss: 0.1947 - accuracy: 0.923 - ETA: 38s - loss: 0.1938 - accuracy: 0.924 - ETA: 37s - loss: 0.1951 - accuracy: 0.923 - ETA: 36s - loss: 0.1945 - accuracy: 0.923 - ETA: 35s - loss: 0.1945 - accuracy: 0.923 - ETA: 35s - loss: 0.1953 - accuracy: 0.923 - ETA: 34s - loss: 0.1960 - accuracy: 0.922 - ETA: 33s - loss: 0.1958 - accuracy: 0.922 - ETA: 32s - loss: 0.1957 - accuracy: 0.922 - ETA: 32s - loss: 0.1962 - accuracy: 0.922 - ETA: 31s - loss: 0.1974 - accuracy: 0.921 - ETA: 30s - loss: 0.1967 - accuracy: 0.922 - ETA: 29s - loss: 0.1962 - accuracy: 0.922 - ETA: 28s - loss: 0.1975 - accuracy: 0.922 - ETA: 28s - loss: 0.1976 - accuracy: 0.922 - ETA: 27s - loss: 0.1972 - accuracy: 0.923 - ETA: 26s - loss: 0.1981 - accuracy: 0.922 - ETA: 25s - loss: 0.1988 - accuracy: 0.922 - ETA: 24s - loss: 0.1986 - accuracy: 0.922 - ETA: 24s - loss: 0.1983 - accuracy: 0.922 - ETA: 23s - loss: 0.1993 - accuracy: 0.922 - ETA: 22s - loss: 0.1993 - accuracy: 0.922 - ETA: 21s - loss: 0.1992 - accuracy: 0.922 - ETA: 21s - loss: 0.1985 - accuracy: 0.922 - ETA: 20s - loss: 0.1984 - accuracy: 0.923 - ETA: 19s - loss: 0.1979 - accuracy: 0.923 - ETA: 18s - loss: 0.1978 - accuracy: 0.923 - ETA: 17s - loss: 0.1972 - accuracy: 0.923 - ETA: 17s - loss: 0.1976 - accuracy: 0.923 - ETA: 16s - loss: 0.1974 - accuracy: 0.923 - ETA: 15s - loss: 0.1972 - accuracy: 0.923 - ETA: 14s - loss: 0.1973 - accuracy: 0.923 - ETA: 14s - loss: 0.1981 - accuracy: 0.922 - ETA: 13s - loss: 0.1974 - accuracy: 0.923 - ETA: 12s - loss: 0.1978 - accuracy: 0.922 - ETA: 11s - loss: 0.1984 - accuracy: 0.922 - ETA: 10s - loss: 0.1981 - accuracy: 0.922 - ETA: 10s - loss: 0.1981 - accuracy: 0.922 - ETA: 9s - loss: 0.1980 - accuracy: 0.922 - ETA: 8s - loss: 0.1978 - accuracy: 0.92 - ETA: 7s - loss: 0.1979 - accuracy: 0.92 - ETA: 7s - loss: 0.1980 - accuracy: 0.92 - ETA: 6s - loss: 0.1982 - accuracy: 0.92 - ETA: 5s - loss: 0.1981 - accuracy: 0.92 - ETA: 4s - loss: 0.1976 - accuracy: 0.92 - ETA: 3s - loss: 0.1975 - accuracy: 0.92 - ETA: 3s - loss: 0.1973 - accuracy: 0.92 - ETA: 2s - loss: 0.1970 - accuracy: 0.92 - ETA: 1s - loss: 0.1964 - accuracy: 0.92 - ETA: 0s - loss: 0.1960 - accuracy: 0.92 - 84s 3ms/step - loss: 0.1953 - accuracy: 0.9238 - val_loss: 0.2771 - val_accuracy: 0.8978\n",
      "Epoch 13/30\n",
      "30000/30000 [==============================] - ETA: 1:21 - loss: 0.0970 - accuracy: 0.97 - ETA: 1:21 - loss: 0.1380 - accuracy: 0.95 - ETA: 1:19 - loss: 0.1373 - accuracy: 0.95 - ETA: 1:19 - loss: 0.1497 - accuracy: 0.94 - ETA: 1:18 - loss: 0.1410 - accuracy: 0.94 - ETA: 1:16 - loss: 0.1350 - accuracy: 0.95 - ETA: 1:15 - loss: 0.1408 - accuracy: 0.94 - ETA: 1:14 - loss: 0.1414 - accuracy: 0.94 - ETA: 1:13 - loss: 0.1414 - accuracy: 0.94 - ETA: 1:11 - loss: 0.1445 - accuracy: 0.94 - ETA: 1:10 - loss: 0.1451 - accuracy: 0.94 - ETA: 1:09 - loss: 0.1510 - accuracy: 0.94 - ETA: 1:08 - loss: 0.1530 - accuracy: 0.94 - ETA: 1:08 - loss: 0.1516 - accuracy: 0.94 - ETA: 1:07 - loss: 0.1498 - accuracy: 0.94 - ETA: 1:06 - loss: 0.1458 - accuracy: 0.94 - ETA: 1:05 - loss: 0.1514 - accuracy: 0.94 - ETA: 1:05 - loss: 0.1528 - accuracy: 0.94 - ETA: 1:04 - loss: 0.1555 - accuracy: 0.94 - ETA: 1:03 - loss: 0.1563 - accuracy: 0.94 - ETA: 1:02 - loss: 0.1577 - accuracy: 0.94 - ETA: 1:01 - loss: 0.1577 - accuracy: 0.94 - ETA: 1:00 - loss: 0.1603 - accuracy: 0.94 - ETA: 59s - loss: 0.1609 - accuracy: 0.9407 - ETA: 58s - loss: 0.1608 - accuracy: 0.940 - ETA: 58s - loss: 0.1631 - accuracy: 0.939 - ETA: 57s - loss: 0.1620 - accuracy: 0.940 - ETA: 56s - loss: 0.1618 - accuracy: 0.940 - ETA: 55s - loss: 0.1613 - accuracy: 0.940 - ETA: 54s - loss: 0.1633 - accuracy: 0.940 - ETA: 54s - loss: 0.1646 - accuracy: 0.940 - ETA: 53s - loss: 0.1646 - accuracy: 0.940 - ETA: 52s - loss: 0.1658 - accuracy: 0.939 - ETA: 51s - loss: 0.1653 - accuracy: 0.940 - ETA: 50s - loss: 0.1655 - accuracy: 0.939 - ETA: 50s - loss: 0.1660 - accuracy: 0.939 - ETA: 49s - loss: 0.1648 - accuracy: 0.939 - ETA: 48s - loss: 0.1654 - accuracy: 0.939 - ETA: 47s - loss: 0.1659 - accuracy: 0.939 - ETA: 46s - loss: 0.1662 - accuracy: 0.939 - ETA: 46s - loss: 0.1660 - accuracy: 0.938 - ETA: 45s - loss: 0.1667 - accuracy: 0.938 - ETA: 44s - loss: 0.1672 - accuracy: 0.938 - ETA: 43s - loss: 0.1680 - accuracy: 0.937 - ETA: 43s - loss: 0.1674 - accuracy: 0.937 - ETA: 42s - loss: 0.1670 - accuracy: 0.938 - ETA: 41s - loss: 0.1689 - accuracy: 0.937 - ETA: 40s - loss: 0.1707 - accuracy: 0.936 - ETA: 39s - loss: 0.1707 - accuracy: 0.936 - ETA: 38s - loss: 0.1702 - accuracy: 0.937 - ETA: 38s - loss: 0.1709 - accuracy: 0.936 - ETA: 37s - loss: 0.1704 - accuracy: 0.936 - ETA: 36s - loss: 0.1699 - accuracy: 0.937 - ETA: 35s - loss: 0.1704 - accuracy: 0.936 - ETA: 35s - loss: 0.1706 - accuracy: 0.936 - ETA: 34s - loss: 0.1707 - accuracy: 0.936 - ETA: 33s - loss: 0.1721 - accuracy: 0.936 - ETA: 32s - loss: 0.1716 - accuracy: 0.936 - ETA: 31s - loss: 0.1717 - accuracy: 0.936 - ETA: 31s - loss: 0.1718 - accuracy: 0.936 - ETA: 30s - loss: 0.1723 - accuracy: 0.935 - ETA: 29s - loss: 0.1720 - accuracy: 0.936 - ETA: 28s - loss: 0.1727 - accuracy: 0.935 - ETA: 28s - loss: 0.1722 - accuracy: 0.936 - ETA: 27s - loss: 0.1716 - accuracy: 0.936 - ETA: 26s - loss: 0.1721 - accuracy: 0.936 - ETA: 25s - loss: 0.1721 - accuracy: 0.936 - ETA: 24s - loss: 0.1724 - accuracy: 0.936 - ETA: 24s - loss: 0.1730 - accuracy: 0.936 - ETA: 23s - loss: 0.1734 - accuracy: 0.935 - ETA: 22s - loss: 0.1735 - accuracy: 0.935 - ETA: 21s - loss: 0.1732 - accuracy: 0.935 - ETA: 21s - loss: 0.1734 - accuracy: 0.935 - ETA: 20s - loss: 0.1732 - accuracy: 0.935 - ETA: 19s - loss: 0.1729 - accuracy: 0.935 - ETA: 18s - loss: 0.1732 - accuracy: 0.935 - ETA: 17s - loss: 0.1729 - accuracy: 0.935 - ETA: 17s - loss: 0.1725 - accuracy: 0.936 - ETA: 16s - loss: 0.1725 - accuracy: 0.935 - ETA: 15s - loss: 0.1724 - accuracy: 0.936 - ETA: 14s - loss: 0.1726 - accuracy: 0.936 - ETA: 13s - loss: 0.1732 - accuracy: 0.936 - ETA: 13s - loss: 0.1731 - accuracy: 0.935 - ETA: 12s - loss: 0.1729 - accuracy: 0.936 - ETA: 11s - loss: 0.1731 - accuracy: 0.935 - ETA: 10s - loss: 0.1737 - accuracy: 0.935 - ETA: 10s - loss: 0.1737 - accuracy: 0.935 - ETA: 9s - loss: 0.1736 - accuracy: 0.935 - ETA: 8s - loss: 0.1736 - accuracy: 0.93 - ETA: 7s - loss: 0.1734 - accuracy: 0.93 - ETA: 7s - loss: 0.1731 - accuracy: 0.93 - ETA: 6s - loss: 0.1727 - accuracy: 0.93 - ETA: 5s - loss: 0.1731 - accuracy: 0.93 - ETA: 4s - loss: 0.1729 - accuracy: 0.93 - ETA: 3s - loss: 0.1729 - accuracy: 0.93 - ETA: 3s - loss: 0.1723 - accuracy: 0.93 - ETA: 2s - loss: 0.1728 - accuracy: 0.93 - ETA: 1s - loss: 0.1723 - accuracy: 0.93 - ETA: 0s - loss: 0.1720 - accuracy: 0.93 - 84s 3ms/step - loss: 0.1721 - accuracy: 0.9362 - val_loss: 0.2893 - val_accuracy: 0.8901\n",
      "Epoch 14/30\n",
      "30000/30000 [==============================] - ETA: 1:18 - loss: 0.1144 - accuracy: 0.96 - ETA: 1:16 - loss: 0.1306 - accuracy: 0.95 - ETA: 1:17 - loss: 0.1390 - accuracy: 0.95 - ETA: 1:17 - loss: 0.1530 - accuracy: 0.94 - ETA: 1:15 - loss: 0.1494 - accuracy: 0.95 - ETA: 1:14 - loss: 0.1500 - accuracy: 0.95 - ETA: 1:13 - loss: 0.1477 - accuracy: 0.95 - ETA: 1:12 - loss: 0.1412 - accuracy: 0.95 - ETA: 1:11 - loss: 0.1479 - accuracy: 0.95 - ETA: 1:10 - loss: 0.1456 - accuracy: 0.95 - ETA: 1:10 - loss: 0.1469 - accuracy: 0.95 - ETA: 1:09 - loss: 0.1489 - accuracy: 0.95 - ETA: 1:08 - loss: 0.1509 - accuracy: 0.95 - ETA: 1:07 - loss: 0.1477 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1448 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1444 - accuracy: 0.95 - ETA: 1:05 - loss: 0.1437 - accuracy: 0.95 - ETA: 1:04 - loss: 0.1446 - accuracy: 0.94 - ETA: 1:04 - loss: 0.1436 - accuracy: 0.94 - ETA: 1:03 - loss: 0.1425 - accuracy: 0.94 - ETA: 1:02 - loss: 0.1423 - accuracy: 0.94 - ETA: 1:01 - loss: 0.1412 - accuracy: 0.94 - ETA: 1:01 - loss: 0.1414 - accuracy: 0.94 - ETA: 1:00 - loss: 0.1410 - accuracy: 0.94 - ETA: 59s - loss: 0.1414 - accuracy: 0.9497 - ETA: 58s - loss: 0.1413 - accuracy: 0.949 - ETA: 57s - loss: 0.1415 - accuracy: 0.949 - ETA: 56s - loss: 0.1420 - accuracy: 0.949 - ETA: 56s - loss: 0.1414 - accuracy: 0.950 - ETA: 55s - loss: 0.1411 - accuracy: 0.950 - ETA: 54s - loss: 0.1417 - accuracy: 0.950 - ETA: 53s - loss: 0.1406 - accuracy: 0.950 - ETA: 52s - loss: 0.1407 - accuracy: 0.950 - ETA: 52s - loss: 0.1410 - accuracy: 0.950 - ETA: 51s - loss: 0.1407 - accuracy: 0.950 - ETA: 50s - loss: 0.1399 - accuracy: 0.950 - ETA: 49s - loss: 0.1390 - accuracy: 0.951 - ETA: 49s - loss: 0.1391 - accuracy: 0.951 - ETA: 48s - loss: 0.1392 - accuracy: 0.950 - ETA: 47s - loss: 0.1398 - accuracy: 0.950 - ETA: 46s - loss: 0.1394 - accuracy: 0.951 - ETA: 45s - loss: 0.1402 - accuracy: 0.950 - ETA: 45s - loss: 0.1403 - accuracy: 0.950 - ETA: 44s - loss: 0.1398 - accuracy: 0.950 - ETA: 43s - loss: 0.1402 - accuracy: 0.950 - ETA: 42s - loss: 0.1404 - accuracy: 0.950 - ETA: 41s - loss: 0.1409 - accuracy: 0.950 - ETA: 41s - loss: 0.1409 - accuracy: 0.950 - ETA: 40s - loss: 0.1415 - accuracy: 0.949 - ETA: 39s - loss: 0.1407 - accuracy: 0.950 - ETA: 38s - loss: 0.1414 - accuracy: 0.949 - ETA: 37s - loss: 0.1415 - accuracy: 0.949 - ETA: 36s - loss: 0.1410 - accuracy: 0.949 - ETA: 36s - loss: 0.1403 - accuracy: 0.950 - ETA: 35s - loss: 0.1408 - accuracy: 0.950 - ETA: 34s - loss: 0.1404 - accuracy: 0.949 - ETA: 33s - loss: 0.1405 - accuracy: 0.949 - ETA: 32s - loss: 0.1417 - accuracy: 0.949 - ETA: 32s - loss: 0.1416 - accuracy: 0.949 - ETA: 31s - loss: 0.1434 - accuracy: 0.948 - ETA: 30s - loss: 0.1434 - accuracy: 0.948 - ETA: 29s - loss: 0.1442 - accuracy: 0.948 - ETA: 29s - loss: 0.1452 - accuracy: 0.947 - ETA: 28s - loss: 0.1457 - accuracy: 0.947 - ETA: 27s - loss: 0.1461 - accuracy: 0.947 - ETA: 26s - loss: 0.1461 - accuracy: 0.947 - ETA: 25s - loss: 0.1470 - accuracy: 0.946 - ETA: 25s - loss: 0.1472 - accuracy: 0.946 - ETA: 24s - loss: 0.1473 - accuracy: 0.946 - ETA: 23s - loss: 0.1469 - accuracy: 0.947 - ETA: 22s - loss: 0.1470 - accuracy: 0.947 - ETA: 21s - loss: 0.1470 - accuracy: 0.947 - ETA: 21s - loss: 0.1470 - accuracy: 0.947 - ETA: 20s - loss: 0.1473 - accuracy: 0.946 - ETA: 19s - loss: 0.1467 - accuracy: 0.947 - ETA: 18s - loss: 0.1473 - accuracy: 0.946 - ETA: 18s - loss: 0.1476 - accuracy: 0.946 - ETA: 17s - loss: 0.1472 - accuracy: 0.946 - ETA: 16s - loss: 0.1481 - accuracy: 0.946 - ETA: 15s - loss: 0.1493 - accuracy: 0.945 - ETA: 14s - loss: 0.1496 - accuracy: 0.945 - ETA: 14s - loss: 0.1494 - accuracy: 0.945 - ETA: 13s - loss: 0.1498 - accuracy: 0.945 - ETA: 12s - loss: 0.1499 - accuracy: 0.945 - ETA: 11s - loss: 0.1498 - accuracy: 0.945 - ETA: 10s - loss: 0.1502 - accuracy: 0.945 - ETA: 10s - loss: 0.1502 - accuracy: 0.945 - ETA: 9s - loss: 0.1508 - accuracy: 0.945 - ETA: 8s - loss: 0.1511 - accuracy: 0.94 - ETA: 7s - loss: 0.1510 - accuracy: 0.94 - ETA: 7s - loss: 0.1513 - accuracy: 0.94 - ETA: 6s - loss: 0.1518 - accuracy: 0.94 - ETA: 5s - loss: 0.1518 - accuracy: 0.94 - ETA: 4s - loss: 0.1517 - accuracy: 0.94 - ETA: 3s - loss: 0.1522 - accuracy: 0.94 - ETA: 3s - loss: 0.1523 - accuracy: 0.94 - ETA: 2s - loss: 0.1520 - accuracy: 0.94 - ETA: 1s - loss: 0.1520 - accuracy: 0.94 - ETA: 0s - loss: 0.1518 - accuracy: 0.94 - 84s 3ms/step - loss: 0.1518 - accuracy: 0.9445 - val_loss: 0.2930 - val_accuracy: 0.8931\n",
      "Epoch 15/30\n",
      "30000/30000 [==============================] - ETA: 1:23 - loss: 0.1418 - accuracy: 0.93 - ETA: 1:18 - loss: 0.1208 - accuracy: 0.95 - ETA: 1:15 - loss: 0.1304 - accuracy: 0.94 - ETA: 1:14 - loss: 0.1400 - accuracy: 0.94 - ETA: 1:14 - loss: 0.1367 - accuracy: 0.94 - ETA: 1:14 - loss: 0.1368 - accuracy: 0.94 - ETA: 1:13 - loss: 0.1413 - accuracy: 0.94 - ETA: 1:11 - loss: 0.1429 - accuracy: 0.94 - ETA: 1:11 - loss: 0.1376 - accuracy: 0.94 - ETA: 1:10 - loss: 0.1363 - accuracy: 0.94 - ETA: 1:09 - loss: 0.1340 - accuracy: 0.94 - ETA: 1:09 - loss: 0.1311 - accuracy: 0.95 - ETA: 1:08 - loss: 0.1323 - accuracy: 0.95 - ETA: 1:07 - loss: 0.1312 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1315 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1304 - accuracy: 0.95 - ETA: 1:05 - loss: 0.1328 - accuracy: 0.94 - ETA: 1:04 - loss: 0.1351 - accuracy: 0.94 - ETA: 1:04 - loss: 0.1373 - accuracy: 0.94 - ETA: 1:03 - loss: 0.1376 - accuracy: 0.94 - ETA: 1:02 - loss: 0.1350 - accuracy: 0.94 - ETA: 1:01 - loss: 0.1351 - accuracy: 0.94 - ETA: 1:00 - loss: 0.1355 - accuracy: 0.94 - ETA: 59s - loss: 0.1355 - accuracy: 0.9475 - ETA: 59s - loss: 0.1361 - accuracy: 0.947 - ETA: 58s - loss: 0.1379 - accuracy: 0.946 - ETA: 57s - loss: 0.1344 - accuracy: 0.948 - ETA: 56s - loss: 0.1354 - accuracy: 0.948 - ETA: 55s - loss: 0.1353 - accuracy: 0.948 - ETA: 54s - loss: 0.1359 - accuracy: 0.948 - ETA: 54s - loss: 0.1362 - accuracy: 0.948 - ETA: 53s - loss: 0.1357 - accuracy: 0.949 - ETA: 52s - loss: 0.1347 - accuracy: 0.949 - ETA: 51s - loss: 0.1349 - accuracy: 0.949 - ETA: 51s - loss: 0.1346 - accuracy: 0.949 - ETA: 50s - loss: 0.1340 - accuracy: 0.950 - ETA: 49s - loss: 0.1339 - accuracy: 0.950 - ETA: 48s - loss: 0.1333 - accuracy: 0.950 - ETA: 48s - loss: 0.1345 - accuracy: 0.949 - ETA: 47s - loss: 0.1339 - accuracy: 0.950 - ETA: 46s - loss: 0.1346 - accuracy: 0.950 - ETA: 45s - loss: 0.1339 - accuracy: 0.950 - ETA: 44s - loss: 0.1345 - accuracy: 0.949 - ETA: 43s - loss: 0.1343 - accuracy: 0.949 - ETA: 43s - loss: 0.1344 - accuracy: 0.950 - ETA: 42s - loss: 0.1336 - accuracy: 0.950 - ETA: 41s - loss: 0.1331 - accuracy: 0.950 - ETA: 40s - loss: 0.1333 - accuracy: 0.950 - ETA: 39s - loss: 0.1333 - accuracy: 0.950 - ETA: 39s - loss: 0.1339 - accuracy: 0.950 - ETA: 38s - loss: 0.1343 - accuracy: 0.950 - ETA: 37s - loss: 0.1344 - accuracy: 0.950 - ETA: 37s - loss: 0.1342 - accuracy: 0.950 - ETA: 36s - loss: 0.1356 - accuracy: 0.950 - ETA: 35s - loss: 0.1358 - accuracy: 0.950 - ETA: 34s - loss: 0.1358 - accuracy: 0.950 - ETA: 33s - loss: 0.1386 - accuracy: 0.949 - ETA: 33s - loss: 0.1384 - accuracy: 0.949 - ETA: 32s - loss: 0.1389 - accuracy: 0.949 - ETA: 31s - loss: 0.1389 - accuracy: 0.949 - ETA: 30s - loss: 0.1385 - accuracy: 0.949 - ETA: 29s - loss: 0.1386 - accuracy: 0.949 - ETA: 29s - loss: 0.1396 - accuracy: 0.948 - ETA: 28s - loss: 0.1397 - accuracy: 0.948 - ETA: 27s - loss: 0.1415 - accuracy: 0.947 - ETA: 26s - loss: 0.1430 - accuracy: 0.947 - ETA: 25s - loss: 0.1427 - accuracy: 0.947 - ETA: 25s - loss: 0.1438 - accuracy: 0.946 - ETA: 24s - loss: 0.1453 - accuracy: 0.945 - ETA: 23s - loss: 0.1460 - accuracy: 0.945 - ETA: 22s - loss: 0.1467 - accuracy: 0.945 - ETA: 21s - loss: 0.1473 - accuracy: 0.944 - ETA: 21s - loss: 0.1475 - accuracy: 0.944 - ETA: 20s - loss: 0.1479 - accuracy: 0.944 - ETA: 19s - loss: 0.1476 - accuracy: 0.944 - ETA: 18s - loss: 0.1475 - accuracy: 0.944 - ETA: 18s - loss: 0.1477 - accuracy: 0.944 - ETA: 17s - loss: 0.1482 - accuracy: 0.944 - ETA: 16s - loss: 0.1486 - accuracy: 0.944 - ETA: 15s - loss: 0.1479 - accuracy: 0.944 - ETA: 14s - loss: 0.1479 - accuracy: 0.944 - ETA: 14s - loss: 0.1480 - accuracy: 0.944 - ETA: 13s - loss: 0.1482 - accuracy: 0.944 - ETA: 12s - loss: 0.1484 - accuracy: 0.944 - ETA: 11s - loss: 0.1487 - accuracy: 0.944 - ETA: 10s - loss: 0.1487 - accuracy: 0.944 - ETA: 10s - loss: 0.1488 - accuracy: 0.944 - ETA: 9s - loss: 0.1488 - accuracy: 0.944 - ETA: 8s - loss: 0.1493 - accuracy: 0.94 - ETA: 7s - loss: 0.1492 - accuracy: 0.94 - ETA: 7s - loss: 0.1491 - accuracy: 0.94 - ETA: 6s - loss: 0.1491 - accuracy: 0.94 - ETA: 5s - loss: 0.1491 - accuracy: 0.94 - ETA: 4s - loss: 0.1483 - accuracy: 0.94 - ETA: 3s - loss: 0.1484 - accuracy: 0.94 - ETA: 3s - loss: 0.1482 - accuracy: 0.94 - ETA: 2s - loss: 0.1488 - accuracy: 0.94 - ETA: 1s - loss: 0.1481 - accuracy: 0.94 - ETA: 0s - loss: 0.1484 - accuracy: 0.94 - 84s 3ms/step - loss: 0.1482 - accuracy: 0.9450 - val_loss: 0.2862 - val_accuracy: 0.8940\n",
      "Epoch 16/30\n",
      "30000/30000 [==============================] - ETA: 1:18 - loss: 0.1341 - accuracy: 0.97 - ETA: 1:16 - loss: 0.1573 - accuracy: 0.94 - ETA: 1:17 - loss: 0.1490 - accuracy: 0.94 - ETA: 1:15 - loss: 0.1402 - accuracy: 0.95 - ETA: 1:13 - loss: 0.1306 - accuracy: 0.95 - ETA: 1:12 - loss: 0.1362 - accuracy: 0.95 - ETA: 1:12 - loss: 0.1414 - accuracy: 0.95 - ETA: 1:11 - loss: 0.1418 - accuracy: 0.95 - ETA: 1:10 - loss: 0.1372 - accuracy: 0.95 - ETA: 1:09 - loss: 0.1344 - accuracy: 0.95 - ETA: 1:08 - loss: 0.1329 - accuracy: 0.95 - ETA: 1:07 - loss: 0.1320 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1303 - accuracy: 0.95 - ETA: 1:06 - loss: 0.1288 - accuracy: 0.95 - ETA: 1:05 - loss: 0.1280 - accuracy: 0.95 - ETA: 1:04 - loss: 0.1262 - accuracy: 0.95 - ETA: 1:03 - loss: 0.1256 - accuracy: 0.95 - ETA: 1:03 - loss: 0.1233 - accuracy: 0.95 - ETA: 1:02 - loss: 0.1240 - accuracy: 0.95 - ETA: 1:02 - loss: 0.1242 - accuracy: 0.95 - ETA: 1:02 - loss: 0.1241 - accuracy: 0.95 - ETA: 1:01 - loss: 0.1229 - accuracy: 0.95 - ETA: 1:00 - loss: 0.1230 - accuracy: 0.95 - ETA: 59s - loss: 0.1207 - accuracy: 0.9568 - ETA: 58s - loss: 0.1201 - accuracy: 0.957 - ETA: 58s - loss: 0.1193 - accuracy: 0.957 - ETA: 57s - loss: 0.1192 - accuracy: 0.957 - ETA: 56s - loss: 0.1210 - accuracy: 0.957 - ETA: 55s - loss: 0.1201 - accuracy: 0.957 - ETA: 54s - loss: 0.1189 - accuracy: 0.958 - ETA: 53s - loss: 0.1195 - accuracy: 0.957 - ETA: 53s - loss: 0.1199 - accuracy: 0.957 - ETA: 52s - loss: 0.1187 - accuracy: 0.958 - ETA: 51s - loss: 0.1186 - accuracy: 0.957 - ETA: 50s - loss: 0.1189 - accuracy: 0.957 - ETA: 50s - loss: 0.1186 - accuracy: 0.958 - ETA: 49s - loss: 0.1199 - accuracy: 0.957 - ETA: 48s - loss: 0.1201 - accuracy: 0.957 - ETA: 47s - loss: 0.1203 - accuracy: 0.957 - ETA: 47s - loss: 0.1209 - accuracy: 0.956 - ETA: 46s - loss: 0.1203 - accuracy: 0.957 - ETA: 45s - loss: 0.1209 - accuracy: 0.956 - ETA: 44s - loss: 0.1215 - accuracy: 0.956 - ETA: 43s - loss: 0.1223 - accuracy: 0.956 - ETA: 43s - loss: 0.1229 - accuracy: 0.955 - ETA: 42s - loss: 0.1231 - accuracy: 0.956 - ETA: 41s - loss: 0.1228 - accuracy: 0.956 - ETA: 40s - loss: 0.1224 - accuracy: 0.956 - ETA: 40s - loss: 0.1212 - accuracy: 0.956 - ETA: 39s - loss: 0.1207 - accuracy: 0.956 - ETA: 38s - loss: 0.1209 - accuracy: 0.956 - ETA: 37s - loss: 0.1203 - accuracy: 0.956 - ETA: 36s - loss: 0.1203 - accuracy: 0.956 - ETA: 36s - loss: 0.1203 - accuracy: 0.957 - ETA: 35s - loss: 0.1198 - accuracy: 0.957 - ETA: 34s - loss: 0.1190 - accuracy: 0.957 - ETA: 33s - loss: 0.1195 - accuracy: 0.957 - ETA: 32s - loss: 0.1196 - accuracy: 0.957 - ETA: 32s - loss: 0.1192 - accuracy: 0.958 - ETA: 31s - loss: 0.1204 - accuracy: 0.957 - ETA: 30s - loss: 0.1205 - accuracy: 0.957 - ETA: 29s - loss: 0.1209 - accuracy: 0.957 - ETA: 29s - loss: 0.1211 - accuracy: 0.957 - ETA: 28s - loss: 0.1207 - accuracy: 0.957 - ETA: 27s - loss: 0.1208 - accuracy: 0.957 - ETA: 26s - loss: 0.1207 - accuracy: 0.957 - ETA: 25s - loss: 0.1205 - accuracy: 0.957 - ETA: 25s - loss: 0.1209 - accuracy: 0.957 - ETA: 24s - loss: 0.1206 - accuracy: 0.957 - ETA: 23s - loss: 0.1207 - accuracy: 0.957 - ETA: 22s - loss: 0.1207 - accuracy: 0.957 - ETA: 21s - loss: 0.1208 - accuracy: 0.957 - ETA: 21s - loss: 0.1209 - accuracy: 0.957 - ETA: 20s - loss: 0.1207 - accuracy: 0.957 - ETA: 19s - loss: 0.1206 - accuracy: 0.957 - ETA: 18s - loss: 0.1206 - accuracy: 0.957 - ETA: 18s - loss: 0.1204 - accuracy: 0.957 - ETA: 17s - loss: 0.1201 - accuracy: 0.957 - ETA: 16s - loss: 0.1197 - accuracy: 0.957 - ETA: 15s - loss: 0.1197 - accuracy: 0.957 - ETA: 14s - loss: 0.1194 - accuracy: 0.957 - ETA: 14s - loss: 0.1194 - accuracy: 0.957 - ETA: 13s - loss: 0.1195 - accuracy: 0.957 - ETA: 12s - loss: 0.1197 - accuracy: 0.957 - ETA: 11s - loss: 0.1197 - accuracy: 0.957 - ETA: 10s - loss: 0.1200 - accuracy: 0.957 - ETA: 10s - loss: 0.1203 - accuracy: 0.957 - ETA: 9s - loss: 0.1210 - accuracy: 0.957 - ETA: 8s - loss: 0.1213 - accuracy: 0.95 - ETA: 7s - loss: 0.1218 - accuracy: 0.95 - ETA: 7s - loss: 0.1219 - accuracy: 0.95 - ETA: 6s - loss: 0.1222 - accuracy: 0.95 - ETA: 5s - loss: 0.1221 - accuracy: 0.95 - ETA: 4s - loss: 0.1230 - accuracy: 0.95 - ETA: 3s - loss: 0.1230 - accuracy: 0.95 - ETA: 3s - loss: 0.1229 - accuracy: 0.95 - ETA: 2s - loss: 0.1234 - accuracy: 0.95 - ETA: 1s - loss: 0.1234 - accuracy: 0.95 - ETA: 0s - loss: 0.1231 - accuracy: 0.95 - 84s 3ms/step - loss: 0.1228 - accuracy: 0.9566 - val_loss: 0.3075 - val_accuracy: 0.8922\n",
      "Epoch 17/30\n",
      "30000/30000 [==============================] - ETA: 1:16 - loss: 0.0843 - accuracy: 0.97 - ETA: 1:14 - loss: 0.0941 - accuracy: 0.96 - ETA: 1:15 - loss: 0.0876 - accuracy: 0.97 - ETA: 1:16 - loss: 0.0943 - accuracy: 0.96 - ETA: 1:14 - loss: 0.0889 - accuracy: 0.97 - ETA: 1:13 - loss: 0.0860 - accuracy: 0.97 - ETA: 1:12 - loss: 0.0868 - accuracy: 0.97 - ETA: 1:12 - loss: 0.0860 - accuracy: 0.97 - ETA: 1:11 - loss: 0.0900 - accuracy: 0.96 - ETA: 1:10 - loss: 0.0926 - accuracy: 0.96 - ETA: 1:09 - loss: 0.0939 - accuracy: 0.96 - ETA: 1:08 - loss: 0.0918 - accuracy: 0.96 - ETA: 1:08 - loss: 0.0909 - accuracy: 0.96 - ETA: 1:07 - loss: 0.0869 - accuracy: 0.97 - ETA: 1:06 - loss: 0.0876 - accuracy: 0.96 - ETA: 1:06 - loss: 0.0867 - accuracy: 0.96 - ETA: 1:05 - loss: 0.0872 - accuracy: 0.96 - ETA: 1:05 - loss: 0.0899 - accuracy: 0.96 - ETA: 1:04 - loss: 0.0921 - accuracy: 0.96 - ETA: 1:04 - loss: 0.0911 - accuracy: 0.96 - ETA: 1:03 - loss: 0.0931 - accuracy: 0.96 - ETA: 1:02 - loss: 0.0950 - accuracy: 0.96 - ETA: 1:01 - loss: 0.0953 - accuracy: 0.96 - ETA: 1:00 - loss: 0.0965 - accuracy: 0.96 - ETA: 59s - loss: 0.0965 - accuracy: 0.9675 - ETA: 58s - loss: 0.0958 - accuracy: 0.967 - ETA: 58s - loss: 0.0957 - accuracy: 0.967 - ETA: 57s - loss: 0.0952 - accuracy: 0.966 - ETA: 56s - loss: 0.0950 - accuracy: 0.967 - ETA: 55s - loss: 0.0946 - accuracy: 0.966 - ETA: 54s - loss: 0.0946 - accuracy: 0.966 - ETA: 53s - loss: 0.0958 - accuracy: 0.966 - ETA: 52s - loss: 0.0986 - accuracy: 0.965 - ETA: 52s - loss: 0.0995 - accuracy: 0.965 - ETA: 51s - loss: 0.0985 - accuracy: 0.966 - ETA: 50s - loss: 0.0997 - accuracy: 0.965 - ETA: 49s - loss: 0.1010 - accuracy: 0.965 - ETA: 48s - loss: 0.1018 - accuracy: 0.965 - ETA: 48s - loss: 0.1042 - accuracy: 0.964 - ETA: 47s - loss: 0.1050 - accuracy: 0.963 - ETA: 46s - loss: 0.1064 - accuracy: 0.963 - ETA: 45s - loss: 0.1079 - accuracy: 0.962 - ETA: 44s - loss: 0.1077 - accuracy: 0.962 - ETA: 44s - loss: 0.1089 - accuracy: 0.962 - ETA: 43s - loss: 0.1097 - accuracy: 0.961 - ETA: 42s - loss: 0.1098 - accuracy: 0.961 - ETA: 41s - loss: 0.1093 - accuracy: 0.961 - ETA: 40s - loss: 0.1105 - accuracy: 0.961 - ETA: 40s - loss: 0.1124 - accuracy: 0.960 - ETA: 39s - loss: 0.1121 - accuracy: 0.960 - ETA: 38s - loss: 0.1126 - accuracy: 0.960 - ETA: 37s - loss: 0.1137 - accuracy: 0.959 - ETA: 36s - loss: 0.1136 - accuracy: 0.959 - ETA: 36s - loss: 0.1136 - accuracy: 0.959 - ETA: 35s - loss: 0.1142 - accuracy: 0.959 - ETA: 34s - loss: 0.1142 - accuracy: 0.958 - ETA: 33s - loss: 0.1137 - accuracy: 0.959 - ETA: 33s - loss: 0.1136 - accuracy: 0.959 - ETA: 32s - loss: 0.1138 - accuracy: 0.959 - ETA: 31s - loss: 0.1132 - accuracy: 0.959 - ETA: 30s - loss: 0.1129 - accuracy: 0.959 - ETA: 29s - loss: 0.1137 - accuracy: 0.959 - ETA: 29s - loss: 0.1135 - accuracy: 0.959 - ETA: 28s - loss: 0.1132 - accuracy: 0.959 - ETA: 27s - loss: 0.1133 - accuracy: 0.959 - ETA: 26s - loss: 0.1132 - accuracy: 0.959 - ETA: 25s - loss: 0.1129 - accuracy: 0.959 - ETA: 25s - loss: 0.1132 - accuracy: 0.958 - ETA: 24s - loss: 0.1140 - accuracy: 0.958 - ETA: 23s - loss: 0.1141 - accuracy: 0.958 - ETA: 22s - loss: 0.1143 - accuracy: 0.958 - ETA: 21s - loss: 0.1149 - accuracy: 0.958 - ETA: 21s - loss: 0.1152 - accuracy: 0.957 - ETA: 20s - loss: 0.1156 - accuracy: 0.957 - ETA: 19s - loss: 0.1159 - accuracy: 0.957 - ETA: 18s - loss: 0.1161 - accuracy: 0.957 - ETA: 18s - loss: 0.1160 - accuracy: 0.957 - ETA: 17s - loss: 0.1159 - accuracy: 0.957 - ETA: 16s - loss: 0.1160 - accuracy: 0.957 - ETA: 15s - loss: 0.1156 - accuracy: 0.957 - ETA: 14s - loss: 0.1156 - accuracy: 0.957 - ETA: 14s - loss: 0.1154 - accuracy: 0.957 - ETA: 13s - loss: 0.1157 - accuracy: 0.957 - ETA: 12s - loss: 0.1155 - accuracy: 0.957 - ETA: 11s - loss: 0.1155 - accuracy: 0.957 - ETA: 10s - loss: 0.1162 - accuracy: 0.957 - ETA: 10s - loss: 0.1160 - accuracy: 0.957 - ETA: 9s - loss: 0.1157 - accuracy: 0.957 - ETA: 8s - loss: 0.1157 - accuracy: 0.95 - ETA: 7s - loss: 0.1159 - accuracy: 0.95 - ETA: 7s - loss: 0.1160 - accuracy: 0.95 - ETA: 6s - loss: 0.1164 - accuracy: 0.95 - ETA: 5s - loss: 0.1163 - accuracy: 0.95 - ETA: 4s - loss: 0.1160 - accuracy: 0.95 - ETA: 3s - loss: 0.1165 - accuracy: 0.95 - ETA: 3s - loss: 0.1166 - accuracy: 0.95 - ETA: 2s - loss: 0.1167 - accuracy: 0.95 - ETA: 1s - loss: 0.1170 - accuracy: 0.95 - ETA: 0s - loss: 0.1169 - accuracy: 0.95 - 84s 3ms/step - loss: 0.1166 - accuracy: 0.9574 - val_loss: 0.3187 - val_accuracy: 0.8886\n"
     ]
    }
   ],
   "source": [
    "# let's begin training our network\n",
    "history = model.fit(data[\"trainX\"], data[\"trainY\"], batch_size=batchSize, epochs=epochs, \n",
    "                    validation_data=(data[\"valX\"], data[\"valY\"]), callbacks=[ES], verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 1, ..., 0, 1, 1], dtype=int64)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# time to test our model\n",
    "prediction = model.predict(data[\"testX\"], batch_size=batchSize)\n",
    "\n",
    "# let's grab the class with maximum probability\n",
    "predictionIdx = np.argmax(prediction, axis=1)\n",
    "\n",
    "predictionIdx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    negative       0.90      0.90      0.90      5005\n",
      "    positive       0.90      0.89      0.90      4995\n",
      "\n",
      "    accuracy                           0.90     10000\n",
      "   macro avg       0.90      0.90      0.90     10000\n",
      "weighted avg       0.90      0.90      0.90     10000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# let's have a view at classification_report\n",
    "print(classification_report(data[\"testY\"].argmax(axis=1), predictionIdx, target_names=['negative', 'positive']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] Saving the model to disk...\n"
     ]
    }
   ],
   "source": [
    "# serializing model to disk\n",
    "print(\"[INFO] Saving the model to disk...\")\n",
    "model.save(modelName)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1152x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# let's plot a graph for training history\n",
    "epochs = 17\n",
    "\n",
    "plt.figure(figsize=(16, 10))\n",
    "\n",
    "plt.plot(np.arange(0, epochs), history.history[\"loss\"], label=\"train_loss\")\n",
    "plt.plot(np.arange(0, epochs), history.history[\"val_loss\"], label=\"val_loss\")\n",
    "plt.plot(np.arange(0, epochs), history.history[\"accuracy\"], label=\"train_acc\")\n",
    "plt.plot(np.arange(0, epochs), history.history[\"val_accuracy\"], label=\"val_acc\")\n",
    "\n",
    "plt.title(\"Training Loss and Accuracy on Dataset!\")\n",
    "plt.xlabel(\"Epoch #\")\n",
    "plt.ylabel(\"Loss/Accuracy\")\n",
    "plt.legend(loc=\"lower left\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
