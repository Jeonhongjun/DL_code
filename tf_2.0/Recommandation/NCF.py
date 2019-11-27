# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "-1"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
# %matplotlib inline
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import dot
from tensorflow.keras.models import Model

# specifically for deeplearning.
from tensorflow.keras.layers import Dropout, Flatten,Activation,Input,Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
from IPython.display import SVG

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

train = pd.read_csv('Recommandation/ml-20m/ratings.csv')

df=train.copy()
df['userId'].unique()
len(df['userId'].unique())
df['movieId'].unique()
len(df['movieId'].unique())

df['userId'].isnull().sum()
df['rating'].isnull().sum()
df['movieId'].isnull().sum()

df['rating'].min() # minimum rating
df['rating'].max() # minimum rating
len(df['rating'].unique())
df.userId = df.userId.astype('category').cat.codes.values
df.movieId = df.movieId.astype('category').cat.codes.values

df['userId'].value_counts(ascending=True)

len(df['movieId'].unique())

# creating utility matrix.
index = list(df['userId'].unique())
columns = list(df['movieId'].unique())
index = sorted(index)
columns = sorted(columns)
df = df[:1000000]
util_df = pd.pivot_table(data=df,values='rating',index='userId',columns='movieId')
# Nan implies that user has not rated the corressponding movie.
util_df.fillna(0)

# Creating Training and Validation Sets.

# x_train,x_test,y_train,y_test=train_test_split(df[['userId','movieId']],df[['rating']],test_size=0.20,random_state=42)
users = df.userId.unique()
movies = df.movieId.unique()

userid2idx = {o: i for i, o in enumerate(users)}
movieid2idx = {o: i for i, o in enumerate(movies)}

df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])

split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]
print(train.shape, valid.shape)

# Matrix Factorization
# Creating the Embeddings ,Merging and Making the Model from Embeddings.

n_latent_factors = 50
n_movies = len(df['movieId'].unique())
n_users = len(df['userId'].unique())

user_input = Input(shape=(1,), name='user_input', dtype='int64')
user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)
user_vec=Flatten(name='FlattenUsers')(user_embedding)
user_vec=Dropout(0.40)(user_vec)
#user_vec.shape

movie_input=Input(shape=(1,),name='movie_input',dtype='int64')
movie_embedding=Embedding(n_movies,n_latent_factors,name='movie_embedding')(movie_input)
movie_vec=Flatten(name='FlattenMovies')(movie_embedding)
movie_vec=Dropout(0.40)(movie_vec)
#movie_vec

sim = dot([user_vec,movie_vec],name='Simalarity-Dot-Product',axes=1)
nn_inp = Dense(128,activation='relu')(sim)
nn_inp = Dropout(0.4)(nn_inp)
# nn_inp=BatchNormalization()(nn_inp)
nn_inp = Dense(1,activation='relu')(nn_inp)

nn_model = Model([user_input, movie_input],nn_inp)
nn_model.summary()
# A summary of the model is shown below-->

# Compiling the Model

nn_model.compile(optimizer=Adam(lr=1e-3), loss='mse')
print(train.shape)
batch_size=128
epochs=10

# Fitting on Training set & Validating on Validation Set.
# with tf.device('/gpu:0'):
History = nn_model.fit([train.userId.values,train.movieId.values],train.rating.values, batch_size=batch_size,
                              epochs =epochs, validation_data = ([valid.userId.values,valid.movieId.values],valid.rating.values),
                              verbose = 1)

# nn_model.predict([valid.userId,valid.movieId])


# Evaluating the Model Performance

from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
import matplotlib.pyplot as plt
plt.plot(History.history['loss'] , 'g')
plt.plot(History.history['val_loss'] , 'b')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)
plt.show()
