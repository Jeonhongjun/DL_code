'''
Regression example taken from @apaszke's example from the
main pytorch repo.

https://github.com/pytorch/examples/blob/master/regression/main.py

Adapted to demo fmpytorch.
'''

#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import numpy as np
import pandas as pd
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from fmtorch.fmpytorch.second_order.fm import FactorizationMachine
from sklearn.utils import shuffle
import torch.nn as nn
criterion = nn.CrossEntropyLoss()

train = pd.read_csv('ml-20m/ratings.csv')
df=train.copy()
df.userId = df.userId.astype('category').cat.codes.values
df.movieId = df.movieId.astype('category').cat.codes.values

df['userId'].value_counts(ascending=True)

len(df['movieId'].unique())

# creating utility matrix.
index=list(df['userId'].unique())
columns=list(df['movieId'].unique())
index=sorted(index)
columns=sorted(columns)
df = df[:1000000]
util_df = pd.pivot_table(data=df,values='rating',index='userId',columns='movieId')
# Nan implies that user has not rated the corressponding movie.
util_df.fillna(0)
# Creating Training and Validation Sets.
# x_train,x_test,y_train,y_test=train_test_split(df[['userId','movieId']],df[['rating']],test_size=0.20,random_state=42)
users = df.userId.unique()
movies = df.movieId.unique()
userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}
df['userId'] = df['userId'].apply(lambda x: userid2idx[x])
df['movieId'] = df['movieId'].apply(lambda x: movieid2idx[x])

split = np.random.rand(len(df)) < 0.8
train = df[split]
valid = df[~split]

train_x = train.drop(["rating", 'timestamp'], axis = 1)
train_y = train.drop(["userId", 'movieId', 'timestamp'], axis = 1)
valid_x = valid.drop(["rating", 'timestamp'], axis = 1)
valid_y = valid.drop(["userId", 'movieId', 'timestamp'], axis = 1)

print(train.shape , valid.shape)
train_x = train_x.values
train_y = train_y.values
valid_x = valid_x.values
valid_y = valid_y.values

POLY_DEGREE = 2
W_target = torch.randn(POLY_DEGREE, 1) * 3
b_target = torch.randn(1) * 3
train_x.shape
train_y.shape
valid_x.shape
valid_y.shape

#
# def make_features(x):
#     """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
#     x = x.unsqueeze(1)
#     return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)
#
#
# def f(x):
#     """Approximated function."""
#     return x.mm(W_target) + b_target[0]
#
#
# def poly_desc(W, b):
#     """Creates a string description of a polynomial."""
#     result = 'y = '
#     for i, w in enumerate(W):
#         result += '{:+.2f} x^{} '.format(w, len(W) - i)
#     result += '{:+.2f}'.format(b[0])
#     return result
#
#
# def get_batch(batch_size=32):
#     """Builds a batch i.e. (x, f(x)) pair."""
#     random = torch.randn(batch_size)
#     x = make_features(random)
#     y = f(x)
#     return Variable(x), Variable(y)

# Define model
fc = FactorizationMachine(W_target.size(0), 3)

batch_size = 64
batch_no = len(train) // batch_size

num_epochs = 100
for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch+1))
    train_x, train_y = shuffle(train_x, train_y)
    # Mini batch learning
    for i in range(batch_no):
    # Get data
        start = i * batch_size
        end = start + batch_size
        x_var = Variable(torch.FloatTensor(train_x[start:end]))
        y_var = Variable(torch.FloatTensor(train_y[start:end]))

        # Reset gradients
        fc.zero_grad()

        # Forward pass
        output = criterion(fc(x_var), y_var)
        loss = output.data

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-0.01 * param.grad.data)
    print('train loss:', loss.item())
    # Stop criterion
    # if loss < 1e-3:
    #     break
    # print(loss)

with torch.no_grad():
    pred = fc(test)
    loss = criterion(pred, y_test)
    print('test loss:', loss.item())

#########################################
