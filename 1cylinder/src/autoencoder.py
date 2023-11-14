import pandas as pd
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import os, sys
import urllib.request
from keras.models import load_model
import pickle
import copy, h5py
from copy import deepcopy as cp

def build_autoencoder(input_dim, layer_dim, active):
  active = 'relu'
  input_img = Input(shape=(input_dim))
  # layer_dim = [64*16,16*8,10*4,15,2]
  # Dense-Encoder
  x = Dense(layer_dim[0],activation=active)(input_img)
  x = Dense(layer_dim[1],activation=active)(x)
  x = Dense(layer_dim[2],activation=active)(x)
  x = Dense(layer_dim[3],activation=active)(x)
  # x = Dense(layer_dim[3],activation='tanh')(x)  # 加了一层
  encoded = Dense(layer_dim[-1])(x)   # 中间层暂时不用sigmoid
  # Dense-Decoder
  x1 = Dense(layer_dim[3],activation=active)(encoded)
  x1 = Dense(layer_dim[2],activation=active)(x1)
  x1 = Dense(layer_dim[1],activation=active)(x1)
  x1 = Dense(layer_dim[0],activation=active)(x1)
  decoded = Dense(input_dim)(x1)

  autoencoder_sae = Model(input_img, decoded)
  autoencoder_sae.compile(optimizer='adam', loss='mse')

  # Check the network structure
  autoencoder_sae.summary()
  return autoencoder_sae


def train_network(autoencoder_sae, filename_ae, X0_sae):
  n_epoch=200  #500 # Number of epoch【原测试中有10000个时间，我只测试2k】
  pat=50 # Patience
  timesnap = 2000
  tempfn='./'+filename_ae+'.hdf5'
  model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1)
  early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
  cb = [model_cb, early_cb]
  X_train,X_test,y_train,y_test=train_test_split(X0_sae,X0_sae,test_size=0.3,random_state=1)

  history=autoencoder_sae.fit(X_train, y_train,
                  epochs=n_epoch,
                  batch_size=10,
                  shuffle=True,
                  validation_data=(X_test, y_test),
                  callbacks=cb )
  df_results = pd.DataFrame(history.history)
  df_results['epoch'] = history.epoch
  tempfn='./'+filename_ae+'.csv'
  df_results.to_csv(path_or_buf=tempfn,index=False)    


def load_autoencoder(model_name, input_dim, latent_dim):
  autoencoder_sae_pod = load_model(model_name)
  autoencoder_sae_pod.summary()

  # 用于model_name='./sae_5layer_tanh.hdf5'
  # enc.
  enc_input = Input(shape=(128*64))  # input层

  deco = autoencoder_sae_pod.layers[1](enc_input)
  deco = autoencoder_sae_pod.layers[2](deco)
  deco = autoencoder_sae_pod.layers[3](deco)
  deco = autoencoder_sae_pod.layers[4](deco)
  enc = autoencoder_sae_pod.layers[5](deco)
  print(enc.shape)

  # create the decoder model
  encoder_sae_pod = Model(enc_input, enc)
  encoder_sae_pod.summary()

  # dec
  encoded_input = Input(shape=(2,))

  deco = autoencoder_sae_pod.layers[6](encoded_input)  # Dense层
  deco = autoencoder_sae_pod.layers[7](deco)
  deco = autoencoder_sae_pod.layers[8](deco)
  deco = autoencoder_sae_pod.layers[9](deco)
  deco = autoencoder_sae_pod.layers[10](deco)

  # create the decoder model
  decoder_sae_pod = Model(encoded_input, deco)
  decoder_sae_pod.summary()

  return autoencoder_sae_pod, encoder_sae_pod, decoder_sae_pod

  
