#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import datasets, transforms, models

import torch
import time
import copy
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F


SEED = 23
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:

print(device)

# In[42]:


train_path = "./dataset/ECG5000_TRAIN.arff"
test_path = "./dataset/ECG5000_TEST.arff"
with open(train_path) as f:
  train = pd.DataFrame(loadarff(f)[0])

with open(test_path) as f:
  test = pd.DataFrame(loadarff(f)[0])


# Because we're identifying anamolies, we need as much data as possible.

# In[43]:


# concatinating the data
df = train.append(test)

# shuffling the data
df = df.sample(frac=1.0)


# In[44]:


CLASS = 1
classes = ['Normal', 'R on T', 'PVC', 'SP', 'UB']


# ## Data Preprocessing

# In[45]:


# changing name of target column
new_cols = list(df.columns)
new_cols[-1] = 'target'
df.columns = new_cols
df['target'].replace({b'1': 1, b'2': 2, b'3': 3, b'4': 4, b'5': 5}, inplace=True)


# ## Data Exploration

# In[46]:


# imbalance in data can observed
df.target.value_counts()


# In[47]:


ax = sns.countplot(x=df.target)
ax.set_xticklabels(classes)
fig = ax.get_figure()
fig.savefig("counts.png")


# In[48]:


def plot_time_series_class(data, classes, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.525
  )
  ax.set_title(classes)


# In[49]:


target_classes = df.target.unique()

fig, axs = plt.subplots(
  nrows = 3,
  ncols = 2,
  sharey = True,
  figsize = (9, 9)
)


for i, cls in enumerate(target_classes):
  ax = axs.flat[i]
  data = df[df.target == cls] \
    .drop(labels='target', axis=1) \
    .mean(axis=0) \
    .to_numpy()
  plot_time_series_class(data, classes[i], ax)

fig.delaxes(axs.flat[-1])
fig.tight_layout();
fig.savefig("morph.png")


# In[50]:


df['target'].replace({1: 1, 2: 0, 3: 0, 4: 0, 5: 0}, inplace=True)


# In[51]:


normal_df = df[df.target ==  CLASS]
anomaly_df = df[df.target != CLASS]

# splitting the normal data into training and validation data
train_df, val_df = train_test_split(normal_df, test_size=0.15, random_state=SEED)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=SEED)

class DFWL:
  full = df
  train = train_df
  val = val_df
  test = test_df
  anomaly = anomaly_df


# In[52]:


def gen_dataset(tdf):
  tdf = tdf.drop(labels='target', axis=1)
  seq = tdf.astype(np.float32).to_numpy().tolist()
  dataset = [torch.tensor(s).unsqueeze(1) for s in seq]
  n_seq, seq_len, n_feat = torch.stack(dataset).shape
  return dataset, seq_len, n_feat


# In[53]:


# convert dataframes to sequences (numpy array)

full_ds, _, _ = gen_dataset(DFWL.full)
train_ds, seq_len, n_feat = gen_dataset(DFWL.train)
val_ds, _, _ = gen_dataset(DFWL.val)
test_ds, _, _ = gen_dataset(DFWL.test)
anomaly_ds, _, _ = gen_dataset(DFWL.anomaly)


# In[54]:


class options:
  lr = 0.001
  seq_len = seq_len
  batch_size = 1
  num_workers = 2


# ## AUTOENCODERS

# In[55]:


class LSTMEncoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMEncoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True,
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

class LSTMDecoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(LSTMDecoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True,
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class LSTMAE(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMAE, self).__init__()

    self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

class KLLSTMAE(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(KLLSTMAE, self).__init__()

    self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    h = self.encoder(x)
    x = self.decoder(h)

    return x, torch.log(F.softmax(h, dim=1))


# In[56]:


class BiEncoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(BiEncoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
      bidirectional=True,
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim * 2,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
      bidirectional=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.hidden_dim * 2))

class BiDecoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(BiDecoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 4 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim * 4,
      hidden_size=input_dim * 2,
      num_layers=1,
      batch_first=True,
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim * 2,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim * 4))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class BiLSTM(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(BiLSTM, self).__init__()

    self.encoder = BiEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = BiDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

class KLBiLSTM(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(KLBiLSTM, self).__init__()

    self.encoder = BiEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = BiDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    h = self.encoder(x)
    x = self.decoder(h)

    return x, torch.log(F.softmax(h, dim=1))


# In[57]:


class GRUEncoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(GRUEncoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.GRU(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.rnn2 = nn.GRU(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True,
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, _ = self.rnn1(x)
    x, hidden_n = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

class GRUDecoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(GRUDecoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.GRU(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True,
    )

    self.rnn2 = nn.GRU(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True,
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, _ = self.rnn1(x)
    x, _ = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class GRUAE(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(GRUAE, self).__init__()

    self.encoder = GRUEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = GRUDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

class KLGRUAE(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(KLGRUAE, self).__init__()

    self.encoder = GRUEncoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = GRUDecoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    h = self.encoder(x)
    x = self.decoder(h)

    return x, torch.log(F.softmax(h, dim=1))


# In[58]:


def loss_func(out, real):
    pred, latent = out[0], out[1]
    z = torch.rand(latent.size(0), latent.size(1), device=device)

    klloss_f = nn.KLDivLoss(reduction="batchmean")
    l1loss_f = nn.L1Loss(reduction='sum').to(device)
    mseloss_f = nn.MSELoss(reduction='sum').to(device)

    kl_loss = klloss_f(latent, z)
    mseloss = mseloss_f(pred, real)
    tloss = mseloss + kl_loss

    return tloss


# In[59]:


def train_model(model, train_dataset, val_dataset, n_epochs, kl=False):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      if (kl == True):
        loss = loss_func(seq_pred, seq_true)
      else:
        loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_ds:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        if (kl == True):
          loss = loss_func(seq_pred, seq_true)
        else:
          loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    history['val'].append(val_loss)
    train_loss = np.mean(train_losses)
    history['train'].append(train_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history


# In[60]:


def train_iterate_models(models, epochs=150, kl=False):
  for model_key in models:
    model = models[model_key].to(device)
    model, history = train_model(model, train_ds, val_ds, n_epochs=epochs, kl=kl)
    model_save_name = '{}{}_gpu.pth'.format(model_key, epochs)
    torch.save(model, model_save_name)


# In[61]:


models = {"lstm": LSTMAE(seq_len, n_feat, 128), "bilstm": BiLSTM(seq_len, n_feat, 128), "gru": GRUAE(seq_len, n_feat, 128)}
kl_models = {"kl_lstm": KLLSTMAE(seq_len, n_feat, 128), "kl_gru": KLGRUAE(seq_len, n_feat, 128), "kl_bilstm": KLBiLSTM(seq_len, n_feat, 128)}
# train_iterate_models(models, epochs=300, kl=False)
# train_iterate_models(kl_models, epochs=300, kl=True)


# In[62]:


class LModel:
  def __init__(self, name, model, kl):
    self.name = name
    self.model = model
    self.kl = kl

def get_loaded_model(path, name, model, kl=False):
  if kl == False:
    model.load_state_dict(torch.load(path))
  model = model.to(device)
  return LModel(name, model, kl)

lstm_model = get_loaded_model('./models/lstm.pth', 'LSTM', LSTMAE(seq_len, n_feat, 128))
gru_model = get_loaded_model('./models/gru.pth', 'GRU', GRUAE(seq_len, n_feat, 128))

bilstm_model = LModel('BiLSTM', torch.load('./models/bilstm.pth'), False)

kl_lstm_model = LModel('kl_LSTM', torch.load('./models/kl_lstm.pth'), True)
kl_gru_model = LModel('kl_GRU', torch.load('./models/kl_gru.pth'), True)
kl_bilstm_model = LModel('kl_BiLSTM', torch.load('./models/kl_bilstm.pth'), True)

loaded_models = [bilstm_model, lstm_model, gru_model]
loaded_kl_models = [kl_bilstm_model, kl_lstm_model, kl_gru_model]


# # Choosing a threshold

# In[63]:


def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.MSELoss(reduction='sum').to(device)
  kl = model.kl
  model = model.model
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)
      if kl:
        seq_pred = seq_pred[0]
      loss = criterion(seq_pred, seq_true)
      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses


# In[64]:


val_df = DFWL.test.copy()
test_df = DFWL.anomaly.sample(219).copy()
val_test_df = pd.concat([val_df, test_df], axis=0)
true_preds = val_test_df['target'].values.tolist()
val_test_ds, _, _ = gen_dataset(val_test_df)

anomaly_ds, _, _ = gen_dataset(DFWL.anomaly)
normal_ds, _, _ = gen_dataset(DFWL.train.sample(1500).copy()) 


# In[65]:


val_df.shape


# In[66]:


# _, val_test_l = predict(model, val_test_ds)


# In[67]:


from sklearn.metrics import recall_score as RC, precision_score as PC
from sklearn.metrics import accuracy_score as AC

def get_preds(th, val_test_l):
  val_test_preds = []
  for loss in val_test_l:
    pred = 1 if loss <= th else 0
    val_test_preds.append(pred)

  return val_test_preds

def calc_threshold(val_test_l):
  TH = 0
  F1 = 0
  for th in range(1, 101):
    val_test_preds = get_preds(th, val_test_l)
    recall = RC(true_preds, val_test_preds, average='micro')
    prec = PC(true_preds, val_test_preds, average='micro')

    f1score = (2 * (prec * recall)) / (prec + recall)

    TH, F1 = (th, f1score) if f1score > F1 else (TH, F1)

  return TH


# In[68]:


def get_prec_recall_f1(val_test_l, th):

  val_test_preds = get_preds(th, val_test_l)

  # micro - Calculate metrics globally by counting the total true positives, false negatives and false positives.
  recall = RC(true_preds, val_test_preds, average='micro')
  prec = PC(true_preds, val_test_preds, average='micro')
  accur = AC(true_preds, val_test_preds)


  f1score = (2 * (prec * recall)) / (prec + recall)


  print(f"\nAccuracy (%) : {accur * 100:.3f} F1score : {f1score:.3f}")


# In[69]:


def print_metrics(models):
  for model in models:
    _, loss = predict(model, val_test_ds)
    th = calc_threshold(loss)
    print('\n' + model.name)
    get_prec_recall_f1(loss, th)


# In[70]:


print_metrics(loaded_models)


# In[71]:


print_metrics(loaded_kl_models)


# In[72]:


def plot_preds(model, dataset, title, ax):

  predictions, losses = predict(model, [data])

  ax.plot(data, label='true')
  ax.plot(predictions[0], label='predicted')
  ax.set_title(f'Construction Loss ({title}): {np.around(losses[0], 2)}')


# In[73]:

total_cols = 2
fig, axs = plt.subplots(nrows=2, ncols=total_cols, sharex=True, sharey=True, figsize=(14, 8))

for i, data in enumerate(normal_ds[:4]):
    if i <= 1:
      plot_preds(kl_bilstm_model, data, title='Normal', ax=axs[0, i % 2])
    else:
      plot_preds(kl_bilstm_model, data, title='Normal', ax=axs[1, i % 2])

fig.tight_layout()
fig.savefig("recon_norm.png")

total_cols = 2
fig, axs = plt.subplots(nrows=2, ncols=total_cols, sharex=True, sharey=True, figsize=(14, 8))

for i, data in enumerate(anomaly_ds[:4]):
    if i <= 1:
      plot_preds(kl_bilstm_model, data, title='Abnormal', ax=axs[0, i % 2])
    else:
      plot_preds(kl_bilstm_model, data, title='Abnormal', ax=axs[1, i % 2])


fig.tight_layout()
fig.savefig("recon_abnorm.png")
