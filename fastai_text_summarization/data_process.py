# Script with functions to load, process and save data for our model

import datetime
from google.cloud import storage
import numpy as np
import pandas as pd

from fastai.text import *

DATA_PATH = 'data/news_summary/'
EMB_PATH = 'embeddings/'
#TEMP_DATA_PATH = f'{DATA_PATH}/temp'
#glove_filename=f'glove.6B.100d.txt'

def save_df(inputs, targets, outputs,model_dir):
    """Saves the predictions to Google Cloud Storage"""
    #Convert the list of predictions to a Dataframe and save it to file
    df = pd.DataFrame(data={"input": inputs, "targets": targets, "predictions": outputs})
    filename=datetime.datetime.now().strftime('preds_%Y%m%d_%H%M%S.csv')
    df.to_csv(filename, sep='\t',index=False)
    #Create the blob object to the GCS
    bucket = storage.Client().bucket(model_dir)
    blob = bucket.blob('{}/{}'.format('predictions',filename))
    blob.upload_from_filename(filename)

def save_model(model_dir, model_name, model_path='models/'):
    """Saves the model to Google Cloud Storage"""
    bucket = storage.Client().bucket(model_dir)
    model_filename=model_name+'.pth'
    blob = bucket.blob('{}/{}'.format(
        datetime.datetime.now().strftime('summa_%Y%m%d_%H%M%S'),
        model_filename))
    blob.upload_from_filename(model_path+model_filename)
    
def download_data(model_dir, file_path, filename):
    """Download the data from Google Cloud Storage"""
    # Load the Dataset from the GCS bucket
    bucket = storage.Client().bucket(model_dir)
    # Path to the data inside the public bucket
    blob = bucket.blob(file_path+filename)
    # Download the data
    blob.download_to_filename(filename)

# The first thing is that we will need to collate inputs and targets in a batch: they have different lengths 
# so we need to add padding to make the sequence length the same
def seq2seq_collate(samples, pad_idx=1, pad_first=True, backwards=False):
    "Function that collect samples and adds padding. Flips token order if needed"
    samples = to_data(samples)
    max_len_x,max_len_y = max([len(s[0]) for s in samples]),max([len(s[1]) for s in samples])
    res_x = torch.zeros(len(samples), max_len_x).long() + pad_idx
    res_y = torch.zeros(len(samples), max_len_y).long() + pad_idx
    if backwards: pad_first = not pad_first
    for i,s in enumerate(samples):
        if pad_first: 
            res_x[i,-len(s[0]):],res_y[i,-len(s[1]):] = LongTensor(s[0]),LongTensor(s[1])
        else:         
            res_x[i,:len(s[0]):],res_y[i,:len(s[1]):] = LongTensor(s[0]),LongTensor(s[1])
    if backwards: res_x,res_y = res_x.flip(1),res_y.flip(1)
    return res_x,res_y

# Then we create a special DataBunch that uses this collate function.
class Seq2SeqDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=32, val_bs:int=None, pad_idx=1,
               dl_tfms=None, pad_first=False, device:torch.device=None, no_check:bool=False, backwards:bool=False, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(seq2seq_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

# And a subclass of TextList that will use this DataBunch class in the call .databunch and will use TextList to label (since our targets are other texts).

class Seq2SeqTextList(TextList):
    _bunch = Seq2SeqDataBunch
    _label_cls = TextList


def load_data(filename, max_length=76, batch_size=32):
    """Loads the data"""
    #Load the data to a dataframe
    df = pd.read_csv(filename)
    # To lowercase
    df['text'] = df['text'].apply(lambda x:x.lower())
    df['headlines'] = df['headlines'].apply(lambda x:x.lower())
    
    #src = Seq2SeqTextList.from_df(df, path = TEMP_DATA_PATH, cols='text').split_by_rand_pct(seed=42).label_from_df(cols='headlines',label_cls=TextList)
    src = Seq2SeqTextList.from_df(df, path = '', cols='text').split_by_rand_pct(seed=42).label_from_df(cols='headlines',label_cls=TextList)
    
    #max_length=76
    src = src.filter_by_func(lambda x,y: len(x) > max_length or len(y) > max_length)
    
    data = src.databunch()
    
    return data

# Create a Class that load the Glove embeddings and generates the word2vec (mapping words to embedding vector). The Class defines a function to transforme a list of sentences to a list of embeddings vectors
class GloveVectorizer:
  def __init__(self, embedding_file):
    # load in pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    embedding = []
    idx2word = []
    with open(embedding_file, encoding="utf-8") as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
    print('Found %s word vectors.' % len(word2vec))

    # save for later
    self.word2vec = word2vec
    self.embedding = np.array(embedding)
    self.word2idx = {v:k for k,v in enumerate(idx2word)}
    self.V, self.D = self.embedding.shape

  def fit(self, data):
    pass

  def transform(self, data):
    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.lower().split()
      vecs = []
      for word in tokens:
        if word in self.word2vec:
          vec = self.word2vec[word]
          vecs.append(vec)
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    #print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


def create_emb(vecs, itos, em_sz=100, mult=1.):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    #vec_dic = {w:vecs.word2vec[w] for w in vecs.get_words()}
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = tensor(vecs.word2vec[w])
        except: miss.append(w)
    return emb

