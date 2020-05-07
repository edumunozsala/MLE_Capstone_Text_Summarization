
import argparse
import torch
from fastai.text import *

#from trainer import data_process
#from trainer import model
import data_process
import model

def train_model(args):
    """Load the data, train the model, export / save the model, save predictions to file  
    """
    # Download the dataset
    data_process.download_data(args.model_dir, data_process.DATA_PATH, args.data_filename)

    #Load the data and create the DataBunch
    data = data_process.load_data(args.data_filename, max_length=args.output_length, batch_size=args.batch_size)
    
    # Download the embeddings vector
    data_process.download_data(args.model_dir, data_process.EMB_PATH, args.emb_filename)

    #Load the Glove embeddings
    vectorizer = data_process.GloveVectorizer(args.emb_filename)
    
    # Create the embeddings
    emb_enc = data_process.create_emb(vectorizer, data.x.vocab.itos, vectorizer.embedding.shape[1])
    emb_dec = data_process.create_emb(vectorizer, data.y.vocab.itos, vectorizer.embedding.shape[1])
    
    # Create the model
    seq_model = model.Seq2SeqRNN_attn(emb_enc, emb_dec, args.hidden_size, args.output_length)
    learn = Learner(data, seq_model, loss_func=model.seq2seq_loss, metrics=[model.seq2seq_acc],
                callback_fns=partial(model.TeacherForcing, end_epoch=args.epochs))

    # Train the model
    learn.fit_one_cycle(args.epochs, args.lr)
    
    # Export the trained model
    learn.save(args.model_name)
    
    if args.model_dir:
        # Save the model to GCS
        data_process.save_model(args.model_dir, args.model_name)
        
    #Evaluate the model, predicting on the validation set
    inputs, targets, outputs = model.get_predictions(learn)
    #Save the output predictions to file
    data_process.save_df(inputs, targets, outputs,args.model_dir)

def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='Text Summarization Fastai')
    parser.add_argument('--model-dir',
                        type=str,
                        default='mlend_bucket',
                        help='Where to save the model')
    parser.add_argument('--model-name',
                        type=str,
                        default='text_summa',
                        help='What to name the saved model file')
    parser.add_argument('--data_filename',
                        type=str,
                        default='news_summary_more.csv',
                        help='The filename with the data')
    parser.add_argument('--emb_filename',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='The filename with the embeddings vector')
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--hidden_size',
                        type=float,
                        default=50,
                        help='sizew of hidden layer')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.03,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--output_length',
                        type=float,
                        default=76,
                        help='max output length (default: 76)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed (default: 42)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    train_model(args)


if __name__ == '__main__':
    main()