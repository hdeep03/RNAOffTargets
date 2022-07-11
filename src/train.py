import sys
from dataloader import OneHotEncoder, load_fasta_data, RNASeqDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, MaxPooling1D, Dropout
import warnings
import os

def main():
    '''
    Trains a model using the system arguments for which type of base editor, which replicate,
    and the 
    '''
    arg1 = sys.argv[1] # Target file and directory
    print("Loading data...")
    be_type = arg1.split('/')[-2].upper()
    sample =  arg1.split('/')[-1][:-3].upper() # Number of sample replicate (e.g. 156B for ABEmax 1st replicate)
    num_epochs = 0
    if len(sys.argv)<3:
        num_epochs = 4
    else:
        num_epochs = int(sys.argv[2])
    
    data_dir = "data/raw/{0}/{0}-sequence/".format(be_type)
    
    
    train_chr = [str(x) for x in range(1, 20)] # Train on 1-19
    test_chr = [str(x) for x in range(20, 23)]+['X'] # Test on 20-22 & X
    
    train_seqs, train_ers, test_seqs, test_ers = [], [], [], []
    
    for rep in train_chr:
        seq, er = load_fasta_data(sample, rep, data_dir)
        train_seqs = seq+train_seqs
        train_ers = er+train_ers
    
    for rep in test_chr:
        seq, er = load_fasta_data(sample, rep, data_dir)
        test_seqs = seq+test_seqs
        test_ers = er+test_ers
        
    traingen = RNASeqDataGenerator(train_seqs, train_ers, batch_size=1024, logits=False)
    testgen = RNASeqDataGenerator(test_seqs, test_ers, batch_size=1024, logits=False)
    print('Successfully Loaded Data!\nBuilding Model...')
    
    # Use GPU for Training        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv1D(32, kernel_size=7, activation='relu', input_shape=(101,4)))
    model.add(BatchNormalization())
    model.add(Conv1D(32, kernel_size=7, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1D(16, kernel_size=7, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer="RMSprop", loss="mse", metrics=["mae", 'mse'])
    print('Model built!')
    print(model.summary())
    
    print('Training model...')
    history = model.fit(traingen, validation_data=testgen, workers=40, use_multiprocessing=True, epochs=num_epochs, verbose=2)
    print('Model trained!')
    
    model.save(sys.argv[1])
    print("Model written to '{}'".format(sys.argv[1]))
    
    
if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    warnings.filterwarnings('ignore')
    main()