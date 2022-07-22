'''
Dataloader with the ability to onehot encode sequences and read from a fasta
@author Johannes Linder
@author Harsh Deep
@version 7.8.22
'''
from tensorflow import keras
import numpy as np 
import os
import gzip

class RNASeqDataGenerator(keras.utils.Sequence):
    
    def __init__(self, seqs, er, batch_size=256, logits=False):
        self.sequences = seqs
        self.elements = len(seqs)
        self.batch_size = batch_size
        self.edit_rates = er
        self.log_edit = logits
        self.indexes = np.arange(len(self.sequences))
        self.encoder = OneHotEncoder(len(seqs[0]), {'A':0, 'C':1, 'G':2, 'T':3})
        
    def __len__(self):
        return self.elements // self.batch_size
    
    def __on_epoch_end__(self):
        self.indexes = np.arange(self.elements)
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]     
        return self.__data_generation(indexes)
    

    def __data_generation(self, list_IDs_temp): 
        X = np.array([self.encoder.encode(self.sequences[i]) for i in list_IDs_temp])
        if self.log_edit:
            y = np.array([logit_transform(self.edit_rates[i]) for i in list_IDs_temp])
        else:
            y = np.array([self.edit_rates[i] for i in list_IDs_temp])
        return X, y


def logit_transform(x):
    if x < 1/(1+math.exp(7)):
         return -7
    if x>1/(1+math.exp(-7)):
          return 7
    return math.log(x/(1-x))

def load_fasta_data(rep, chrm, base_path):
    file_path = os.path.join(base_path, f"{rep}.chr{chrm}.fasta.gz")
    er = []
    seqs = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line[0] == ">":
                idx = 0
                if '_F_' in line:
                    idx = line.index('_F_')+3
                else:
                    idx = line.index('_R_')+3
                line = line[idx:]
                line = line[:line.index("_")]
                er.append(float(line)) 
            else:
                seqs.append(line[:-1]) #Remove extra newline character
    assert len(er) == len(seqs) 
    return (seqs, er)

class SequenceEncoder :
    
    def __init__(self, encoder_type_id, encode_dims) :
        self.encoder_type_id = encoder_type_id
        self.encode_dims = encode_dims
    
    def encode(self, seq) :
        raise NotImplementedError()
    
    def encode_inplace(self, seq, encoding) :
        raise NotImplementedError()
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def decode(self, encoding) :
        raise NotImplementedError()
    
    def decode_sparse(self, encoding_mat, row_index) :
        raise NotImplementedError()
    
    def __call__(self, seq) :
        return self.encode(seq)
    
class OneHotEncoder(SequenceEncoder) :
    
    def __init__(self, seq_length, channel_map) :
        super(OneHotEncoder, self).__init__('onehot', (seq_length, len(channel_map)))
        
        self.seq_len = seq_length
        self.n_channels = len(channel_map)
        self.encode_map = channel_map
        self.decode_map = {
            val : key for key, val in channel_map.items()
        }
    
    def encode(self, seq) :
        encoding = np.zeros((self.seq_len, self.n_channels))
        
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.

        return encoding
    
    def encode_inplace(self, seq, encoding) :
        for i in range(len(seq)) :
            if seq[i] in self.encode_map :
                channel_ix = self.encode_map[seq[i]]
                encoding[i, channel_ix] = 1.
    
    def encode_inplace_sparse(self, seq, encoding_mat, row_index) :
        raise NotImplementError()
    
    def decode(self, encoding) :
        seq = ''
    
        for pos in range(0, encoding.shape[0]) :
            argmax_nt = np.argmax(encoding[pos, :])
            max_nt = np.max(encoding[pos, :])
            if max_nt == 1 :
                seq += self.decode_map[argmax_nt]
            else :
                seq += self.decode_map[self.n_channels - 1]

        return seq
    
    def decode_sparse(self, encoding_mat, row_index) :
        encoding = np.array(encoding_mat[row_index, :].todense()).reshape(-1, 4)
        return self.decode(encoding)