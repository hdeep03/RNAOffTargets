import sys
import math
from sklearn.linear_model import Ridge
from dataloader import load_fasta_data
import numpy as np 
import scipy

from tqdm import tqdm

encode = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T':3
}
def logit_transform(x):
    if x < 1/(1+math.exp(7)):
         return -7
    if x>1/(1+math.exp(-7)):
          return 7
    return math.log(x/(1-x))
def seq2twohot(seq):
    accumulator = 0
    seq = seq[47:54]
    for i in range(3):
        accumulator += encode[seq[i:i+1]]*4**i
        
    arr = np.zeros(64, dtype=int)
    arr[accumulator] = 1
    accumulator = 0
    for i in range(4,7):
        accumulator += encode[seq[i:i+1]]*4**(i-4)   
    arr2 = np.zeros(64, dtype=int)
    arr2[accumulator] = 1
    return np.hstack([arr, arr2])

def main():
    '''
    Trains a regression using the system arguments for which type of base editor, which replicate
    '''
    arg1 = sys.argv[1] # Target file and directory
    print("Loading data...")
    be_type = arg1.split('/')[-2].upper()
    sample =  arg1.split('/')[-1].split('.')[0].upper() # Number of sample replicate (e.g. 156B for ABEmax 1st replicate)
    
    data_dir = "data/raw/{0}/{0}-sequence/".format(be_type)
    
    logit = False
    if len(sys.argv)>2 and sys.argv[2].lower()=='logit':
        logit = True
        
    
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
        
    train_seq_reg = np.array([seq2twohot(x) for x in tqdm(train_seqs)])
    test_seq_reg = np.array([seq2twohot(x) for x in tqdm(test_seqs)])
    
    if logit:
        train_er_reg = np.array([logit_transform(x) for x in tqdm(train_ers)])
        test_er_reg = np.array([logit_transform(x) for x in tqdm(test_ers)])
    else:
        train_er_reg = np.array(train_ers)
        test_er_reg = np.array(test_ers)
    print('Fitting regression...')
    reg = Ridge().fit(train_seq_reg, train_er_reg)
    print('Regression fitted!')
    print('R^2:{}; Pearson-r:{}'.format(reg.score(test_seq_reg, test_er_reg),scipy.stats.pearsonr(reg.predict(test_seq_reg), test_er_reg)))
    
    with open(arg1, 'wb') as f:
        np.save(f, reg.coef_)
        np.save(f, reg.intercept_)

if __name__ == '__main__':
    main()