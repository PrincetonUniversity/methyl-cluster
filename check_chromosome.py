# Imports

import time
import numpy as np
import pandas as pd
import os, os.path
from itertools import *
import math
import random
import scipy.stats
import sys
from joblib import Parallel, delayed
import multiprocessing
nproc = max(1, multiprocessing.cpu_count() - 1)
from functools import partial

from intervaltree import IntervalTree, Interval
from sklearn import linear_model
from sklearn import ensemble
from scipy.spatial import distance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.grid_search import GridSearchCV

# Warnings

import warnings
warnings.filterwarnings('ignore')

# Idempotent data retrieval script

chromosomes = [1, 2, 6, 7, 11]
def chromosome_files(n):
    base = 'intersected_final_chr'
    spec = '_cutoff_20_'
    suffixes = ['train.bed', 'sample_partial.bed', 'sample_full.bed']
    return [base + str(n) + spec + suffix for suffix in suffixes]
all_chromosomes = set(chain.from_iterable(chromosome_files(n) for n in chromosomes))

if 'methylation_imputation' not in [x for x in os.listdir('.') if os.path.isdir(x)]:
    raise Exception('Missing assignment repository in cwd')

if not os.path.exists('data'):
    os.mkdir('data')

def have_chromosomes(): return all_chromosomes.issubset(set(os.listdir('data')))
if not have_chromosomes():
    raise Exception('Error, missing chromosomes data')

encode_file = 'wgEncodeRegTfbsClusteredV3.bed'
annotations_files = {encode_file}
def have_annotations(): return annotations_files.issubset(set(os.listdir('data')))
if not have_annotations():
    raise Exception('Error, missing ENCODE data')

def read_tsv(name): return pd.read_csv(name, sep='\t', header=None)

def local_impute(data):
    #http://stackoverflow.com/questions/9537543/replace-nans-in-numpy-array-with-closest-non-nan-value
    mask = np.isnan(data)
    data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
    return data

def correlation_similarity(x, y):
    c = distance.correlation(x, y)
    if np.isnan(c): return 0
    else: return 1 - c
    
def log_range(lo, hi): return [10 ** i for i in range(lo, hi)]

def start(s): 
    print(s.ljust(50), end='')
    sys.stdout.flush()
    return time.time()

def end(t):
    print('{: <8.2f} sec'.format(time.time() - t))
    sys.stdout.flush()

def chromosome_df(n):
    t = start('loading chromosome data')
    train_chr = read_tsv('data/' + chromosome_files(n)[0])
    test_chr_part = read_tsv('data/' + chromosome_files(n)[1])
    test_chr_full = read_tsv('data/' + chromosome_files(n)[2])
    end(t)

    t = start('feature extraction on chromosome data')
    test_ix = np.where((test_chr_part[5] == 0) & ~np.isnan(test_chr_full[4]))[0]
    train_ix = np.where((test_chr_part[5] == 1) & ~np.isnan(test_chr_part[4]))[0]

    train_df = train_chr
    train_tissues = ['b' + str(x) for x in range(train_chr.shape[1] - 4)]
    train_df.columns = ['chromosome', 'start', 'end', 'strand'] + train_tissues

    test_df = test_chr_full
    test_df.columns = ['chromosome', 'start', 'end', 'strand', 'filled', 'assayed']
    test_df['missing'] = test_chr_part[4]

    train_df['strand'] = train_df['strand'] == '+'
    test_df['strand'] = test_df['strand'] == '+'
    
    for i in train_tissues:
        train_df[i] = local_impute(train_df[i].copy())
        
    similarities = np.array([correlation_similarity(train_df[t].iloc[train_ix],
                                                    test_df['filled'].iloc[train_ix])
                             for t in train_tissues])
    k = np.sum(np.fabs(similarities))
    cf = pd.Series(train_df[train_tissues].values.dot(similarities) / k, index=train_df.index)
    train_df['cf'] = cf
    end(t)

    t = start('loading CRE data')
    rawe = read_tsv('data/' + encode_file)
    rawe.drop([4, 5, 6, 7], axis=1, inplace=True)
    rawe.columns = ['chromosome', 'start', 'end', 'tf']
    tfs = set(rawe['tf'])
    end(t)


    def inside_tf(iv, row):
        overlap = [x.data for x in iv[row['start']:row['end']]]
        return pd.Series({tf: (tf in overlap) for tf in tfs})
    
    t = start('creating interval tree')
    iv_chr = IntervalTree((Interval(*x[1]) for x in
                           rawe[rawe['chromosome'] == ('chr' + str(n))][['start', 'end', 'tf']].iterrows()))
    end(t)
    
    t = start('merging CRE intersections')
    # TODO: this could be parallelized easily here and in the iPython notebook
    train_df = train_df.merge(train_df[['start', 'end']].apply(partial(inside_tf, iv_chr), axis=1), copy=False,
                              left_index=True, right_index=True)
    end(t)
    
    t = start('training blunt model')
    best_combined = linear_model.Lasso()
    grid = GridSearchCV(best_combined, {'alpha': log_range(-10, 10)}, cv=50, n_jobs=nproc)
    grid.fit(train_df.iloc[train_ix], test_df['filled'].iloc[train_ix])
    alpha = grid.best_params_['alpha']
    end(t)
    
    print('blunt alpha {:02f}'.format(alpha))
    
    t = start('training precise model')
    alpha_range = np.arange(0, 2 * alpha, alpha / 100)
    grid = GridSearchCV(best_combined, {'alpha': alpha_range}, cv=50, n_jobs=nproc)
    grid.fit(df_with_cf.iloc[train_ix], test_df['filled'].iloc[train_ix])
    end(t)
                                                               
    guess = grid.predict(df_with_cf.iloc[test_ix])
    exact = test_df['filled'].iloc[test_ix].values

    rmse = math.sqrt(mean_squared_error(exact, guess))
    acc = correct_half(exact, guess)
    r2 = r2_score(exact, guess)
    res = guess - exact

    print('Ensemble on test: rmse {:04f} methyl acc {:04f} R^2 {:04f} alpha {:04f}'
          .format(rmse, acc, r2, grid.best_params_['alpha']))
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python check_chromosome.py n\n\n' + \
              'Runs the entire pipeline on chromosome n, outputting final performance.' + \
              'May spawn nproc jobs.', file=sys.stderr)
    chromosome_df(int(sys.argv[1]))

