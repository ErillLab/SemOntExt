#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:03:13 2017

INPUT
Reads in a file resulting from the test of a textOn test run.
The file should have the following format:
- TSV file: with header
- First row: ontology entry identifier (e.g. GO:0000003)
- First column: sentence identifier (12312432_1), where the first number is
  the PMID and the number after the underscore the sentence ID.
  This makes reference to whatever portion of the PMID (e.g. abstract) has
  been analyzed.
- Cells: the likelihood product: <P(G|A)>·P(S|G) for that GO-sentence pair

Reads in another file containing the answer key (i.e. which ontology entry
should have been associated with which sentence in a document).
This file should have the following format:
- TSV file: no header
- First column: sentence identifier (12312432_1), where the first number is
  the PMID and the number after the underscore the sentence ID.
  This makes reference to whatever portion of the PMID (e.g. abstract) has
  been analyzed.
- Second column: ontology entry identifier (e.g. GO:0000003)

OPERATION
The script reads the input file and creates a two-dimensional array.

@author: Ivan Erill
"""

#import sys for argument processing
import sys

#import options parser
import getopt

#import pandas for data frame features
import pandas as pd

#import numpy for arrays
import numpy as np

#import plotting
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
from scipy import interp

#import ROC + AUC functions from scikit-learn
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

#print 'Usage: ROC_generator.py system_output_file answer_key_file'
#print 'Input parameters: ', str(sys.argv)

try:
    opts, args = getopt.getopt(sys.argv[1:],"hs:k:",["so_file=","k_file="])
except getopt.GetoptError as err:
    print 'ROC_generator.py -s <so_file> -k <k_file>'
    print str(err)
    sys.exit(2)

so_file_name=''
k_file_name=''

for opt, arg in opts:
    print arg
    if opt == '-h':
        print 'ROC_generator.py -s <so_file> -k <k_file>'
        sys.exit()
    elif opt in ("-s", "--so_file"):
        so_file_name = arg
    elif opt in ("-k", "--k_file"):
        k_file_name = arg
    else:
        assert False, "unhandled option"       
print 'Expected output file is ', so_file_name
print 'Answer key file is ', k_file_name

#read the two-dimensional matrix of likelihood products into a pandas dataframe
with open(so_file_name, 'rb') as inputfile1:
    sysout = pd.read_csv(inputfile1,sep='\t',header=0, index_col=0)

#logarize expected outputs
sysout=np.log(sysout)

#create a dataframe prefilled with zeros of same dimensions as sysout
expout=sysout.copy(deep=True)
expout[:] = 0

#read the answer key into a pandas dataframe
with open(k_file_name, 'rb') as inputfile2:
    akey = pd.read_csv(inputfile2,sep='\t',header=None)


#go through answer key, set corresponding ontology entries on expout to 1 
#traverse answer key row-wise
for ak_index, ak_row in akey.iterrows():
    #traverse system outputs row-wise
    for eo_index, eo_row in expout.iterrows():
        #if sentence IDs match
        if (ak_row[0]==eo_index):
            #set expected output to one for the ontology entry in answer key
            expout.set_value(eo_index,ak_row[1],1)

#compute ROC-AUC
#convert datasets to arrays
eo_array=expout.as_matrix()#.transpose()
so_array=sysout.as_matrix()#.transpose()
            
print 'Micro AUC: ', roc_auc_score(eo_array, so_array, average='micro')
#print 'Macro AUC: ', roc_auc_score(eo_array, so_array, average='samples')

n_classes=eo_array.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(eo_array[:, i], so_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(eo_array.ravel(), so_array.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()

# Compute macro-average ROC curve and ROC area

## First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
## Finally average it and compute AUC
#mean_tpr /= n_classes
#
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

matplotlib.rc('font', **font)

# Plot all ROC curves
plt.figure()

#plt.plot(fpr["macro"], tpr["macro"],
#         label='random AUC = {0:0.2f})'
#               ''.format(0.5),
#         color='navy', linestyle=':', linewidth=4)

plt.plot(fpr["micro"], tpr["micro"],
         label='GO average' #(AUC={0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

#checkout 0006067 0052170 0043565

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
class_inds=[expout.columns.get_loc('GO:0009432'),expout.columns.get_loc('GO:0045893'),expout.columns.get_loc('GO:0045892')]
classes=['GO:0009432','GO:0045893','GO:0045892']
for i, cl, color in zip(class_inds, classes, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} '#(AUC={1:0.2f})'
             ''.format(cl, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curves')
plt.legend(loc="lower right")
plt.show()

