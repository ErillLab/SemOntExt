import os
import sys
import glob
import argparse
import time
import numpy as np
np.seterr(all='raise')

def_llword = "DEFAULT_ONT_L_L"
  
def setup_doc_dirs(root_dir, section):
  s_dir = os.path.join(root_dir, "input_documents/"+section+"s")
  c_dir = os.path.join(s_dir, "corpora")
  g_dir = os.path.join(root_dir, "genmodel_outputs")
  return (s_dir, c_dir, g_dir)
  
def getImmediateSubdirectories(dir):
  return [name for name in os.listdir(dir)
      if os.path.isdir(os.path.join(dir, name))]

def get_column(fname, col):
  icnt = 0
  item_idx = {}
  idx_item = {}
  with open(fname, "r") as infile:
    for l in infile:
      line = l.strip()
      if len(line) < 1:
        continue
      items = line.split("\t")
      the_info = items[0]
      if not the_info in item_idx:
        item_idx[the_info] = icnt
        idx_item[icnt] = the_info
        icnt += 1
  return (item_idx, idx_item)  

def get_ont_lls(fname):
  # Ontology LL info positions
  ont_id_pos = 0
  def_lh_pos = 1
  opos = 0
  item_idx = {}
  idx_item = {}
  lls = {}
  with open(fname, "r") as infile:
    for l in infile:
      line = l.strip()
      if len(line) < 1:
        continue
      this_temp = {}
      items = line.split("\t")
      ont_id = items[ont_id_pos]
      item_idx[ont_id] = opos
      idx_item[opos] = ont_id
      opos+=1
      
      # Get the log likelihood for a ont word not in the entry. 
      def_ll = np.log(float(items[def_lh_pos]))
      #print ont_id, items[def_lh_pos], def_ll
      this_temp[def_llword] = def_ll 
     
      # Process the remaining columns -- the ont words in the entry 
      num_terms = len(items)-2
      #print "  NT:", num_terms
      for i in range(2, len(items)):
        info = items[i].split("~~")
        word = info[0]
        ll_val = np.log(float(info[1]))
        this_temp[word] = ll_val
        #print "  ", word, info[1], ll_val
      lls[ont_id] = this_temp
  return (item_idx, idx_item, lls)    
  
  
def get_ont_words(fname):
  ont_id_pos = 0
  ont_words_pos = 1
  opos = 0
  item_idx = {} # for the ontology ids
  idx_item = {}
  curwords = {}
  with open(fname, "r") as infile:
    for l in infile:
      line = l.strip()
      if len(line) < 1:
        continue
      items = line.split("\t")
      ont_id = items[ont_id_pos]
      ont_words_str = items[ont_words_pos]
      item_idx[ont_id] = opos
      idx_item[opos] = ont_id
      opos+=1
      the_words = ont_words_str.split(" ")
      num_terms = len(the_words)
      #print "  NT:", num_terms
      curwords[ont_id] = the_words # here we keep repeated words separate
  return (item_idx, idx_item, curwords)    
      
def get_author_info(corpora_dir, author_idx, idx_author, doc_authors, doc_idx, idx_doc, audir_name):
  acnt = len(author_idx)
  dcnt = len(doc_idx)
  print "Authors start at acnt", acnt, "Docs start at", dcnt
  corpus_dirs = getImmediateSubdirectories(corpora_dir)
  for cdir in corpus_dirs:
    complete_path = os.path.join(corpora_dir, cdir)
    author_dir = os.path.join(complete_path, audir_name)
    file_to_glob = os.path.join(author_dir, "authors_*.txt")
    file_array = glob.glob(file_to_glob)
    for f in file_array:
      with open(f, "r") as infile:
        #print "authors file:", f
        fname = os.path.basename(f)
        froot = os.path.splitext(fname)[0]
        fitems = froot.split("_")
        pmid = fitems[1]
        if pmid not in doc_idx:
          doc_idx[pmid] = dcnt
          idx_doc[dcnt] = pmid
          dcnt += 1
        authors = []
        for l in infile:
          line = l.strip()
          if len(line) < 1:
            continue
          items = line.split("\t")
          if len(items) >= 2:
            aname = items[0]+" "+items[1]
          else:
            aname = items[0]
          authors.append(aname)
          if aname not in author_idx:
            author_idx[aname] = acnt
            idx_author[acnt] = aname
            acnt += 1
        doc_authors[pmid] = authors
        #print "Doc's authors:", pmid, authors
  return acnt
  
def get_sentences(corpora_dir, sentence_idx, idx_sentence, all_sentences, voc_idx, idx_voc, section, outdir_name):
  vstart = len(voc_idx.keys())
  print "Start v idx at", vstart
  n_sentences = len(sentence_idx)
  print "Start n_sentences at:", n_sentences
  corpus_dirs = getImmediateSubdirectories(corpora_dir)
  print corpus_dirs
  for cdir in corpus_dirs:
    complete_path = os.path.join(corpora_dir, cdir)
    doc_dir = os.path.join(complete_path, outdir_name)
    file_to_glob = os.path.join(doc_dir, section+"_*.txt")
    file_array = glob.glob(file_to_glob)
    for f in file_array:
      with open(f, "r") as infile:
        #print f
        fname = os.path.basename(f)
        froot = os.path.splitext(fname)[0]
        fitems = froot.split("_")
        pmid = fitems[1]
        scnt = 0
        for l in infile:
          line = l.strip()
          if len(line) < 1:
            continue
          scnt += 1
          items = line.split("\t")
          words = items[1].split("~~")
          # add any new S words to the vocab
          for w in words:
            if w not in voc_idx:
              voc_idx[w] = vstart
              idx_voc[vstart] = w
              if w=="Bacillus" or w=="Zur" or w=="Rhizobium" or w=="FAKE":
                print "Added word", w, "at pos", vstart
              if vstart == 63 or vstart==561:
                print "word", w, "added at", vstart
              vstart += 1
          s_label = pmid+"_"+str(scnt)
          sentence_idx[s_label] = n_sentences
          idx_sentence[n_sentences] = s_label
          all_sentences[s_label] = words
          #print "S label:", s_label
          #print "   words:", words
          n_sentences+=1
  return n_sentences

###########################################################################################################
test_name = sys.argv[1]
if len(sys.argv) > 2:
  otype = sys.argv[2]+"_"
else:
  otype = ""
print "otype:", otype

update_AG = True
use_indicator = False

if update_AG:
  at_odir = "testing_author_topics_predictions"
  at_idir = "author_topics_predictions"
else:
  at_odir = "testing_topics_predictions"
  at_idir = "topics_predictions"
  
num_iterations = 25

# For collectf:
annot_file = "/projects/document_repositories/repository_collectf/sent_answer_keys/"
#annot_file += "pruned_n3_by_topics_collect_abstracts_GOs.txt"
annot_file += "collect_abstracts_GOs.txt"
(annot_idx, idx_annot) = get_column(annot_file, 0)

root_dir = "/projects/test_directories_collectf/" + test_name + "/"
section = "abstract"
(section_dir, corpora_dir, gm_dir) = setup_doc_dirs(root_dir, section)
dcname = os.path.join(root_dir, "input_data/GO/go_prok_mol_bio2_"+otype+"dc_all.txt")
if use_indicator:
  lhsname = os.path.join(root_dir, "input_data/GO/go_prok_mol_bio2_"+otype+"all_lhs.txt")
else:
  lhsname = os.path.join(root_dir, "input_data/GO/go_prok_mol_bio2_"+otype+"all.txt")
#odirname is where the parsed sentences are
audirname = "authors"
autdirname = "authors_test"
odirname = "outdir"
otdirname = "outdir_test"

stime = 5
test_onts_out = ["GO:0009399", "GO:0009432", "GO:0043565", "GO:0051301"]
#test_onts_out = []

# For synthetic data:
#corpora_dir = "generated_documents/"+ test_name # docs_10_authors_20_topics_10"
#dcname = os.path.join(corpora_dir, "ontology_dc_all.txt")
#if use_indicator:
  #lhsname = os.path.join(corpora_dir, "ontology_all_lhs.txt")
#else:
  #lhsname = os.path.join(corpora_dir, "ontology_all.txt")
#section = "abstract"
#odirname is where the parsed sentences are
#odirname = "outdir"
#stime = 1

# For methods as one corpus
#root_dir = "/projects/test_directories/" + test_name + "/"
#section = "method" # Not plural
#(section_dir, corpora_dir, gm_dir) = setup_doc_dirs(root_dir, section)
#dcname = os.path.join(root_dir, "input_data/ECO/eco_orig_"+otype+"dc_all.txt")
#if use_indicator:
  #lhsname = os.path.join(root_dir, "input_data/ECO/eco_orig_"+otype+"all_lhs.txt")
#else:
  #lhsname = os.path.join(root_dir, "input_data/ECO/eco_orig_"+otype+"all.txt")
#odirname is where the parsed sentences are
#odirname = "outdir"
#stime = 1
#test_onts_out = ["ECO:0000085", "ECO:0000225", "ECO:0001828", "ECO:0005554"]

training_dir = os.path.join(gm_dir, at_idir)
if not os.path.exists(gm_dir):
  os.mkdir(gm_dir)
output_dir = os.path.join(gm_dir, at_odir)
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
  
  
psc = 0.00000001
#psc = 1.0 

print "Corpora dir:", corpora_dir
print "Using input files:"
print dcname, lhsname


# Get the vocabulary from the ontology
(voc_idx, idx_voc) = get_column(dcname, 0)
ovocab_words = voc_idx.keys()
onum_words = len(ovocab_words)
print "# vocab from ontology", onum_words
  

# Get the ontology ids and their original LL's
if use_indicator:
  (ont_idx, idx_ont, ont_lls) = get_ont_lls(lhsname)
else:
  (ont_idx, idx_ont, ont_words) = get_ont_words(lhsname)
print "Use indicator?", use_indicator

ont_entries = ont_idx.keys()
num_ont_entries = len(ont_entries)

test_onts_out_ids = []
for o in test_onts_out:
    o_idx = ont_idx[o]
    test_onts_out_ids.append(o_idx)
print test_onts_out, "indexes:", test_onts_out_ids

# Get document and author information
doc_authors = {}
doc_idx = {}
idx_doc = {}
author_idx = {}
idx_author = {} 
num_authors_training = get_author_info(corpora_dir, author_idx, idx_author, doc_authors, doc_idx, idx_doc, audirname)
num_authors = get_author_info(corpora_dir, author_idx, idx_author, doc_authors, doc_idx, idx_doc, autdirname)
print "# au training:", num_authors_training, "total:", num_authors
the_pmids = doc_authors.keys()
num_docs = len(the_pmids)

# Get the sentences and their words. This will add to the vocabulary.
all_sentences = {}
sentence_idx = {}
idx_sentence = {}
num_sentences_training = get_sentences(corpora_dir, sentence_idx, idx_sentence, all_sentences, voc_idx, idx_voc, section, odirname)
num_sentences = get_sentences(corpora_dir, sentence_idx, idx_sentence, all_sentences, voc_idx, idx_voc, section, otdirname)
print "#S training:", num_sentences_training, "# total S:", num_sentences
#print "SENTENCES:", sentence_idx

# After reading in the sentences have the complete vocabulary
vocab_words = voc_idx.keys()
num_words = len(vocab_words)


print "Num ontology entries:", num_ont_entries
print "Total num docs:", num_docs
print "Total num authors:", num_authors
print "Total num sentences:", num_sentences
print "Total num words:", num_words


# Save run config
cfname = os.path.join(output_dir, "run_config.txt")
with open(cfname, "w") as fpo:
  fpo.write("%s\n" % test_name)
  fpo.write("%s\n" % corpora_dir)
  fpo.write("Parsed Training S subdir: %s\n" % odirname)
  fpo.write("Parsed Testing S subdir: %s\n" % otdirname)
  fpo.write("Training matrices input dir: %s\n" % training_dir)
  fpo.write("Indicator? %s\n" % str(use_indicator))
  fpo.write("Num ontology entries: %d\n" % num_ont_entries)
  fpo.write("Num docs: %d\n" % num_docs)
  fpo.write("Num authors: %d\n" % num_authors)
  fpo.write("Num sentences: %d\n" % num_sentences)
  fpo.write("Num words: %d\n" % num_words)

#test_words = ["reaction", "nitrogen", "fixation", "process", "atmosphere"]
#test_words = ["SPR", "analysis", "Thus"]
test_words = []
for tw in test_words:
  print tw, voc_idx[tw], idx_voc[voc_idx[tw]]

#print "Word for 15012:", idx_voc[15012], idx_voc[15013], idx_voc[15014]

time.sleep(1)

#########################################################################################
# Create matrices (ndarrays)

######################################################
# Constant count matrices
# wS [w x S] - # times a word appears in a sentence
wS = np.zeros(shape=(num_words, num_sentences)) #, dtype=np.uint32) # was 16
# SD [S x D] - presence/absence of each S in each doc
SD = np.zeros(shape=(num_sentences, num_docs)) #, dtype=np.uint32) # was 8

# Note: uint8 etc. take less space but dot product with integers is so MUCH slower
# https://stackoverflow.com/questions/11856293/numpy-dot-product-very-slow-using-ints
# Apparently different code paths are taken

# Fill in these matrices
#print "Fill wS and SD"
for s_label in all_sentences:
  s_idx = sentence_idx[s_label]
  s_words = all_sentences[s_label]
  #print "Sentence: ", s_label, s_idx, s_words
  # Set the word counts
  for w in s_words:
    w_idx = voc_idx[w]
    wS[w_idx, s_idx] += 1.0
    #print "wS", w, w_idx, ",", s_idx, '=', wS[w_idx, s_idx]
  # Indicate the sentence is in the document. The PMID is part of the sentence label
  (pmid, scnt) = s_label.split("_")
  d_idx = doc_idx[pmid]
  SD[s_idx, d_idx] = 1
  #print "SD", s_label, pmid, s_idx, ",", d_idx, "=", SD[s_idx, d_idx]
#print wS[130:133, 16131:16134]
#word_at = idx_voc[131] # gene
#S_at = idx_sentence[16132] # occurs 2x in 161132, once in 16131
#print word_at, S_at, all_sentences[S_at]


# o_Gw [G x w] - # times a word in original ont entry definition
# Words not occurring get pseudocount
o_Gw = np.full(shape=(num_ont_entries, num_words), fill_value=psc) # if log do np.log(psc)
for ont_id in ont_entries:
  o_idx = ont_idx[ont_id]
  if use_indicator:
    words = ont_lls[ont_id] # returns dictionary
  else:
    words = ont_words[ont_id] # returns list
  #if ont_id in test_onts_out:
    #print ont_id, o_idx, words
  for w in words:
    if w==def_llword:
      continue
    w_idx = voc_idx[w]
    o_Gw[o_idx, w_idx] += 1
    #if ont_id in test_onts_out:
      #print ont_id, w, o_idx, w_idx, o_Gw[o_idx, w_idx]


# DA [D x A] - 1/#authors_of_doc, 0 otherwise
DA = np.zeros(shape=(num_docs, num_authors))
# AD [A x D] - presence/absence of each author for each doc
# Same as DA except unnormalized and dimensions reversed
AD = np.zeros(shape=(num_authors, num_docs)) #, dtype=np.uint32) # was 8
# Fill in these matrices
#print "Fill DA, AD"
for pmid in the_pmids:
  d_idx = doc_idx[pmid]
  authors = doc_authors[pmid]
  num_d_authors = len(authors)
  #print pmid, authors
  for a in authors:
    a_idx = author_idx[a]
    DA[d_idx, a_idx] = 1.0/float(num_d_authors)
    AD[a_idx, d_idx] = 1
    #print "DA", pmid, a, d_idx, ",", a_idx, DA[d_idx, a_idx]
    #print "AD", a, pmid, a_idx, ",", d_idx, AD[a_idx, d_idx]


######################################################
# Operational count matrices

# Gw [G x w] - # times each word seen in current ont entry definition
Gw = np.zeros(shape=(num_ont_entries, num_words))
gw_name = os.path.join(training_dir, "Gw_matrix.txt")
print "GW in", gw_name
with open(gw_name, "r") as fpi:
  for l in fpi:
    line = l.strip()
    if len(line) < 1:
      continue
    #print line
    items = line.split("\t")
    # If we have no words, this ont id had all 0's for its word counts
    if len(items) >= 2:
      the_ont_id = items[0]
      owords_str = items[1]
      the_ont_idx = ont_idx[the_ont_id]
      #print "ONT:", the_ont_id, the_ont_idx
      owordvals = owords_str.split("~~")
      for owv in owordvals:
        #print "OWV:", owv
        (the_word, the_val_str) = owv.split(":;:")
        the_word_idx = voc_idx[the_word]
        the_val = float(the_val_str)
        #print "  ", the_word, the_word_idx, the_val
        Gw[the_ont_idx, the_word_idx] = the_val

# AG [A x G] - # times author is author of a doc with a S paired to ont entry
# one's in training AG = np.ones(shape=(num_authors, num_ont_entries)) #, dtype=np.uint32) # was 16
# Training AG = np.zeros(shape=(num_authors, num_ont_entries)) #, dtype=np.uint32) # was 16
AG = np.full(shape=(num_authors, num_ont_entries), fill_value=psc) # Since only wrote out counts
ag_name = os.path.join(training_dir, "AG_matrix.txt")
print "AG in", ag_name
with open(ag_name, "r") as fpi:
  for l in fpi:
    line = l.strip()
    if len(line) < 1:
      continue
    items = line.split("\t")
    if len(items) >= 2:
      the_author = items[0]
      aonts_str = items[1]
      the_author_idx = author_idx[the_author]
      #print "AU:", the_author, the_author_idx
      aontvals = aonts_str.split("~~")
      for aov in aontvals:
        (the_ont_id, the_val_str) = aov.split(":;:")
        the_ont_idx = ont_idx[the_ont_id]
        the_val = float(the_val_str)
        #print "  ", the_ont_id, the_ont_idx, the_val
        AG[the_author_idx, the_ont_idx] += the_val


######################################################
# Other matrices used in computation

# NOTE: it is much more efficient to allocate all arrays and reuse them
# rather than have Python/numpy allocate/deallocate them in a loop
# which will quickly eat up memory.

# ?? may not need both
lp_Gw = np.zeros(shape=(num_ont_entries, num_words))

# LPSG -- log of P(S|G) [G x S]
LPSG = np.zeros(shape=(num_ont_entries, num_sentences))

# ?? may not need this one and LPSG
# P(S|G) [G x S]
PSG = np.zeros(shape=(num_ont_entries, num_sentences))

# ??
p_AG = np.zeros(shape=(num_authors, num_ont_entries))

# PGD [D x G] -- prob of ont entry given a document (given the authors)
PGD = np.zeros(shape=(num_docs, num_ont_entries))

# LPDG [G x D] -- 
LPDG = np.zeros(shape=(num_ont_entries, num_docs))

# ?? may not need both
PDG = np.zeros(shape=(num_ont_entries, num_docs))

# Likelihood?
Likelihood = np.zeros(shape=(num_docs, num_docs))

# PGS [S x G] -- likelihood of authors of doc to a S
PGS = np.zeros(shape=(num_sentences, num_ont_entries))
#??
LPGS = np.zeros(shape=(num_sentences, num_ont_entries))
# Unclear if need separate array for transpose of if it will do a view

# LPL [G x S] -- log likelihood product
LPL = np.zeros(shape=(num_ont_entries, num_sentences))
#?
PL = np.zeros(shape=(num_ont_entries, num_sentences))

# MaxS [1 x S] -- indices of ont entries that are maximum
MaxS = np.zeros(shape=(num_sentences), dtype=np.int64)
prevMaxS = np.zeros(shape=(num_sentences), dtype=np.int64)

# MaxGS [G x S] -- 1/0 - identifies which ont is best for an S
MaxGS = np.zeros(shape=(num_ont_entries, num_sentences)) #, dtype=np.uint32) # was 8

# MaxGD [G x D] -- # count of S in D for which the ont entry maximized
# the likelihood of S
MaxGD = np.zeros(shape=(num_ont_entries, num_docs)) #, dtype=np.uint32)

#########################

print "Matrices created"

prev_lh = 0.0
prev_lhs = []
iteration = 1

for i in range(iteration):
  print "Begin iteration:", iteration
  foname = os.path.join(output_dir, "testing_sent_onts_iter_"+str(iteration)+".txt")
  sent_onts = {}
  
  # Compute P(S|G)
  print "Compute P(S|G)"
  np.add(Gw, o_Gw, out=Gw)
  gwname = os.path.join(output_dir, "Gw_initial_iter_"+str(iteration)+".txt")
  with open(gwname, "w") as gwo:
    for o in range(len(test_onts_out_ids)):
      gidx = test_onts_out_ids[o]
      ostr = test_onts_out[o]+","+str(np.max(Gw[gidx]))+","+str(np.min(Gw[gidx]))
      for gpos in range(len(Gw[gidx])):
        ostr+=","+str(Gw[gidx,gpos])
      gwo.write("%s\n" % ostr)
  gwo.close()
  #for o in range(len(test_onts_out_ids)):
  #  gidx = test_onts_out_ids[o]
  #  for tw in test_words:
  #    twidx = voc_idx[tw]
  #    print test_onts_out[o], gidx, "--", tw, twidx, "o_Gw:", o_Gw[gidx, twidx], "Gw:", Gw[gidx, twidx]
      
  
  # https://stackoverflow.com/questions/16202348/numpy-divide-row-by-row-sum
  lp_Gw = Gw/Gw.sum(axis=1, keepdims=True) # this is f_Gw
  lp_Gw = np.log(lp_Gw)
  LPSG = np.dot(lp_Gw, wS)
  LPSG[LPSG<-700]= -700 # to prevent underflow
  PSG = np.exp(LPSG) # [G x S]
  
  # Compute P(G|D)
  print "Compute P(G|D)"
  print "Shape of AG:", AG.shape
  print "AG:", AG[0:10, 0:10]
  p_AG = AG/AG.sum(axis=1, keepdims=True)
  print "p_AG:", p_AG[0:10, 0:10]
  #for i in range(p_AG.shape[0]-1): # one less than last
  #  print "Entropy of row", i, -np.sum(p_AG[i] * np.log2(p_AG[i]))
  #for i in range(p_AG.shape[0]-1): # one less than last
  #  for j in range(i+1, p_AG.shape[0]):
  #    print "i", i, "j", j
  #    print "p_AG row", i, "min:", np.min(p_AG[i]), "max:", np.max(p_AG[i]), "; p_AG row", j, "min:", np.min(p_AG[j]), "max:", np.max(p_AG[j]), "Eucl:", np.linalg.norm(p_AG[j]-p_AG[i])
  
  PGD = np.dot(DA, p_AG)  # [D x G]

  
  # Compute the likelihood
  print "Compute the likelihood"
  LPDG = np.dot(LPSG, SD) # [G x D]  
  LPDG[LPDG<-700]= -700 # to prevent underflow
  PDG = np.exp(LPDG) # [G x D]
  
  # Likelihood array
  print "  dot to get likelihood", PGD.shape, PDG.shape
  Likelihood = np.dot(PGD, PDG)
  lh = np.prod(np.diag(Likelihood))
  temp = np.diag(Likelihood)
  print "   max diag value:", np.amax(temp)
  num_non_zero = np.count_nonzero(temp)
  print "  LH value:", lh, "prev LH:", prev_lh
  num_diag = np.size(temp)
  print "      size:", num_diag, "# zero:", num_diag-num_non_zero
  print "      ", temp[0:100]
  prev_lhs.append(lh)
  prev_lh = lh
  
  # Computing the maximum likelihood
  print "Compute the maximum likelihood"
  
  # Compute the likelihood product PL
  print "Compute the likelihood product PL"
  print "  Propagate <P(G|A)>, dot SD and PGD", SD.shape, PGD.shape
  print "  SD:" 
  print SD[0:10, 0:10] # This looks right
  print "  PGD:"
  print PGD[0:10, 0:10] # Lots of 0's but some values
  num_non_zero = np.count_nonzero(PGD)
  print "  # zero elems", np.size(PGD) - num_non_zero
  #PGD[PGD<9.8596765437597708e-305] = 9.8596765437597708e-305
  #print "  PGD again:"
  #print PGD[0:10, 0:10] # Lots of 0's but some values
  LPGS = np.dot(SD, np.log(PGD)) # [S x G]
  # 1st loop, PGS elements I see are = 0.00012786
  # 2nd loop, PGS has 0's, which fail on log below
  print "   LPGS:", np.size(LPGS) 
  print LPGS[0:10, 0:10]
  num_m700 = (LPGS==-700).sum()
  print "   max value in LPGS:", np.amax(LPGS), "# -700:", num_m700
  
  #print "  log PGS"
  #num_non_zero = np.count_nonzero(PGS)
  #print "  # zero elems", np.size(PGS) - num_non_zero
  #LPGS = np.log(PGS) # divide by 0 in log = trying to take log(0)
  LPL = np.add(LPGS.transpose(), LPSG)
  print "  exp to get PL"
  LPL[LPL<-700]= -700 # to prevent underflow
  PL = np.exp(LPL) # [G x S]

  
  # Determine maximum Ontology id for each S
  print "Determine maximum ontology index for each S"
  # Note:write up says "row-wise" but we want the max in each S, which is by column
  # See below after stackoverflow link
  MaxS = np.argmax(PL, axis=0) # [1 x S], has indices of G that are maximum
  #print "  MaxS:", MaxS[0:10]
  #mvtemp = np.amax(PL[:,0])
  #print "  Max value at 1525, 0:", PL[1525, 0], mvtemp

  print "Output S,Ont pairs"
  with open(foname, "w") as fpo:  
    for i in range(len(MaxS)):
      slabel = idx_sentence[i]
      ont_id = idx_ont[MaxS[i]]
      #print "S:", slabel, "->", ont_id
      ostr = slabel+"\t"+ont_id + "\t" + str(PL[MaxS[i], i])
      fpo.write("%s\n" % ostr)
  
  print "Annot PL"
  plfname = os.path.join(output_dir, "annot_PL.txt")
  with open(plfname, "w") as fpo:
    ostr = "GO"
    for i in range(num_ont_entries):
      ostr += "\t"+idx_ont[i];
    fpo.write("%s\n" % ostr)
    
    for i in range(num_sentences):
      ostr = ""
      if i in idx_annot:
        ostr = idx_annot[i]
        for j in range(num_ont_entries):
          ostr+="\t"+str(PL[j, i])
        fpo.write("%s\n" % ostr)
      
print "Done"
print "LHs:", prev_lhs
