# Generative model matrix computation

## Definitions and notation
We define four main entities in the generative models:

- **Documents** (D) are co-authored by multiple authors and thus define a *uniform distribution over authors*
- **Authors** (A) work on several topics (GO entries) and hence have an associated *distribution over GO entries*
- **GO entries** (G) define, using natural language, specific topics in biology and hence determine a *probability distribution over words*

### Generative model
Our generative model is hence of the form:

[1] <```D -> A -> G -> S```>

In other words, we first pick a document (D), from it we pick an author (A) of the document, we then pick a topic (G) the author "likes" and from it we generate a setence (S).

### Goal
Given the generative model above, our goal is to obtain an expression that allows us to compute:

[2] <```P(G|S, D=[A]))```>

that is, the probability of a specific GO entry (G) given a sentence (S) in a document (D), conceptualized as a list of authors ([A]).

### Assumptions

We assume that the document is a uniform distribution over authors, defined on the authors of the document. Namely:

[3] <```P(A|D)=\begin{Bmatrix}  1/|A|& A \in D \\  0 & otherwise \end{Bmatrix}```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(A|D)=\begin{Bmatrix}&space;1/|A|&&space;A&space;\in&space;D&space;\\&space;0&space;&&space;otherwise&space;\end{Bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|D)=\begin{Bmatrix}&space;1/|A|&&space;A&space;\in&space;D&space;\\&space;0&space;&&space;otherwise&space;\end{Bmatrix}" title="P(A|D)=\begin{Bmatrix} 1/|A|& A \in D \\ 0 & otherwise \end{Bmatrix}" /></a>

We also assume that the words in a sentence are independent (i.e. bag-of-words), and hence:

[4] <```P(S|G)=\prod_{{w \in S}}P(w|G)```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(S|G)=\prod_{{w&space;\in&space;S}}P(w|G)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(S|G)=\prod_{{w&space;\in&space;S}}P(w|G)" title="P(S|G)=\prod_{{w \in S}}P(w|G)" /></a>

Given these, from the generative model we have outlined, we know that the other term that we need to be able to compute (and update) in order to obtain P(G|S,D) is the probability of the GO entry given an author P(G|A).

For the sake of simplicity, we'll assume that the probability of a GO entry given an author A propagates into a document (author list [A]) as the (unweighted) expectation over the authors of the document:

[5] <```P(G|D)=<P(G|A)>, A∈D```>

### Data
We have different data sources available. Most notably:

- [(G,w)] a list of pairs of GO entries and words in their natural languate definition
- [(D,A)] a list of pairs of documents and their authors
- [(D,S)] a list of pairs of sentences in documents
- [(G_x,G_y)] a list of pairs of GO entries with direct connection (parent-offspring relationship) in the GO topology

## Likelihood
Given the above definitions for the generative model and data sources available, we can define the likelihood of our system (i.e. the probability of the data [whole set of documents] given our current hypothesis/model [P(G|A) & P(S|G)]) as follows:

[6]<```P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G)  \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right ) + \lambda_1 F_1(P(G|A)) + \lambda_2 F_2(P(w|G)) ```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)&space;&plus;&space;\lambda_1&space;F_1(P(G|A))&space;&plus;&space;\lambda_2&space;F_2(P(w|G))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)&space;&plus;&space;\lambda_1&space;F_1(P(G|A))&space;&plus;&space;\lambda_2&space;F_2(P(w|G))" title="P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G) \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right ) + \lambda_1 F_1(P(G|A)) + \lambda_2 F_2(P(w|G))" /></a>

in this likelihood we have a main component on the left-hand side that captures the nature of the generative model. Essentially, it states that the likelihood of the data (i.e. all documents) is a function of the likelihood of GO entries given the authors (i.e. how much the document *D* authors *[A]* tend to write about a GO entry *G*) times the likelihood of the sentence *S* given the GO entry *G*.

The terms on the right-hand side of the equation represent regularization contraints that we add to the likelihood based on a few *known* properties of the system:

- The product over G of P(w|G), based on the *original* words *w* \in *G*, imposes a restriction on GO entries, so that they cannot *run wild* during the optimization (they must still tend to recognize the original GO entry).
- The functions F_1 and F_2 are of the form:

	[7] <```F_1=L^2\left ( P(G|A) \right )```>

    <a href="https://www.codecogs.com/eqnedit.php?latex=F_1=L^2\left&space;(&space;P(G|A)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_1=L^2\left&space;(&space;P(G|A)&space;\right&space;)" title="F_1=\frac{1}{L^2\left ( P(G|A) \right )}" /></a>

    They impose the restriction that the L^2 norm (sqrt of sum of components) be large, leading to sparse distributions of words over GO entries and GO entries over authors. This reflects our intuition that authors typically work (and hence write about) a few GO entries, and that GO entries can be typically defined using a relatively small amount of words.

- Additional regularization terms may be considered. For instance, one may want to specify that GO entries be as orthogonal as possible in their distribution over words (i.e. little overlap), or that such an overlap be proportional to the topological distance of GO entries on the ontology.

## Likelihood maximization
Given the above, our goal is to maximize the likelihood P(D|H) of the model, and we will make use of Expectation Maximization to maximize the likelihood. For the EM approach, we will consider our visible data to be the documents and their sentences, and our latent variable the GO entries that map to sentences. To simplify, for now, the EM setup, we will consider a simplified version of the likelihood without the penalties:

[8] <```P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G)  \right ) \right ) ```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)" title="P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G) \right ) \right )" /></a>

In this equation, the likelihood is a linear combination of the terms inside the sum over *G*. Furthermore, since P(G|A) is constant within the internal product over S \in D, it can be pushed inside the product without loss of generality (formally it gets in exponentiated to 1/{|S| \in D}). Hence, for any given sentence, it suffices to maximize ```<P(G|A)>·P(S|G)``` in order to guarantee that the likelihood is maximized.

### EM approach
The EM approach will consist of two distinc steps:

#### Expectation
In the expectation step, we will compute the likelihood (i.e. ```<P(G|A)>·P(S|G)```) and determine the most likely assignment (S,G) [i.e. the *G* that maximizes the likelihood of *S*]. This is a "hard-max" approach. Alternatively, we could compute the likelihood of each (S,G) pair and use that for maximization (soft-max).

#### Maximization
In the maximization step, we will use the (S,G) assignments to reassess P(G|A) and P(S|G). We will do so by counting the number of sentences mapped to a particular *G* that are present in documents authored by author *A* [P(G|A)], and the number of times a word *w* belongs to a sentence mapped to *G* [P(w|G)]. Again, using a soft-max approach this would be done by using the likelihood of each (S,G) pairing, instead of a single (S,G) assignment.

#### Convergence assessment
Convergence will be assessed by evaluating the difference, at each step, between (S,G) pairings. Once the EM process converges, the (S,G) pairings should not change (although we note that using the hard-max could lead to oscillatory behavior). More generically, convergence could be assessed on an evaluation of the full likelihood function, which is detailed below.

## Matrix computation

### Computing the likelihood

#### Notation

In the following, matrices are named following a row-column notation. For instance, a matrix named *Gw* has dimensions [G x w], where *G* is the number of rows and *w* the number of columns. Modifiers are inserted on the left-hand side, followed by underscore (_). Hence *o_Gw* denotes *original* *Gw*. This does not apply to derived matrices for which we have generated names (e.g. PSG stands for P(S|G)).

#### Operational count matrices

We define two operational count matrices. These are the matrices that will be updated in the EM procedure:

- ```Gw [G x w]``` contains the number of times each word (*w*) is seen in the *current* GO entry (*G*) definition
	- This matrix is originally initialized to zeroes
	- This matrix *will* be updated
- ```AG [A x G]``` contains the number of times an author (*A*) is the author of a document (*D*) in which a sentence (*S*) has been paired to the GO entry (*G*)
	- This matrix is initialized to ones
	- This matrix *will* be updated

#### Constant count matrices

We define several constant count matrices. These matrices will be set when the corpus is read and then held constant during the EM procedure:

- ```o_Gw [G x w]``` contains the number of times each word (*w*) is seen in the *original* GO entry (*G*) definition
	- This matrix is initialized from the *original* GO entry and *held constant*
	- Words not occurring in the GO entry definition are given a pseudocount value, as described in the [Bayesian approach](Bayesian.md) outline.

- ```wS [w x S]``` contains the number of times a word appears in a sentence
	- This matrix is initialized with word counts when the corpus is read
- ```DA [D x A]``` contains 1/{# authors in D, if A \in D}, 0 otherwise
	- This matrix is initialized with normalized author counts when the corpus is read
- ```SD [SxD]``` contains the presence/absence (1/0) of each sentence in each document
	- This matrix is initialized with sentence-document memberships when the corpus is read
- ```AD [A x D]``` contains 1 if A \in D, zero otherwise (unnormalized author-document count)

#### Computing P(S|G)

To compute *P(S|G)*, we first must bring together the original and document-based counts for words and GO entries:

```Gw = Gw + o_Gw```, where '+' is the component-wise sum

Then, we transform this count matrix into a likelihood matrix using the relative frequency of the words as the MLE:

(9) <```P(w_k^j|GO_i)=\frac{|w_k \in GO_i|+Pc}{\sum_{h=1}^{WSP}\left (|w_h \in GO_i|+Pc  \right )}```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(w_k^j|GO_i)=\frac{|w_k&space;\in&space;GO_i|&plus;Pc}{\sum_{h=1}^{WSP}\left&space;(|w_h&space;\in&space;GO_i|&plus;Pc&space;\right&space;)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(w_k^j|GO_i)=\frac{|w_k&space;\in&space;GO_i|&plus;Pc}{\sum_{h=1}^{WSP}\left&space;(|w_h&space;\in&space;GO_i|&plus;Pc&space;\right&space;)}" title="P(w_k^j|GO_i)=\frac{|w_k \in GO_i|+Pc}{\sum_{h=1}^{WSP}\left (|w_h \in GO_i|+Pc \right )}" /></a>

where WSP denotes the GO-word space. Given the construction of *Gw* above, which already includes the pseudocounts in *o_Gw*, this translates into dividing each component by the row-wise sum in order to obtain p_Gw.

Finally, we further transform *f_Gw* into a log-likehood matrix, by taking the *log* on each component: 

```lp_Gw = log(p_Gw) [G x w]```

Using the transformed *Gw* matrix, we can now compute P(S|G) as follows:

- ```LPSG = lp_Gw x wS [G x S]``` will contain the sum (for each sentence and GO entry) of the product bewteen the log-likelihoods of the words and the number of times they appear in the sentence.
- ```PSG = exp(LPSG) [G x S]``` will contain *P(S|G)*, the probability of a sentence *S* given a GO entry *G* [*exp* is the component-wise exponentiation]

#### Computing ```<P(G|A)>``` as P(G|D)

Given *DA [D x A]* (normalized document authorship matrix) and *AG [A x G]* (author-GO map count), we can compute ```<P(G|A>```, which we can rename *P(G|D)* (for the probability of a GO entry given the document (formally the document's authors)) as follows.

First, just as we did for *Gw*, we need to transform *AG* into a likelihood matrix using the relative frequency of GO entries (*G*) among authors (*A*) as the MLE. Again, to obtain *p_AG* we simply divide each component by the row-wise sum.

Given *p_AG* and *DA*, which contains the normalized membership of each author to each document, we compute PGD ```~<P(G|A)>``` as:

- ```PGD = DA x p_AG [D x G]``` contains the probability of GO entry given a document [formally a document's list of authors]

#### Computing the likelihood

To compute the likelihood, we should first convert *PSG* into a matrix (*PDG*) that contains the product of *PSG* over all sentences in a document. To do this, we first go back to the component-wise log of PSG:

```LPSG = log(PSG) [G x S]```

Given *SD [SxD]* (the indicator matrix of presence/absence of each *S* in each *D*), we compute *PDG* as follows:

- ```LPDG = LPSG x SD [G x D]``` contains, for each GO entry and document, the sum of the products between the log-likelihood of the sentences in the document and their presence in the document
- ```PDG = exp(LPDG) [G x D]``` contains, for each GO entry and document, the product of the likelihoods given that GO entry of the sentences in the document

Once we have PDG, then we just multiply:

```Likelihood = PGD x PDG [DxD]``` that will contain, for each document pair, the sum of the products between the likelihood of G given the document's authors P(G|D) and the likelihood of the document given that G.

The total likelihood will be the diagonal product of this matrix (when the document for both likelihood components is the same).

### Computing the maximum likelihood

For each sentence *S* in a document *D* authored by *[A]*, we wish to find out which GO entry *G* maximizes the product of ```PL=<P(G|A)>·P(S|G)```, which is a sufficient condition to maximize the total likelihood. This can be divided in two steps: (1) find the likelihood component for all ```PL=<P(G|A)>·P(S|G)``` pairs and (2) identify the *G* that maximizes *PL* for each *G*.

#### Computing the likelihood product PL  (expectation)

Given ```P(G|D) [PGD]``` and ```P(S|G) [PSG]``` obtained as described in the above sections, if we wish to compute ```PL``` we only need to revert to their component logs and compute a component-by-component addition. Since PGD has [D x G] dimensions and PSG [G x S] dimensions (and we wish to obtain a [G x S] matrix pairing sentences and GO entries via their likelihood, we first need to propagate the ```<P(G|A)>``` likelihood stored in PGD among the sentences that make up each document. But before doing that, we take the component logs, so we can later add them. Hence, we first compute:

```LPGD = log(PGD) [D x G]``` (component log)

```LPGS = SD x LPGD [S x G]```, where SD is the sentence-document indicator function. This will assign the document (authors) log-likelihood to each constituent sentence

then, we take the component-wise log to get the log-likelihood:

```LPGS = log(PGS)```

and finally we tranpose LPGS and add the two matrices:

```LPL=LPGS' + LPSG [G x S]```

and then exponentiate to implement the product of likelihoods:

```PL=exp(LPL) [G x S]```, where *exp* is the component-wise exponentiation of LPL

#### Maximizing the likelihood product *PL*

Once we have the likelihood product *PL* computed, we just need to take an in-row (i.e. column-wise) maximum to end-up with a matrix (```MaxS [1 x S]```) with the indices (*G*) of the maximum likelihoods for each sentence (*numpy.argmax*). From *MaxS* we can easily derive a ```MaxGS [G x S]``` array in which elements are set to one if *S* is maximized by *G*, and to zero otherwise.

### Updating the model (maximization)

To update the model we need to recompute the two operational count matrices: ```Gw [G x w]``` and ```AG [A x G]```.

#### Updating AG

The AG matrix can be updated in the following way.

Given our sentence-document indicator matrix (*SD*), and our max matrix (*MaxGS*) generated at the end of the expectation step, the product:

```MaxGD = MaxGS x SD [G x D]``` will contain, for each GO entry and document, the count of sentences in *D* for which *G* maximized the likelihood of *S*.

Then, using the indicator function for authors' contributions to documents (*AD*) [A x D] and transposing MaxGD (to get [D x G]), we can obtain:

```AG = AD x MaxGD' [A x G]``` that shall contain the count (normalized by authors' contributions to documents) of GO entries (*G*) mapping to documents authored by each author (*A*).

#### Updating Gw

To update the word-GO count matrix, we use ```MaxGS``` and our word-sentence count matrix (*wS*):

```Gw = MaxGS [G x S] x wS' [w x S] {G x w}``` will contain the updated count of words (*w*) in sentences that have been mapped to the GO entry  (*G*)