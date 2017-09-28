# Generative model for author-topic model
This is an attempt to summarize our knowledge and ideas about a generative model for the author-topic conception of documents.

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

```P(G|D)=<P(G|A)>, A∈D```

### Data
We have different data sources available. Most notably:

- [(G,w)] a list of pairs of GO entries and words in their natural languate definition
- [(D,A)] a list of pairs of documents and their authors
- [(D,S)] a list of pairs of sentences in documents
- [(G_x,G_y)] a list of pairs of GO entries with direct connection (parent-offspring relationship) in the GO topology

## Likelihood
Given the above definitions for the generative model and data sources available, we can define the likelihood of our system (i.e. the probability of the data [whole set of documents] given our current hypothesis/model [P(G|A) & P(S|G)]) as follows:

[5] <```P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G)  \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right )```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)" title="P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G) \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right )" /></a>

which essentially states that the likelihood of the data is the product (for all docs) of how well on average the sentences of the document fit each of the GO entries (times the expectation that each of those GO entries has been authored by authors in the document). This whole term is multiplied by a second term specifying (for all GO entries) how well defined each GO entries is.

### (loose) Interpretation
In an ideal scenario, we would have super-specific, almost non-overlapping GO entries (maximizing the second term: P(w|G)). That is, we would expect that each GO entry used a completely disjoint set of words in its operational definition.

Furthermore (maximizing first equation term), we would hope that GO entries be highly specific to authors (i.e. each author has its own preferred GO entry, maximizing P(G|A), so that a document has a clearly defined set of GO entries), and when dealing with a document from the author, we would like that the *relevant* GO entries fit squarely (word-wise) with sentences in the document (maximizing P(S|G)).

In other words, maximizing the second term ensures that the GO entry definitions are sharp. Maximizing the first term ensures that GO entries are properly assigned to authors *and* that they adequately match text in author's manuscript.

## Likelihood maximization
Given the above, our goal is to maximize the likelihood P(D|H) of the model (which to some degree is conceptually equivalent to making a point estimate of the posterior probability P(H|D)), since likelihood and posterior are connected directly through Bayes:

[6] <```P(H|Data)=\frac{P(Data|H)P(H)}{P(Data)}```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(H|Data)=\frac{P(Data|H)P(H)}{P(Data)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(H|Data)=\frac{P(Data|H)P(H)}{P(Data)}" title="P(H|Data)=\frac{P(Data|H)P(H)}{P(Data)}" /></a>

Having formulated the problem, we can in fact go through either route to obtain a solution.

### Maximum likelihood
If we wish to maximize P(Data|H), we seek an algorithm capable of updating the terms involved in H [P(G|A) and P(S|G)], so that P(Data|H) becomes maximal. This will us a point estimate for the model H that makes the likelihood maximal.

#### Contraints in ML
Maximizing the likelihood as defined above (Eq. 5) does not guarantee that we obtain sensible solutions, since there may be unobvious local optima. We do, however, have some ideas of how things should look in a working solution. For instance, we know that most authors only deal with a handful of GO entries, so P(G|A) should be a very sparse distribution (given that there are ~7,000 GO entries). Similarly, we expect that most GO entries will only use a small subset of the vocabulary and hence P(w|G) should also be sparse.

The L^2 norm (sqrt of sum of components) provides an indication of the *size* of a vector, and hence a strategy that minimizes the L^2 norm would lead to categorical distributions (i.e. vectors) were only a few components have non-zero probability.

If we define:

[7] <```F_1=L^2\left ( P(G|A) \right )```>

<a href="https://www.codecogs.com/eqnedit.php?latex=F_1=L^2\left&space;(&space;P(G|A)&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F_1=L^2\left&space;(&space;P(G|A)&space;\right&space;)" title="F_1=\frac{1}{L^2\left ( P(G|A) \right )}" /></a>

and an equivalent F_2 for P(w|G), then we can revise the likelihood that we want to maximize to:

[8]<```P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G)  \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right ) + \lambda_1 F_1(P(G|A)) + \lambda_2 F_2(P(w|G)) ```>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)&space;&plus;&space;\lambda_1&space;F_1(P(G|A))&space;&plus;&space;\lambda_2&space;F_2(P(w|G))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(Data|H)=\prod_D&space;\left&space;(\sum_G&space;\left&space;(&space;\left&space;\langle&space;P(G|A)&space;\right&space;\rangle_{A&space;\in&space;D}&space;\prod_{S&space;\in&space;D}&space;P(S|G)&space;\right&space;)&space;\right&space;)&space;\prod_G&space;\left&space;(&space;\prod_{w&space;\in&space;G}&space;P(w|G)&space;\right&space;)&space;&plus;&space;\lambda_1&space;F_1(P(G|A))&space;&plus;&space;\lambda_2&space;F_2(P(w|G))" title="P(Data|H)=\prod_D \left (\sum_G \left ( \left \langle P(G|A) \right \rangle_{A \in D} \prod_{S \in D} P(S|G) \right ) \right ) \prod_G \left ( \prod_{w \in G} P(w|G) \right ) + \lambda_1 F_1(P(G|A)) + \lambda_2 F_2(P(w|G))" /></a>

where we are essentially tagging on a term to make the solution to the optimization process sparse in P(G|A) and P(w|G).

#### Relationship with Bayesian inference
In Bayesian inference we do not aim at obtaining a point estimate of P(D|H) but rather at sampling (or formally deriving) the posterior probability distribution P(H|Data). Typically, thereafter, we will compute some value for P(H|Data) that summarizes it adequately or that we otherwise are interested in (e.g. the mean and stdev).

We compute P(H|Data) using Eq. 6, which introduces the need for estimates of P(H) and P(Data). In straightforward ML (without constraints), the ratio P(H)/P(Data) in Eq. 6 is assumed to be constant, and we focus only on maximizing P(Data|H). The introduction of constraints (or penalties) in ML can be seen as equivalent to the consideration of priors in Bayesian inference. For instance, in Bayesian inference we may set up a prior for P(G|A) that explicitly defines the form of this distribution and is parametrized by some estimate (e.g. by manually assessing a few authors) of the number of GO topics covered by authors. The "shape" for P(G|A) induced by the prior therefore acts in Bayesian inference as the equivalent of the F_1 constraint (Eq. 7).

#### Formulation as an EM algorithm
Given an expression for the likelihood that we wish to optimize, ML can be folded into an expectation-maximization algorithm as follows:

- **Maximization step:** given some value of P(G|A) and P(w|G), compute for all G P(S|G) and identify, for each S in each D, the assignment (G,S) that maximizes the likelihood [that is, assume we have an estimate for P(G|A) and P(w|G), and find the placement of GO entries among sentences that maximizes the likelihood function]. Note that this enforces 1 GO entry per sentence.
- **Expectation step:** given an assignment (G,S) for all sentences (S) in all documents (D), estimate P(w|G) or P(G|A).
	- P(w|G) is estimated by adding to the GO entry word count the words in the sentences paired to that GO entry, times the likelihood P(S|G) of that match
	- P(G|A) is estimated by adding to the author-entry tally the number of times an author A is the author of a document D in which a sentence S has been paired to the GO entry G, modulated again by the likelihood P(S|G).

