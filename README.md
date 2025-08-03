# Single-Cell Foundation Models

Single-cell transcriptomics gives a snapshot of the \~\$20,000-gene expression state of a cell. While the number of possible transcriptomes is astronomically high (in a binary on/off model, it's \$2^{20,000}\$), the space of *biologically plausible* transcriptomes is a tiny, intricately structured manifold. This manifold is shaped by the physical and gene-regulatory laws governing the cell.

Linear methods like Principal Component Analysis (PCA) are great for finding the major axes of variation in this data, but they fall short because biology is fundamentally non-linear. To model complex phenomena like gene XOR gates or the effects of perturbations, you need models that can learn the true, non-linear probability distribution \$p(\text{transcriptome})\$ from which all observed cell states are sampled.

What is commonly referred to as "deep learning" models, with many hidden neurons and non-linear activation functions, can learn arbitrary dependencies between genes. Diffusion models are literally trained to learn the score function, \$\nabla\_x \log p(x)\$, which is the gradient of the log-probability of the data. This allows them to capture the underlying structure of the biological manifold.

---

## Why Foundation Models Outperform PCA

| Capability                  | Why PCA Fails                                                                                                                                                     | How Foundation Models Succeed                                                                                                                        |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Non-linear Dependencies** | PCA is a linear method, assuming additive or independent effects. It can't capture XOR-type interactions where a gene's effect depends on the absence of another. | A diffusion model learns the score of the true data distribution, capturing arbitrary effects.                                      |
| **Conditional Generation**  | PCA has no native mechanism for conditional generation, like predicting the effect of a drug perturbation.                                                        | Models can be explicitly trained to learn the conditional distribution $`p(\text{masked\_gene} \mid \text{visible\_genes}, \text{perturbation})`$.      |
| **Long-range Dependencies** | While its latent components (PCs) are global, they are dense linear combinations of all genes.                                                                    | The attention mechanism allows the model to learn sparse, context-dependent interactions between any two genes, no matter how distant in the genome. |

---

## Petabyte Scale Slices of the Cell

The scale of available single-cell data is staggering. The Sequence Read Archive (SRA) holds over 47 petabytes of raw sequencing data, and its single-cell RNA-seq (scRNA-seq) component is growing exponentially. A key resource is **scBaseCount**, a dataset from the Arc Institute containing nearly 300 million transcriptomes curated from the SRA.

This dataset primarily covers experiments using 10x Genomics technology, which are highly valuable for transcriptomics. Based on a starting point of 330 million 10x-labeled cells in early 2025 and a conservative 20% year-on-year growth, the public archive will hit **one billion cells by mid-2031**. This timeline could accelerate to 2029–2030 as more data types are processed and sequencing throughput increases.

Its worth noting that the data is heavily skewed towards human and mouse cells, mostly from lab-grown embryonic or differentiated cell lines. This creates a significant distribution shift from *in vivo* cells, which are influenced by complex cell-cell signaling and homeostatic rhythms not present in a culture dish.

### Total Cells per Organism in scBaseCount (Top 10)

| Organism                  | Total Cells (Millions) |
| :------------------------ | :--------------------- |
| *Homo sapiens*            | 118.7                  |
| *Mus musculus*            | 98.4                   |
| *Macaca mulatta*          | 4.1                    |
| *Danio rerio*             | 3.7                    |
| *Sus scrofa*              | 1.8                    |
| *Drosophila melanogaster* | 1.7                    |
| *Arabidopsis thaliana*    | 1.6                    |
| *Gallus gallus*           | 0.7                    |
| *Heterocephalus glaber*   | 0.5                    |
| *Bos taurus*              | 0.4                    |

---

## Private Evaluation

I ran two private evaluations on a 40M parameter diffusion-transformer model trained on 3 epochs of 500,000 human brain scRNA-seq data.    

### 1. Linear Covariance Evaluation

**Goal:** Does the model understand that synaptic (SYN) and mitochondrial (MT) gene groups have distinct correlation patterns? In healthy neurons, synaptic genes are co-expressed, while mitochondrial gene expression can indicate stress.

**Method:** I calculated the average pairwise Spearman correlation (\$\bar{\rho}\$) within SYN genes, within MT genes, and between the two groups. I compared the *sign* of these correlations from real data to the model's imputed values for the same cells.

```python
# Core logic of eval_covariance.py

# 1. Sample real expression data (N, G)
X_real = sample_real_data(n_cells)
real_syn, real_mt = X_real[:, syn_indices], X_real[:, mt_indices]

# 2. Calculate correlations on real data
real_within_syn_rho = avg_pairwise_correlation(real_syn)
real_within_mt_rho = avg_pairwise_correlation(real_mt)
real_between_rho = avg_inter_correlation(real_syn, real_mt)

# 3. Mask target genes and get model predictions
X_masked = mask_genes(X_real, syn_indices + mt_indices)
X_pred = model.impute(X_masked)
pred_syn, pred_mt = X_pred[:, syn_indices], X_pred[:, mt_indices]

# 4. Calculate correlations on predicted data
pred_within_syn_rho = avg_pairwise_correlation(pred_syn)
# ... and so on for the other two

# 5. Compare the signs of real vs. predicted correlations
score = np.mean([
    np.sign(real_within_syn_rho) == np.sign(pred_within_syn_rho),
    np.sign(real_within_mt_rho) == np.sign(pred_within_mt_rho),
    np.sign(real_between_rho) == np.sign(pred_between_rho),
])
```

**Results:** The model correctly learned that synaptic genes are positively correlated and that synaptic and mitochondrial genes also move together. However, it failed to capture the near-zero correlation within mitochondrial genes, instead predicting a strong positive one.

| Aspect                       | Real Data (\$\bar{\rho}\$) | Model (\$\bar{\rho}\$) | Sign Match |
| :--------------------------- | :------------------------: | :--------------------: | :--------: |
| Synaptic Genes Together      |            +0.30           |          +0.63         |      O     |
| Mitochondrial Genes Together |            +0.00           |          +0.56         |      x     |
| Synaptic vs. Mitochondrial   |            +0.26           |          +0.62         |      O     |
| **Final Sign-Match Score**   |                            |                        |  **0.67**  |

### 2. Non-Linear XOR Evaluation

**Goal:** Can the model learn a non-linear XOR relationship? In the brain, a cell is either a neuron (expresses `RBFOX3`, not `GFAP`) or an astrocyte (expresses `GFAP`, not `RBFOX3`). A cell expressing both or neither is not in the brain (and therefore can't arise in the true label since we restrict to brain cells only). 

**Method:** I binarized the expression of `RBFOX3` and `GFAP` to create a ground-truth XOR label. I then masked these two genes and tasked the model with predicting their state, from which I calculated a predicted XOR probability.

**Prediction:** Since `RBFOX3` and `GFAP` are masked, the model must infer its expression state conditioned on the background expression of all other genes. A successful model would learn to identify a non-linear mapping between neuron, astrocyte, and other cell types, and the XOR activation pattern of the two genes.  

```python
# Core logic of eval_xor.py

# 1. Sample real expression data
expr = sample_real_data(n_cells)

# 2. Binarize expression to create ground truth XOR label
x_bin = binarize(expr[:, idx_rbfox3], q=0.6) # Neuron marker
y_bin = binarize(expr[:, idx_gfap], q=0.6)   # Astrocyte marker
y_true = np.bitwise_xor(x_bin, y_bin)

# 3. Mask the two genes and tokenize
tokens = tokenize(expr)
tokens[:, [idx_rbfox3, idx_gfap]] = MASK_TOKEN

# 4. Run model to get probabilities for the "high expression" token
logits = model(tokens)
probs = torch.softmax(logits, dim=-1)
prob_rbfox3_on = probs[:, idx_rbfox3, HIGH_TOKEN]
prob_gfap_on = probs[:, idx_gfap, HIGH_TOKEN]

# 5. Calculate XOR probability from model predictions
p_xor = prob_rbfox3_on * (1 - prob_gfap_on) + prob_gfap_on * (1 - prob_rbfox3_on)
y_pred = (p_xor >= 0.5)

# 6. Evaluate using balanced accuracy and AUROC
bal_acc = balanced_accuracy_score(y_true, y_pred)
auroc = roc_auc_score(y_true, p_xor)
```

**Results:** The model failed this task completely. By always predicting the majority class (XOR = 0), it achieved a balanced accuracy of only 0.5, equivalent to random guessing. This indicates that while the model can capture linear correlations, it struggles with more complex, non-linear biological rules like mutual exclusivity.

* **Balanced accuracy:** 0.500
* **AUROC:** 0.500
* **Confusion Matrix:**

|                    | Predict Value = 0     | Predict Value = 1  |
| :----------------- | :-------------------- | :----------------- |
| **Real Value = 0** | 28458 (True Negative) | 0 (False Positive) |
| **Real Value = 1** | 1542 (False Negative) | 0 (True Positive)  |

---

## Scaling Laws for Biological Transfer Learning

One of the most exciting frontiers is transfer learning: between organisms, across cell types, or from lab-grown cells to *in-vivo* samples. Can pre-training on petabytes of yeast or fly data help understand human biology? [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) gives empirically fittable power laws that quantify how valuable transfer learning is.

The effective data transferred (\$D\_t\$) from a pre-training source to a fine-tuning target can be modeled as a function of the fine-tuning data size (\$D\_f\$) and the model/compute size (\$N\$):

\$D\_t = k \cdot (D\_f)^\alpha \cdot N^\beta\$

Here, \$\alpha\$ measures the similarity between the source and target data, and \$\beta\$ represents how effectively more compute translates to better performance. This formulation allows trading data for compute. For example, to find the equivalent data multiplier (\$\text{data}\_c\$) gained from a certain model size multiplier (\$\text{model}\_c\$), set the contributions to be equal:

\$(\text{data}\_c \cdot D\_f)^\alpha \cdot N^\beta = (D\_f)^\alpha \cdot (\text{model}\_c \cdot N)^\beta\$

\$\Rightarrow \text{data}\_c^\alpha = \text{model}\_c^\beta \Rightarrow \text{data}\_c = \text{model}\_c^{\beta/\alpha}\$

Suppose I have abundant data from lab-grown cells but only a small, precious dataset of cells sampled directly from a human body. The "ecological validity" of the lab data is low, reflected by a transfer similarity of, say, \$\alpha = 0.38\$. If I need the performance equivalent of a **1,000,000x** larger ecologically valid dataset (\$\text{data}\_c = 10^6\$), how much larger does my pre-trained model need to be?

Assuming \$\beta\$ is relatively constant from similar empirical studies (making \$\beta/\alpha \approx 1/\alpha\$), I can solve for the model multiplier \$\text{model}\_c\$:

\$\text{model}\_c = (\text{data}\_c)^\alpha = (10^6)^{0.38} = 10^{2.28} \approx 191\$

This is a powerful result: by pre-training a **\~200x** larger model on abundant (but less relevant) data and then fine-tuning it on the ecologically valid data, I might achieve the same performance as if I had collected a million times more ecologically valid data.

---

## Interpretable Biological Simulators

Deep learning models trained on time-series data from dynamical systems, like the Lorenz attractor, have been shown to learn the underlying governing equations in their weights. By enforcing sparsity during training, for example, with the Sparse Identification of Nonlineaer Dynamics algorithm (SINDy), it's possible to recover the exact polynomial terms of the original ordinary differential equations (ODEs):

> In SINDY, they initalize a dictionary \$\Theta(x)\$ whose columns are candidate nonlinear basis functions \$(1, x, y, z, x^2, xy, \dots)\$. They then solve the regression \$\dot{x} = \Theta(x) \Xi\$ with an \$\ell\_1\$ penalty on \$\Xi\$. The sparsity constraint drives most columns of \$\Xi\$ to exactly zero... For simulated Lorenz data... the algorithm keeps just six columns—corresponding to the terms \$(y-x)\$, \$x\$, \$xz\$, \$xy\$, and \$z\$—and the non-zero coefficients reproduce the canonical ODEs:
>
> \$\dot{x} = 10(y-x)\$
> \$\dot{y} = 28x - xz - y\$
> \$\dot{z} = xy - \frac{8}{3}z\$

Single-cell foundation models are likely doing something analogous in a much higher-dimensional, non-polynomial function space. The training process, guided by the loss function and a strong optimizer, prunes away parameter configurations that don't improve the conditional likelihood of real transcriptomes. What remains is an entangled, superpositioned library of "active columns" that encode the true, non-linear governing equations of the cell dynamic.

There's a lot of alpha in developing methods, whether through mechanistic interpretability or SINDy-style sparsity constraints, to disentangle these learned features and extract the generalizable "governing equations" of the cell.
