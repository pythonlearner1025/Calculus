# Single-Cell Foundation Models: Learning the Orchestra of the Cell

Single-cell transcriptomics gives us a snapshot of the \~20,000-gene expression state of a cell. While the number of possible transcriptomes is astronomically high (in a binary on/off model, it's $`2^{20,000}\`$), the space of *biologically plausible* transcriptomes is a tiny, intricately structured manifold. This manifold is shaped by the physical and gene-regulatory laws governing the cell.

Linear methods like Principal Component Analysis (PCA) are great for finding the major axes of variation in this data, but they fall short because biology is fundamentally non-linear. To model complex phenomena like gene XOR gates or the effects of perturbations, we need models that can learn the true, non-linear probability distribution `$p(\text{transcriptome})$` from which all observed cell states are sampled.

Enter foundation models. Architectures like transformers, with their non-linear activation functions, can learn the complex, high-order dependencies between genes. Diffusion models are a prime example, as their training objective is to learn the score function, `$\nabla_x \log p(x)$`, which is the gradient of the log-probability of the data. This allows them to capture the underlying structure of the biological manifold in detail.

-----

## Why Foundation Models Outperform PCA

| Capability | Why PCA Fails | How Foundation Models Succeed |
| :--- | :--- | :--- |
| **Non-linear Dependencies** | PCA is a linear method, assuming additive or independent effects. It can't capture XOR-type interactions where a gene's effect depends on the absence of another. | A diffusion model learns the score of the true data distribution, capturing arbitrary, non-linear dependencies. |
| **Conditional Generation** | PCA has no native mechanism for conditional generation, like predicting the effect of a drug perturbation. | Models can be explicitly trained to learn the conditional distribution `$p(\text{masked\_gene} | \text{visible\_genes, perturbation})$`. |
| **Long-range Dependencies** | While its latent components (PCs) are global, they are dense linear combinations of all genes. | The attention mechanism allows the model to learn sparse, context-dependent interactions between any two genes, no matter how distant in the genome. |

-----

## The Data: A Petabyte-Scale Glimpse into the Cell

The scale of available data is staggering. The Sequence Read Archive (SRA) holds over 47 petabytes of raw sequencing data, and its single-cell RNA-seq (scRNA-seq) component is growing exponentially. A key resource is **scBaseCount**, a dataset from the Arc Institute containing nearly 300 million transcriptomes curated from the SRA.

This dataset primarily covers experiments using 10x Genomics technology, which are highly valuable for transcriptomics. Based on a starting point of 330 million 10x-labeled cells in early 2025 and a conservative 20% year-on-year growth, we project the public archive will hit **one billion cells by mid-2031**. This timeline could accelerate to 2029-2030 as more data types are processed and sequencing throughput increases.

The data is heavily skewed towards human and mouse cells, mostly from lab-grown embryonic or differentiated cell lines. This creates a significant distribution shift from *in vivo* cells, which are influenced by complex cell-cell signaling and homeostatic rhythms not present in a culture dish.

#### Total Cells per Organism in scBaseCount (Top 10)

| Organism | Total Cells (Millions) |
| :--- | :--- |
| *Homo sapiens* | 118.7 |
| *Mus musculus* | 98.4 |
| *Macaca mulatta* | 4.1 |
| *Danio rerio* | 3.7 |
| *Sus scrofa* | 1.8 |
| *Drosophila melanogaster* | 1.7 |
| *Arabidopsis thaliana* | 1.6 |
| *Gallus gallus* | 0.7 |
| *Heterocephalus glaber* | 0.5 |
| *Bos taurus* | 0.4 |

-----

## Private Evaluation: Putting the Model to the Test

We ran two private evaluations on a diffusion-transformer model trained on human brain scRNA-seq data to test its ability to capture known biological relationships.

### 1\. Linear Covariance Evaluation

**Goal:** Does the model understand that synaptic (SYN) and mitochondrial (MT) gene groups have distinct correlation patterns? In healthy neurons, synaptic genes are co-expressed, while mitochondrial gene expression can indicate stress.

**Method:** We calculated the average pairwise Spearman correlation (`$\bar{\rho}$`) within SYN genes, within MT genes, and between the two groups. We compared the *sign* of these correlations from real data to the model's imputed values for the same cells.

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

| Aspect | Real Data (`$\bar{\rho}$`) | Model (`$\bar{\rho}$`) | Sign Match |
| :--- | :---: | :---: | :---: |
| Synaptic Genes Together | +0.30 | +0.63 | Match |
| Mitochondrial Genes Together | +0.00 | +0.56 | Mismatch |
| Synaptic vs. Mitochondrial | +0.26 | +0.62 | Match |
| **Final Sign-Match Score** | | | **0.67** |

### 2\. Non-Linear XOR Evaluation

**Goal:** Can the model learn a non-linear XOR relationship? In the brain, a cell is either a neuron (expresses `RBFOX3`, not `GFAP`) or an astrocyte (expresses `GFAP`, not `RBFOX3`). A cell expressing both or neither is biologically implausible.

**Method:** We binarized the expression of `RBFOX3` and `GFAP` to create a ground-truth XOR label. We then masked these two genes and tasked the model with predicting their state, from which we calculated a predicted XOR probability.

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

**Results:** The model failed this task completely. By predicting the "off" state for all cells, it ignored the XOR relationship entirely. The high number of true negatives makes metrics like raw accuracy misleading; balanced accuracy reveals the failure. The model was unable to learn this complex, non-linear biological rule from the data.

  * **Balanced accuracy:** 0.500
  * **AUROC:** 0.500
  * **Confusion Matrix:**
    | | Predict Value = 0 | Predict Value = 1 |
    | :--- | :--- | :--- |
    | **Real Value = 0** | 28458 (True Negative) | 0 (False Positive) |
    | **Real Value = 1** | 1542 (False Negative) | 0 (True Positive) |

-----

## Scaling Laws for Biological Transfer Learning

One of the most exciting frontiers is transfer learning: between organisms, across cell types, or from lab-grown cells to *in-vivo* samples. Can pre-training on petabytes of yeast or fly data help us understand human biology? Scaling laws give us a framework to reason about this.

The effective data transferred (`$D_t$`) from a pre-training source to a fine-tuning target can be modeled as a function of the fine-tuning data size (`$D_f$`) and the model/compute size (`$N`$):

$$D_t = k \cdot (D_f)^\alpha \cdot N^\beta$$

Here, `$\alpha$` measures the similarity between the source and target data, and `$\beta$` represents how effectively more compute translates to better performance. This formulation allows us to trade data for compute. For example, to find the equivalent data multiplier (`$data_c$`) gained from a certain model size multiplier (`$model_c$`), we can set the contributions to be equal:

$$(data_c \cdot D_f)^\alpha \cdot N^\beta = (D_f)^\alpha \cdot (model_c \cdot N)^\beta$$
$$data_c^\alpha = model_c^\beta \implies data_c = model_c^{\beta/\alpha}$$

Let's imagine a practical scenario. Suppose we have abundant data from lab-grown cells but only a small, precious dataset of cells sampled directly from a human body. The "ecological validity" of the lab data is low, reflected by a transfer similarity of, say, `$\alpha = 0.38$`. If we need the performance equivalent of a **1,000,000x** larger ecologically valid dataset (`$data_c = 10^6$`), how much larger does our pre-trained model need to be?

Assuming `$\beta$` is relatively constant from similar empirical studies (making `$\beta/\alpha \approx 1/\alpha$`), we can solve for the model multiplier `$model_c$`:

$$model_c = (data_c)^\alpha = (10^6)^{0.38} = 10^{2.28} \approx 191$$

This is a powerful result: by pre-training a **\~200x** larger model on abundant (but less relevant) data, we might achieve the same performance as if we had collected a million times more rare (but highly relevant) data.

-----

## Towards Interpretable Biological Simulators

Deep learning models trained on time-series data from dynamical systems, like the Lorenz attractor, have been shown to learn the underlying governing equations in their weights. By enforcing sparsity during training (e.g., with the SINDy algorithm), it's possible to recover the exact polynomial terms of the original ordinary differential equations (ODEs):

> We build a dictionary `$\Theta(x)$` whose columns are candidate nonlinear basis functions `$(1, x, y, z, x^2, xy, \dots)$`. We then solve the regression `$\dot{x} = \Theta(x) \Xi$` with an `$\ell_1$` penalty on `$\Xi$`. The sparsity constraint drives most columns of `$\Xi$` to exactly zero... For simulated Lorenz data... the algorithm keeps just six columns—corresponding to the terms `$(y-x)$`, `$x$`, `$xz$`, `$xy$` and `$z$`—and the non-zero coefficients reproduce the canonical ODEs:
>
> $$\dot{x} = 10(y-x)$$\>$$\dot{y} = 28x - xz - y$$\>$$\dot{z} = xy - \frac{8}{3}z$$

Single-cell foundation models are likely doing something analogous in a much higher-dimensional, non-polynomial function space. The training process, guided by the loss function and a strong optimizer, prunes away parameter configurations that don't improve the conditional likelihood of real transcriptomes. What remains is an entangled, superpositioned library of "active columns" that encode the true, non-linear gene regulatory network.

The next great challenge is to develop methods—whether through mechanistic interpretability or SINDy-style sparsity constraints—to disentangle these learned features and extract the generalizable "governing equations" of the cell.
