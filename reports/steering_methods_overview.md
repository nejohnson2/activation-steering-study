# Three Approaches to Activation Steering: CAA, PCA, and Linear Probes

**A Primer for Researchers New to Mechanistic Interpretability**

---

## 1. The Core Idea: Concepts as Directions

Before diving into the three methods, we need one foundational insight. Recent work in
mechanistic interpretability has converged on the **Linear Representation Hypothesis**: the
idea that high-level, human-interpretable concepts — politeness, truthfulness, emotional
tone — are encoded as approximately linear directions in a neural network's activation space
(Park et al., 2023; Nanda et al., 2023). If "formality" corresponds to a direction in
activation space, then we can *read* formality by projecting activations onto that direction,
and *write* formality by adding a scaled version of that direction to the activations during
inference. This read/write duality is the engine behind all activation steering methods.

The practical question, then, is: **how do we find the right direction?** The three methods
in this study — Contrastive Activation Addition (CAA), PCA-based extraction, and Linear
Probe extraction — are three different answers to that question. They share the same
two-stage framework:

1. **Extract** a steering vector from contrastive activation data.
2. **Inject** that vector into the model's forward pass via a hook.

They differ only in Step 1 — the mathematics of how the direction is identified.

> **[Diagram note]** A useful figure here would show a simplified 2D activation space with
> two clusters (positive and negative persona activations) and three overlaid arrows
> representing the CAA mean-difference vector, the PCA first principal component, and the
> linear probe decision boundary normal. This would visually convey how the three methods
> can produce different directions from the same data.

---

## 2. Contrastive Activation Addition (CAA)

### Origin

CAA was introduced by Turner et al. (2023) under the name *Activation Addition* (ActAdd)
and extended to a dataset-averaged form by Panickssery et al. (2024) in their work on
steering Llama 2. The core idea is disarmingly simple: if you want a vector that captures
"formality," run the model on formal prompts and informal prompts, and subtract.

### How It Works

Given a set of *N* contrastive pairs — where each pair shares the same user prompt but
differs in the system prompt (one encouraging the target persona, one discouraging it) — we
collect the residual stream activations at a chosen layer *L* for each prompt. The steering
vector is the mean difference:

$$
\mathbf{v}_{\text{CAA}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \mathbf{a}_L(\mathbf{x}_i^+) - \mathbf{a}_L(\mathbf{x}_i^-) \right]
$$

That is it. No optimization, no learned parameters, no hyperparameters beyond the choice
of layer and contrastive dataset. The averaging across many pairs is what distinguishes CAA
from the original single-pair ActAdd: it cancels out prompt-specific noise and isolates
the behavioral direction.

### Connection to This Project

In our codebase, `src/extraction/caa.py` implements this directly. For each layer, it
computes:

```python
pos_mean = pos_activations[layer_idx].float().mean(dim=0)
neg_mean = neg_activations[layer_idx].float().mean(dim=0)
vectors[layer_idx] = pos_mean - neg_mean
```

The contrastive pairs come from `src/data/personas.py`, which generates 100 pairs per
persona — the same user prompt wrapped with opposing system prompts (e.g., "scholarly
language with structured argumentation" vs. "casual, informal... slang, contractions").

### Why It Works

Im & Li (2025) proved that the mean difference vector is the *optimal* solution to the
contrastive steering objective. Specifically, it minimizes the expected squared error
between the steering vector and the individual contrastive differences:

$$
\mathbf{v}_{\text{CAA}} = \arg\min_{\mathbf{v}} \; \mathbb{E} \left\| \mathbf{a}^+ - \mathbf{a}^- - \mathbf{v} \right\|^2
$$

This is a reassuring result: the simplest possible method is also the theoretically optimal
one for this objective.

### Strengths and Limitations

**Strengths:** Computationally trivial (no training loop), theoretically principled, and
empirically strong — it consistently outperforms the other two methods in comparative
evaluations (Im & Li, 2025). It also composes well with other techniques like system
prompts and fine-tuning.

**Limitations:** The vector's effectiveness depends on layer choice (in our experiments, we
sweep layers 0, 4, 8, ..., 31). It also requires well-constructed contrastive pairs — if
the positive and negative prompts differ in ways beyond the target concept, the vector will
capture those confounds too. Effect sizes can be modest when steering only affects the final
generated tokens (as in multiple-choice evaluations).

---

## 3. PCA-Based Extraction

### Origin

PCA-based steering was introduced as part of the *Representation Engineering* (RepE)
framework by Zou et al. (2023). Where CAA takes the mean of contrastive differences, RepE
asks: what is the *principal direction of variation* in those differences?

### How It Works

The procedure begins identically to CAA — collect activations from contrastive pairs. But
instead of averaging the differences, we apply Principal Component Analysis:

1. Concatenate positive and negative activations into a single matrix.
2. Fit PCA and extract the first principal component (PC1).
3. Orient PC1 so it points from the negative centroid toward the positive centroid.

The steering vector is this oriented first principal component.

### Connection to This Project

In `src/extraction/pca.py`, the implementation follows this procedure precisely:

```python
combined = torch.cat([pos, neg], dim=0).numpy()
pca = PCA(n_components=1)
pca.fit(combined)
pc1 = torch.from_numpy(pca.components_[0])

# Orient to point from negative to positive
diff = pos.mean(dim=0) - neg.mean(dim=0)
if torch.dot(pc1, diff) < 0:
    pc1 = -pc1
```

Note the orientation step: PCA components are sign-ambiguous (both $\mathbf{v}$ and
$-\mathbf{v}$ explain the same variance), so we explicitly align the vector with the
positive-minus-negative direction.

### The Subtle Problem

The intuition behind PCA steering sounds reasonable: find the direction of maximum variance
in the contrastive data, and use that as the steering direction. But there is a subtle and
important flaw. PCA finds the direction of maximum *total* variance — not the direction
that best *separates* the two classes.

> **[Diagram note]** A figure contrasting the PCA direction with the mean-difference
> direction would be valuable here. Imagine two elongated Gaussian clusters that are offset
> from each other. PCA captures the direction of elongation (within-cluster variance), while
> the mean difference captures the direction between cluster centers (between-cluster
> separation). These can be nearly orthogonal.

Im & Li (2025) demonstrated this empirically: the PCA direction is often nearly orthogonal
to the actual separation direction between positive and negative activations. On a refusal
steering benchmark, CAA achieved 87.5% accuracy compared to PCA's 64%. The direction of
maximum variance within the data is dominated by linguistic variation (how prompts differ
in wording, topic, and length), not by the behavioral concept we care about.

### Strengths and Limitations

**Strengths:** PCA is a well-understood dimensionality reduction technique. One interesting
property is that PCA-derived vectors have been shown to evade detection by models trained to
identify steering (Steering Awareness, 2025) — precisely because they are orthogonal to the
expected behavioral direction. PCA also provides a natural way to examine multiple
components, not just the first.

**Limitations:** The fundamental variance-vs-separation misalignment means PCA is often a
suboptimal choice for steering. It optimizes for the wrong objective — explaining variance
rather than separating behavioral classes.

---

## 4. Linear Probe Extraction

### Origin

The use of linear probes for steering traces to Li et al. (2023) and their *Inference-Time
Intervention* (ITI) method, which used trained classifiers to identify truthfulness
directions in Llama's attention heads. The approach draws on a longer tradition of *probing
classifiers* in NLP, where linear models are trained on frozen representations to test what
information those representations encode (Alain & Bengio, 2017; Belinkov, 2022).

### How It Works

Unlike CAA and PCA, the linear probe method is *supervised*. We train a binary linear
classifier to distinguish positive from negative activations:

1. Label positive activations as class 1, negative activations as class 0.
2. Train a linear model (a single `nn.Linear` layer) with binary cross-entropy loss.
3. After training, extract the learned weight vector $\mathbf{w}$.
4. Normalize $\mathbf{w}$ to unit norm. This normalized weight vector is the steering direction.

The intuition is elegant: the classifier learns to find the direction in activation space
that best separates the two behavioral classes. Its weight vector is the normal to the
learned decision boundary — precisely the direction along which "positive" and "negative"
are most distinguishable.

### Connection to This Project

In `src/extraction/linear_probe.py`, the probe is a single linear layer trained for 20
epochs with Adam and binary cross-entropy loss:

```python
probe = nn.Linear(hidden_size, 1).to(device)
optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# After training:
weight = probe.weight.data.squeeze(0).detach().cpu()
weight = weight / weight.norm()  # Normalize to unit norm
```

The normalization step is important: it ensures that the steering multiplier (which ranges
from 0.5 to 4.0 in our experiments) controls the magnitude of the intervention in a
consistent way, independent of the classifier's learned scale.

### How It Differs from the Others

The linear probe is the only method that uses gradient-based optimization, which gives it
both its power and its risks. Because it explicitly optimizes for class separation, it
avoids PCA's variance-vs-separation misalignment. But unlike CAA, the probe's objective
(classification accuracy) is not identical to the steering objective (behavioral shift).
A classifier that achieves 95% accuracy may have learned a direction that is excellent for
*detecting* the persona but suboptimal for *inducing* it.

> **[Diagram note]** A schematic of the linear probe training pipeline would help here:
> contrastive activations (labeled +/-) flowing into a single linear layer, trained with
> BCE loss, then the weight vector being extracted and normalized for use as a steering
> direction. This makes concrete how a classification tool becomes a steering tool.

### Strengths and Limitations

**Strengths:** Directly optimizes for class separation (unlike PCA). Can achieve high
classification accuracy, indicating it finds a genuinely discriminative direction. The
training also provides a built-in diagnostic: probe accuracy tells you how linearly
separable the concept is at each layer, which helps identify the best layers for steering.

**Limitations:** Requires a training loop (though a fast one — 20 epochs on ~200 examples).
Susceptible to overfitting on small datasets or capturing dataset-specific artifacts rather
than the general concept. The magnitude of the weight vector is calibrated for classification,
not for steering, which is why normalization is necessary. It is also the least transparent
of the three methods — the learned direction is harder to interpret than a simple mean
difference.

---

## 5. How the Steering Vector Is Applied

Regardless of how the vector is extracted, the injection mechanism is the same across all
three methods. In our project, `src/steering/injector.py` registers a PyTorch forward hook
on the target transformer layer. During each forward pass, the hook adds the scaled steering
vector to the residual stream:

$$
\mathbf{h}'_l = \mathbf{h}_l + \alpha \cdot \mathbf{v}
$$

where $\mathbf{h}_l$ is the original hidden state at layer $l$, $\mathbf{v}$ is the
steering vector, and $\alpha$ is the multiplier controlling steering strength. The vector
is broadcast across all sequence positions and batch elements:

```python
def hook_fn(module, input, output):
    hidden = output[0]  # (batch, seq_len, hidden_size)
    hidden = hidden + steering_vec.unsqueeze(0).unsqueeze(0)
    return (hidden,) + output[1:]
```

Our experiments sweep multipliers from 0.5 to 4.0 across layers sampled every 4th layer,
for each of the three extraction methods, across five personas and three model architectures
(Llama 3.1 8B, Gemma 2 9B, Qwen 2.5 7B). This factorial design allows us to directly
compare how the extraction method interacts with layer choice and steering magnitude.

---

## 6. Summary: Three Methods, One Framework

| Property | CAA | PCA | Linear Probe |
|---|---|---|---|
| **Objective** | Mean contrastive difference | Maximum variance direction | Maximum class separation |
| **Optimization** | None (closed-form) | None (eigendecomposition) | Gradient descent (20 epochs) |
| **Supervision** | Unsupervised (paired) | Unsupervised (paired) | Supervised (labeled) |
| **Theoretical guarantee** | Optimal for contrastive MSE | Optimal for variance explained | Optimal for classification loss |
| **Key risk** | Confounds in contrastive pairs | Variance ≠ separation | Overfitting; scale mismatch |
| **Codebase location** | `src/extraction/caa.py` | `src/extraction/pca.py` | `src/extraction/linear_probe.py` |

The methods represent a spectrum of assumptions. CAA assumes the concept direction *is* the
mean difference — the simplest and most direct approach. PCA assumes it is the direction
of greatest variation — a reasonable but potentially misleading assumption. The linear probe
assumes it is the direction that best classifies — powerful but requiring care in
training. Our project evaluates all three under identical conditions, using three tiers of
evaluation (representation-level metrics, a persona classifier, and an LLM-as-judge) to
determine which approach most reliably steers persona-level behavior across architectures.

---

## References

Alain, G. & Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier
Probes. *ICLR Workshop*. arXiv:1610.01644.

Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances.
*Computational Linguistics*, 48(1), 207–219.

Im, S. & Li, S. (2025). A Unified Understanding and Evaluation of Steering Methods.
arXiv:2502.02716.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-Time
Intervention: Eliciting Truthful Answers from a Language Model. *NeurIPS 2023 (Spotlight)*.
arXiv:2306.03341.

Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent Linear Representations in World
Models of Self-Supervised Sequence Models. *BlackboxNLP Workshop 2023*.

Panickssery, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A.M. (2024).
Steering Llama 2 via Contrastive Activation Addition. *Proceedings of the 62nd Annual
Meeting of the ACL*. arXiv:2312.06681.

Park, K. et al. (2023). The Linear Representation Hypothesis and the Geometry of Large
Language Models. arXiv:2311.03658. *ICML 2024*.

Turner, A.M., Thiergart, L., Leech, G., Udell, D., Vazquez, J.J., Mini, U., &
MacDiarmid, M. (2023). Activation Addition: Steering Language Models Without Optimization.
arXiv:2308.10248.

Wehner, J. et al. (2025). Representation Engineering for Large-Language Models: Survey and
Research Challenges. arXiv:2502.17601.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X.,
Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M.J., Wang, Z., Mallen, A.,
Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J.Z., & Hendrycks, D. (2023).
Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
