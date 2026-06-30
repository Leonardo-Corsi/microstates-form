# MEED

## EEG Microstates

EEG microstate analysis converts a continuous sequence of scalp voltage maps into a small set of recurring topographies and then studies the temporal organization of those states. This framework is useful because it gives compact, reproducible, and interpretable descriptors: map classes, duration, occurrence, coverage, transition structure, and global explained variance. However, the standard pipeline also compresses several distinct phenomena into a single hard label.

The central methodological problem is that we ideally want each sample to be strong in amplitude, stable across time and close to a template but there are ambiguous maps sitting between templates, or poorly described by the assumed map family. These are not equivalent properties. Global field power, topographic dissimilarity, template correlation, and label confidence are related but separable. If they are ignored, later metrics may inherit errors that arise much earlier in the pipeline.

Microstate evidence should be separated into levels, starting from the credibility of the maps themselves, then moving to sample-wise labeling, then to zero-order temporal summaries, first-order transition structure, and finally higher-order sequence metrics. MEED is introduced at this point as a complementary representation: not as source localization, but as an interpretable field-geometric embedding of instantaneous scalp maps helping to define uncertainty and "microstateishness" of a map.

## Metrics

A subject-level microstate result can be understood as a layered object rather than as a single label sequence. Each layer answers a different question.

#### Template level
Template-level analysis asks whether the extracted maps are credible. This is the level of map quality, segmentation quality, and agreement with a normative or canonical reference. Relevant quantities include 
 - GEV (global explained variance), microstate-wise and total
 - silhouette score of a solution
 - alignment with normative templates and microstate collapse

#### Zero order 
Zero-order analysis describes how much each state is used. 
 - Duration measures how long a state persists once entered. 
 - Occurrence measures how often a state appears. 
 - Coverage measures proportion of samples labelled by each microstate 

 Coverage combines how often a state appears with how long it lasts and is therefore useful descriptively, but it should not be treated as an independent mechanism.

#### First order
First-order analysis concerns transitions between states. The transition matrix can be computed on the raw sequence or considering same-label sequences as a single state. The second case means to zero-out the diagonal of the matrix and re-normalizing rows to unit sum. This choice affects whether microstate duration can confound the metrics. Raw transition probabilities are also constrained by state occurrence though: a common state has more opportunities to be entered or exited than a rare state. For this reason, residual, adjusted, or prevalence-corrected transition metrics are more interpretable than raw probabilities alone. There can also be metrics summarizing the entire matrix such as memory depth or entropy production rate, which take root from literature on Markovian processes. 

#### Higher order
Higher-order analysis asks whether the sequence contains structure beyond first-order Markov dynamics. This includes symbolic memory, repeated motifs, entropy rate, higher-order Markov models, sequence complexity, and deviation from a first-order null. These metrics should only be interpreted after verifying that lower-level problems have been controlled. 

## Map comparison with normative templates

Before interpreting any temporal metric, the first question is whether the extracted maps are credible. Any sequence can only be interpreted as an A/B/C/D-like sequence if the subject maps are sufficiently close to the normative maps and if the internal geometry among maps is preserved; otherwise we may think for example that D has a very low occurrence and duration but it is just because the subject does not exhibit the D microstate. It is the same error as estimating an organism has non-detectable levels of a protein because of a metabolic failure, when in reality the gene that should express it is suppressed. Another possible problem is that metrics of two correlated microstates, such as C and D, may change just because uncertain maps are assigned to one of them by chance.

Let the canonical meta-microstate maps be collected in a matrix $M$, and let the subject-level maps be collected in a matrix $X$. Both matrices have $N$ rows, corresponding to channels, and $J$ columns, corresponding to microstate maps:

$$
M = [m_1, m_2, \ldots, m_J] \in \mathbb{R}^{N \times J},
$$

$$
X = [x_1, x_2, \ldots, x_J] \in \mathbb{R}^{N \times J}.
$$

We assume centered and L2-normalized scalp maps as columns:

$$
\mathbf{1}^\top m_j = 0, \quad \mathbf{1}^\top x_j = 0,
$$

$$
||m_j||_2 = 1, \quad ||x_j||_2 = 1.
$$

Because cluster order is arbitrary, maps must be aligned before class-wise interpretation. Given an aligned version of the subject maps, denoted by $X^\star$, we define the cross-correlation matrix

$$
G^\star = M^\top X^\star.
$$

Since microstate maps are polarity-invariant, we use the squared correlation matrix

$$
S^\star = G^\star \odot G^\star.
$$

Each entry $S^\star_{ij}$ is therefore an $R^2$-like topographic similarity between canonical map $m_i$ and aligned subject map $x^\star_j$.

Canonical microstate maps are not orthogonal. Therefore, a distance from canonical maps should not only test whether $S^\star$ is diagonal. Instead, it should compare $S^\star$ with the expected self-similarity structure of the canonical maps:

$$
G_M = M^\top M,
$$

$$
S_M = G_M \odot G_M.
$$

The matching distortion is then defined as

$$
D_{\mathrm{match}} =
\frac{||S^\star - S_M||_F}{||S_M||_F}.
$$

A corresponding quality index can be written as

$$
Q_{\mathrm{match}} = 1 - D_{\mathrm{match}}.
$$

Since a single poorly matched class can contaminate downstream class-specific metrics, the diagonal values of $S^\star$ should also be reported:

$$
R^2_j = S^\star_{jj},
$$

together with the minimum matched class:

$$
R^2_{\min} = \min_j S^\star_{jj}.
$$

This tests whether the subject maps preserve the relational geometry of the canonical maps, not only whether each subject map has a high diagonal match.

This distinction is important because microstate maps form a non-orthogonal frame in sensor space. Two map sets can share a similar span while differing in their internal geometry. Conversely, two maps can be individually similar to canonical maps but still distort the geometry among classes. This is why map-level evaluation should include both atom-wise similarity and representation-level distortion.

The practical report for map comparison should include the aligned labels, diagonal $R^2$ values, the minimum diagonal match, assignment margins, permutation-invariant canonical coverage or collapse scores, novelty scores, collision indicators, and a geometry-distortion score. If these diagnostics fail, later temporal metrics should be treated as exploratory rather than as class-specific physiological evidence.

## Alignment of maps

Before comparing maps, subject-level maps must be aligned to the canonical templates. Greedy strategies may lead to suboptimal solutions, because the best local match for one map can prevent the globally best one-to-one assignment. If we define a distortion metric such as $D_{\mathrm{match}}$, it is natural to define the optimal alignment as the permutation of $X$ that minimizes that distortion.

Let $P$ be a $J \times J$ permutation matrix. We define the aligned subject maps as

$$
X^\star = X P^\star,
$$

where

$$
P^\star =
\arg\min_{P \in \mathcal{P}_J}
\frac{||(M^\top X P \odot M^\top X P) - S_M||_F}{||S_M||_F}.
$$

Equivalently, if we first compute the unaligned squared-similarity matrix

$$
S = (M^\top X) \odot (M^\top X),
$$

then the optimal permutation can be written as

$$
P^\star =
\arg\min_{P \in \mathcal{P}_J}
||S P - S_M||_F.
$$

This formulation makes the alignment permutation-invariant with respect to the arbitrary ordering of the subject-level clusters. It also avoids treating off-diagonal values as errors when they are already expected from the non-orthogonality of the canonical templates.

The optimization does not require explicit enumeration of all $J!$ permutations. Because the Frobenius objective decomposes across columns,

$$
||S P - S_M||_F^2 =
\sum_{j=1}^J ||S_{:, \pi(j)} - S_{M,:,j}||_2^2,
$$

the problem can be solved as a linear assignment problem. Define the cost matrix

$$
A_{ij} =
||S_{:,i} - S_{M,:,j}||_2^2.
$$

Here $A_{ij}$ is the cost of assigning subject map $x_i$ to canonical map $m_j$, after accounting for the full similarity profile of that map rather than only its direct correlation with $m_j$. The optimal assignment is

$$
\pi^\star =
\arg\min_{\pi}
\sum_{j=1}^J A_{\pi(j),j}.
$$

This is solved by the Hungarian algorithm. The resulting permutation defines $P^\star$, and therefore the aligned map matrix

$$
X^\star = X P^\star.
$$

After alignment, the final similarity matrix is

$$
G^\star = M^\top X^\star,
$$

and the final squared-similarity matrix is

$$
S^\star = G^\star \odot G^\star.
$$

The diagonal of $S^\star$ gives the class-wise atom similarity, while the full matrix $S^\star$ can be compared with $S_M$ to quantify preservation of the canonical similarity geometry.

### Assignment margin

The best assignment alone is not sufficient, because a map can be assigned to a canonical class even when the assignment is ambiguous. Therefore, for each subject map $x_i$, we can define an assignment margin using the cost matrix $A$.

Let $j_1(i)$ be the assigned canonical class for subject map $x_i$, and let $j_2(i)$ be the best alternative canonical class:

$$
j_1(i) = \pi^\star(i),
$$

$$
j_2(i) =
\arg\min_{j \neq j_1(i)} A_{ij}.
$$

The assignment margin can be written as

$$
\Delta_i = A_{i j_2(i)} - A_{i j_1(i)}.
$$

A small $\Delta_i$ indicates that the map has no clear canonical assignment. This can happen when two canonical maps are intrinsically similar, when the subject map lies between two canonical classes, or when the extracted map is noisy or unstable. Therefore, assignment margins should be interpreted together with both the diagonal $R^2$ values and the global geometry distortion.

A normalized version can also be used:

$$
\Delta_i^{\mathrm{rel}} =
\frac{A_{i j_2(i)} - A_{i j_1(i)}}{A_{i j_2(i)} + \epsilon}.
$$

This makes margins more comparable across subjects, with $\epsilon$ only included to avoid division by zero.

### Assigned-label failure, canonical collapse, and novelty

The alignment step should not erase the distinction between an assigned class failing and a canonical class being absent. After alignment, an assigned-class failure can be summarized by

$$
c_j^{\mathrm{assigned}} = 1 - S^\star_{jj}.
$$

This is meaningful only when the alignment itself is credible, namely when the diagonal match is high, the assignment margin is not small, and the global geometry distortion is acceptable. If these conditions fail, the safer question is whether canonical class $j$ is represented anywhere among the subject maps before forcing a one-to-one label. Using the unaligned squared-similarity matrix $S = (M^\top X) \odot (M^\top X)$, define canonical coverage as

$$
\mathrm{coverage}_j = \max_i S_{ji},
$$

and the corresponding permutation-invariant collapse score as

$$
c_j^{\mathrm{PI}} = 1 - \max_i S_{ji}.
$$

For example, assigned-D failure is $1-S^\star_{DD}$, whereas D collapse is $1-\max_i S_{Di}$. These are not equivalent. The first says that the map assigned to D is a poor D-like map. The second says that no subject map is a good representative of canonical D.

The dual quantity is observed-map novelty:

$$
\mathrm{novelty}_i = 1 - \max_j S_{ji}.
$$

A high novelty score indicates that an observed map is not well explained by any canonical template. Collisions should also be reported when two canonical maps choose the same best observed representative. If

$$
b(j) = \arg\max_i S_{ji}
$$

and, for instance, $b(C)=b(D)$, then the result is better described as a C-D collision or merger than as a clean class-specific D reduction.

### (Deconfounded representation-level distortion)

The profile-based metric above corrects for the fact that canonical maps are not orthogonal by comparing the subject-canonical similarity matrix with the canonical self-similarity matrix. A stronger correction can be obtained by explicitly removing the similarity structure of the canonical maps.

Using signed correlations, define

$$
G = M^\top X.
$$

If the subject maps were exactly a permuted version of the canonical maps, then

$$
G P^\star = M^\top M = G_M.
$$

Equivalently, before alignment, the cross-similarity matrix should be explainable by the canonical Gram matrix up to a permutation. We can therefore define a deconfounded coefficient matrix

$$
B = G_M^{-1} G.
$$

If the subject maps are a permuted copy of the canonical maps, then $B$ should be close to a permutation matrix. When $G_M$ is ill-conditioned, a ridge-stabilized version should be preferred:

$$
B_\lambda = (G_M + \lambda I)^{-1} G.
$$

The deconfounded assignment can then be obtained by solving

$$
\pi^\star_\lambda =
\arg\max_{\pi}
\sum_i (B_\lambda)_{\pi(i),i},
$$

or equivalently by applying the Hungarian algorithm to the cost matrix

$$
A^\lambda_{ij} = -(B_\lambda)_{ji}.
$$

After the best assignment is found, the deconfounded distortion is

$$
D_{\lambda} =
\frac{||B_\lambda - P^\star_\lambda||_F}{\sqrt{J}}.
$$

The corresponding map-wise anomaly score is

$$
a_i =
||(B_\lambda)_{:,i} - e_{\pi^\star_\lambda(i)}||_2,
$$

where $e_j$ is the $j$-th canonical basis vector. A large $a_i$ means that subject map $x_i$ is not well represented as a single canonical template after accounting for the non-orthogonality of the canonical maps.

This score is useful for distinguishing three situations. If $S^\star$ is not close to diagonal but $D_{\mathrm{match}}$ is low, then the apparent off-diagonal structure is largely expected from the canonical map correlations. If $D_{\mathrm{match}}$ is high but $D_\lambda$ is low, then the difference is mostly explained by the non-orthogonal structure of the canonical frame. If both $D_{\mathrm{match}}$ and $D_\lambda$ are high, then at least one subject map is probably not a credible instance of the canonical set.

Thus, the recommended map comparison report should include both the directly interpretable aligned squared correlations and the corrected geometry-aware distortion, while keeping permutation-invariant collapse and novelty separate from assigned-label failure:

$$
{S^\star_{jj}}_{j=1}^J,
\quad
R^2_{\min},
\quad
{\Delta_i}_{i=1}^J,
\quad
{c_j^{\mathrm{PI}}}_{j=1}^J,
\quad
{\mathrm{novelty}_i}_{i=1}^J,
\quad
D_{\mathrm{match}},
\quad
D_\lambda.
$$


## Labeling uncertainty in backfitting

The next question is whether each sample can be assigned reliably to a microstate class. Standard backfitting assigns each normalized scalp map to the template with the largest absolute topographic correlation.

For a normalized sample $\tilde x_t$ and template maps $\tilde b_m$, define

$$
c_m(t) = \tilde x_t^\top \tilde b_m.
$$

In the polarity-invariant convention, the relevant quantities are

$$
|c_m(t)|
$$

or

$$
c_m(t)^2.
$$

The hard label is

$$
\hat m(t) = \arg\max_m |c_m(t)|.
$$

This label is only the final decision. It does not describe the strength or reliability of the decision. Assignment uncertainty has several components.

The first component is winner strength. Let $c_{(1)}(t)$ be the largest absolute correlation. If $c_{(1)}(t)$ is low, the sample is poorly explained by every template. This is a poor-evidence case.

The second component is local assignment fragility. Let $c_{(2)}(t)$ be the second-largest absolute correlation. The label is fragile when the runner-up is close to the winner. This can be measured by the gap

$$
\Delta c(t) = c_{(1)}(t) - c_{(2)}(t)
$$

or by the ratio

$$
r(t) = \frac{c_{(2)}(t)}{c_{(1)}(t)}.
$$

A high winning correlation does not guarantee a reliable assignment if the runner-up is almost as strong.

The third component is global ambiguity. After normalizing the similarity values into a distribution $p_m(t)$, one can compute an effective number of competitors:

$$
N_{\mathrm{eff}}(t) =
\exp \left(
-\sum_m p_m(t)\log p_m(t)
\right).
$$

The normalized ambiguity index is

$$
\frac{N_{\mathrm{eff}}(t)}{K}.
$$

This measures whether similarity is concentrated on one or two templates, or diffusely spread across the whole template set.

The fourth component is structural redundancy among templates. If the winning template and the runner-up template are themselves highly correlated, ambiguity may arise from the map set rather than from noise in the sample. This can be described by

$$
\mathrm{corr}(b_{\hat m(t)}, b_{m_2(t)}),
$$

where $m_2(t)$ is the runner-up class.

A complete labeling-uncertainty description should therefore report winner strength, runner-up fragility, global ambiguity, and winner-runner-up template redundancy. These quantities separate four distinct cases: weak evidence, local competition, diffuse ambiguity, and structural redundancy.

This matters because a hard label sequence hides the uncertainty that created it. Once the sequence has been reduced to symbols, later metrics no longer know whether a state was assigned cleanly or only barely won against a competitor.

## How lower-level problems affect temporal metrics

Problems at Level 0 and Level 1 propagate upward.

At the zero-order level, uncertain labels affect duration, occurrence, and coverage. A noisy boundary can split one true episode into several short episodes, increasing occurrence and reducing duration. Smoothing can do the opposite: it can merge short ambiguous segments, increasing apparent duration. Peak-only labeling can miss states that occur between GFP peaks. Therefore, zero-order metrics should not be interpreted without knowing whether the underlying labels were stable.

At the first-order level, uncertain labels affect transition probabilities. If a sample lies between two templates, small perturbations can create apparent rapid switching. If maps are structurally redundant, transitions between those classes may reflect template geometry rather than true sequence dynamics. If raw sample-wise transitions are used, duration also contaminates the transition matrix because long states generate many self-transitions.

A cleaner transition analysis uses a collapsed interval sequence. Consecutive repeated labels are reduced to one state visit. For example,

$$
A,A,A,B,B,C,C,C,A
$$

becomes

$$
A,B,C,A.
$$

The transition matrix computed on this collapsed sequence describes state switching rather than sample-wise persistence. Even then, transition probabilities remain prevalence-dependent, so residual or adjusted transition metrics are preferable.

At the higher-order level, the risk is larger. Apparent symbolic memory may be produced by smoothing, ambiguous boundaries, repeated uncertain transitions, or redundancy between maps. Higher-order Markov models, entropy-rate estimates, motif counts, and memory-depth metrics should therefore be tested against lower-order null models that preserve basic state usage and first-order transition structure.

The main principle is that higher-order metrics should only be interpreted after the map set, assignment process, and first-order structure have been characterized. Otherwise, high-level sequence descriptors may simply quantify unresolved lower-level uncertainty.

## MEED

MEED is introduced to add a field-geometric layer to microstate analysis. It should be framed as a methodological representation of scalp topographies, not as source localization.

The goal is not to claim that each microstate is generated by a literal single dipole. The goal is to use a fixed dipole-like scaffold as a compact descriptor of smooth bipolar scalp fields. This is appropriate because canonical microstate maps often have broad, smooth, dipolar-looking topographies, but the interpretation remains at the level of scalp-field geometry.

Each centered and normalized scalp map $\tilde x_t$ is projected into a fixed, geometry-corrected MEED basis:

$$
q_t = S_{\mathrm{orth}}^\top \tilde x_t.
$$

The vector $q_t$ is decomposed into a magnitude and a direction:

$$
q_t = \rho_t u_t
$$

with

$$
\rho_t = |q_t|_2
$$

and

$$
u_t = \frac{q_t}{\rho_t}.
$$

The retained continuous coordinate is the angular direction of $u_t$:

$$
D_t = (\theta_t, \phi_t).
$$

The projection score is

$$
R_t^2 = \rho_t^2.
$$

This measures how much of the normalized scalp map lies inside the fixed-MEED subspace. In practical terms, $R_t^2$ is a sample-wise measure of how well the map is represented by the fixed dipole-like field scaffold.

This is the main conceptual separation introduced by MEED. A sample has a field strength, measured by GFP. It has temporal instability, measured by TD or DISS. It has template proximity, measured by correlation with microstate maps. It also has field conformity, measured by $\rho_t^2$. These are related but not identical.

## MEED as an interpretable representation

MEED turns each instantaneous scalp map into a low-dimensional continuous descriptor:

$$
\tilde x_t \mapsto (\theta_t, \phi_t, \rho_t).
$$

The angles describe where the sample lies in the fixed MEED angular scaffold. The scalar $\rho_t$ describes how strongly the sample is represented by that scaffold.

For each template map $\tilde b_m$, the same projection is computed:

$$
q_m = S_{\mathrm{orth}}^\top \tilde b_m.
$$

Then

$$
q_m = \rho_m u_m
$$

and the template has MEED coordinates

$$
D_m = (\theta_m, \phi_m).
$$

The distance between a data sample and a template in MEED space can be computed as the polarity-invariant angular distance

$$
d_{\pm}(u_t,u_m) =
\arccos(|u_t^\top u_m|).
$$

This makes MEED compatible with standard polarity-invariant microstate analysis while still allowing polarity-sensitive variants to be audited.

The interpretability comes from the fact that deviations are directional. A sample can deviate from a template mainly in the transverse angle, mainly in the sagittal angle, or mainly in projection strength. This is more informative than a single correlation value.

This can be useful when two maps are correlated in sensor space but separated in angular structure. In that case, classical correlation-based labeling may treat the sample as ambiguous, while MEED may show that the ambiguity is concentrated along a specific angular component.

The important constraint is that angular interpretation depends on $\rho$. A point can land near a template in the $(\theta,\phi)$ plane while having low $\rho$. In that case, the apparent angular proximity is weak evidence because the sample is not well represented by the MEED scaffold. Therefore, MEED visualizations should always encode $\rho$ or $R^2=\rho^2$ together with angular position.

## MEED as an uncertainty engine

MEED extends the uncertainty problem beyond template competition. Classical backfitting asks which template is closest. MEED also asks whether the sample is well represented by the field family in which the templates are being compared.

This produces a more explicit uncertainty profile for each sample. A sample should be described by at least five quantities:

* winner strength, given by $c_{(1)}(t)$
* local fragility, given by $c_{(2)}(t)/c_{(1)}(t)$ or $c_{(1)}(t)-c_{(2)}(t)$
* global ambiguity, given by $N_{\mathrm{eff}}(t)/K$
* structural redundancy, given by the correlation between winner and runner-up templates
* MEED conformity, given by $\rho_t^2$

These quantities distinguish cases that are otherwise collapsed by a hard label.

If $c_{(1)}(t)$ is low and $\rho_t^2$ is low, the sample has poor template evidence and poor field conformity.

If $c_{(1)}(t)$ is high but $\rho_t^2$ is low, the sample is close to a template but contains substantial non-MEED residual structure.

If $\rho_t^2$ is high but $c_{(2)}(t)/c_{(1)}(t)$ is also high, the sample is dipole-like but lies between templates.

If $\rho_t^2$ is high, $c_{(1)}(t)$ is high, and $c_{(2)}(t)/c_{(1)}(t)$ is low, the assignment is comparatively reliable.

This is why MEED can be used as an uncertainty engine. It does not replace classical labeling. It qualifies it. The label becomes one part of a richer sample-wise evidence profile.

## MEED, GFP, and TD

MEED should be interpreted together with GFP and TD.

GFP measures the strength of the electric field. TD or DISS measures topographic change between adjacent normalized maps. Template correlation measures proximity to a microstate class. MEED $\rho^2$ measures whether the normalized field is well represented by the fixed dipole-like scaffold.

These variables can dissociate. A sample can have high GFP but poor template separability. A sample can have low TD but still have low MEED conformity. A sample can be close to a template but have a large angular displacement from the corresponding MEED direction. Therefore, none of these quantities should be treated as a substitute for the others.

MEED also allows topographic instability to be decomposed. Let the normalized scalp map be written as

$$
z_t = p_t + e_t,
$$

where $p_t$ is the MEED-projected component and $e_t$ is the orthogonal residual. Because the MEED scaffold is orthonormal, adjacent change can be separated into projected and residual components.

The projected component can be further decomposed into a radial term and an angular term. The radial term reflects changes in $\rho$, meaning changes in dipolarity or MEED conformity. The angular term reflects changes in fixed-dipole direction. The residual term reflects non-MEED instability.

Thus, topographic change can be partitioned into:

* dipolarity change
* dipole-direction change
* non-MEED residual instability

This is useful because a stable high-GFP interval is not necessarily a clean microstate interval. It may be stable in full sensor space but unstable in MEED angle, or stable in angle but weakly represented by MEED.

## MEED with synthetic data: distribution of $\rho^2$

Synthetic data are necessary to understand what $\rho^2$ means. The key validation is not only that MEED fits canonical microstate maps. The key validation is that it rejects maps that should not be interpreted as microstate-like.

Three null families are especially useful.

First, iid random maps are generated as channel-wise random noise, then centered and normalized. These maps contain no smooth spatial structure by construction.

Second, low-order smooth maps are generated from low spatial-frequency structure. These maps resemble broad dipolar or low-order field patterns.

Third, high-order maps are generated from finer spatial-frequency structure. These maps contain structured topographies but are less dipole-like.

Fixed MEED should behave strictly. It should give low $\rho^2$ to iid random maps. It should give lower $\rho^2$ to high-order maps than to low-order smooth maps. It should give higher $\rho^2$ to smooth low-order maps because these are closer to the field family in which canonical microstate maps live.

This validation separates fixed MEED from a more flexible full dipole fit. A full dipole model can increase fit by optimizing source location and orientation. This flexibility can be useful as a diagnostic comparison, but it also means that high full-dipole fit is not by itself strong evidence of microstate-like structure. Fixed MEED is more constrained and therefore more useful as a strict field-conformity measure.

The expected result is therefore not simply

$$
R^2_{\mathrm{full}} > R^2_{\mathrm{fixed}}.
$$

That inequality is expected because the full model is more flexible. The meaningful result is the distribution of fixed-MEED $\rho^2$ across null families:

$$
\rho^2_{\mathrm{iid}} < \rho^2_{\mathrm{low-order}}
$$

and usually

$$
\rho^2_{\mathrm{high-order}} < \rho^2_{\mathrm{low-order}}.
$$

This gives $\rho^2$ an operational interpretation. It becomes a measure of how much an instantaneous scalp map behaves like a smooth microstate-like field under a fixed geometry-corrected scaffold.

## MEED as a low-dimensional continuous sequence

Once MEED is applied sample-wise, the recording is transformed into a continuous sequence:

$$
t \mapsto (\theta_t, \phi_t, \rho_t).
$$

This sequence can be analyzed alongside the classical microstate sequence:

$$
t \mapsto \hat m(t).
$$

The real-data feature table should therefore contain both symbolic and continuous descriptors. For each sample, the useful columns are:

* subject and session identifiers
* time
* GFP
* TD or DISS
* correlations with meta maps
* correlations with subject maps
* hard label under meta maps
* hard label under subject maps
* $\theta_t$
* $\phi_t$
* $\rho_t$
* $R_t^2=\rho_t^2$
* angular distance to each meta template
* angular distance to each subject template

This transforms microstate analysis from a purely symbolic sequence problem into a mixed symbolic-continuous problem. A transition is no longer only a change from label A to label B. It can also be described as a movement in angular space, a change in $\rho$, and a change in distance to competing templates.

This is important for continuous approaches to microstates. Some transitions may be abrupt in label space but gradual in MEED space. Others may appear as rapid symbolic switching but actually reflect lingering near a boundary between two templates. Conversely, a sample may retain the same hard label while drifting substantially in angle or losing MEED conformity.

MEED therefore provides a way to study microstate trajectories without abandoning the classical labels. The hard label remains useful, but it is embedded in a continuous field-geometric representation.

## Practical analysis order

The practical order should be:

1. Validate the normative or meta maps and their Gram structure.
2. Fit subject maps and align them to the meta maps with Hungarian assignment.
3. Report diagonal $R^2$, minimum match, assignment margins, permutation-invariant collapse, novelty, collisions, and geometry distortion.
4. Compute MEED coordinates for meta, subject, and optional group maps.
5. Plot $(\theta,\phi)$ while encoding $\rho$ or $R^2$.
6. Compute sample-wise GFP, TD, template correlations, MEED coordinates, and angular distances.
7. Quantify assignment uncertainty using winner strength, runner-up fragility, global ambiguity, template redundancy, and $\rho^2$.
8. Use these uncertainty variables to qualify zero-order and first-order metrics.
9. Interpret higher-order sequence metrics only after lower-level uncertainty has been characterized.
10. Use synthetic and null data to define the expected distribution of $\rho^2$ under non-microstate-like maps.

The logic is cumulative. Map credibility comes before label credibility. Label credibility comes before duration and transition metrics. Duration and transition metrics come before higher-order sequence claims.

## Final framing

MEED should be presented as an uncertainty-aware extension of EEG microstate analysis.

Classical microstate analysis remains useful. It gives a compact symbolic description of scalp topography and produces interpretable temporal metrics. The problem is not that the classical framework is wrong. The problem is that a hard label sequence hides several forms of uncertainty.

MEED adds a complementary layer. It separates field strength, temporal stability, template proximity, angular displacement, and field conformity. It provides a low-dimensional continuous trajectory for instantaneous scalp maps and a sample-wise projection score that helps identify whether a map is microstate-like under a fixed field-geometric scaffold.

The defensible claim is therefore narrow: MEED is a geometry-corrected, polarity-auditable, low-dimensional representation of scalp voltage topographies that can support assignment uncertainty, continuous trajectory analysis, and reliability-aware sequence metrics.

It is not source localization. It does not claim that each microstate is generated by one neural dipole. It uses a fixed dipole-like scaffold as an interpretable descriptor of scalp-field geometry.
