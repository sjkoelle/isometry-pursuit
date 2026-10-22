# Projective and RAG datasets for Isometry Pursuit

## Recommendation

The next two experiments should be:

1. **York Urban Manhattan-frame recovery**, because it is a direct application on real data where a vector and its negative genuinely represent the same object.
2. **TREC RAG 2025 passage selection in a query-specific sub-narrative space**, because the benchmark already supplies multi-faceted queries, top-100 candidate segments, sub-narratives, relevance judgments, nuggets, and coverage-oriented evaluation.

York Urban tests the geometric claim of the method without changing its meaning. TREC RAG tests a useful retrieval application after replacing ordinary document embeddings with a small, nonnegative passage-to-facet matrix. MuSiQue is the best small RAG pilot after those two: each example has only 20 candidate paragraphs and 2--4 annotated reasoning hops.

Ordinary dense-embedding diversification is not a direct application. In a semantic embedding, $x$ and $-x$ usually mean different things, while the Isometry Pursuit loss is unchanged by independently flipping the sign of any selected column. The report therefore distinguishes exact projective applications, nonnegative facet-space applications where antipodes cannot occur, and speculative adaptations.

No datasets were downloaded while preparing this report.

## The modeling test

Let $X=[x_1,\ldots,x_P]\in\mathbb R^{D\times P}$, with normalized columns, and let $S$ contain $D$ selected columns. For any diagonal sign matrix $J\in\{-1,1\}^{D\times D}$,

\[
(X_SJ)^\dagger=JX_S^\dagger,
\]

so row norms of the pseudoinverse, and hence the current loss, do not change. The selected objects therefore live naturally in real projective space: a column represents the line

\[
[x]=\{x,-x\}.
\]

This is exactly right for unoriented axes, plane normals, and calibrated vanishing directions. It is harmless for a nonnegative facet matrix because $-x$ is outside the representation. It is generally wrong for raw text embeddings because antipodal semantic vectors are not equivalent documents.

A second constraint is just as important: the implementation selects exactly as many columns as the row dimension. A 768-dimensional document embedding asks the method to select 768 passages. For a desired context budget of $K$ passages, the useful construction is therefore a query-specific $K\times P$ feature matrix, not a generic 768-by-$P$ embedding matrix.

## Dataset shortlist

| Priority | Dataset | Proposed $D\times P$ instance | Selected columns | Fit | Ground truth / evaluation | Access and license |
|---:|---|---|---|---|---|---|
| 1 | York Urban | $3\times P$ calibrated candidate vanishing directions per image | Three Manhattan axes | **Direct projective** | Angular frame error after sign/permutation matching; recovery of the three labeled directions | [Official York page](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/); 102 images, 640x480, camera calibration, hand-labeled lines and Manhattan frames. The page provides a public download but does not state a standard license; confirm reuse terms before redistribution. |
| 2 | TREC RAG 2025 | $D_q\times100$ passage-to-sub-narrative support scores for query $q$ | $D_q$ context segments | **Direct facet-space RAG** | sub-narrative and vital-nugget recall, support precision, nDCG, answer completeness | [Official track](https://trec-rag.github.io/trec25/), [2025 overview](https://trec.nist.gov/pubs/trec34/papers/Overview_rag.pdf), [corpus details](https://trec-rag.github.io/annoucements/2025-rag25-corpus/). Corpus terms follow MS MARCO/TREC; verify before redistribution. |
| 3 | MuSiQue | $h\times20$, where $h\in\{2,3,4\}$ is the hop count and each entry scores support for an atomic question | One paragraph per reasoning hop | **Direct nonnegative evidence selection** | supporting-paragraph F1/recall and answer F1 | [Official repository](https://github.com/stonybrooknlp/musique), CC BY 4.0. About 25K 2--4-hop questions; each question has 20 paragraphs. |
| 4 | HotpotQA | $2\times P$ support scores for the two evidence roles, initially with the distractor contexts and later top-$P$ full-wiki retrieval | Two evidence documents or a small sentence set | **Useful but low-dimensional** | supporting-fact EM/F1 and answer EM/F1 | [Official site](https://hotpotqa.github.io/), CC BY-SA 4.0; 113K QA pairs. Train download is 535 MB; processed Wikipedia is 7.4 GB compressed. |
| 5 | HoVer | $h\times P$, $h\in\{2,3,4\}$, using claim-hop support scores | Evidence articles for claim verification | **Direct nonnegative evidence selection** | document recall by hop, joint evidence/label score, claim accuracy | [Official site](https://hover-nlp.github.io/), CC BY-SA 4.0. Annotation downloads are 9.2 MB train, 2.2 MB dev, and 899 KB test; it uses the HotpotQA processed Wikipedia corpus. |
| 6 | TREC Web diversity, especially 2009--2012 | subtopic-by-candidate graded relevance matrix | Results covering distinct query intents | **Direct oracle facet benchmark** | official alpha-nDCG, ERR-IA, intent-aware precision, subtopic recall | [NIST 2012 files](https://pages.nist.gov/trec-browser/trec21/web/data/) and [2010 files](https://trec.nist.gov/data/web10.html) include topics, diversity qrels, and `ndeval`. ClueWeb corpus access is separately licensed. |
| 7 | ASQA / AmbigQA | disambiguated-question-by-candidate evidence score matrix | Passages covering distinct interpretations | **Strong coverage task; scoring layer required** | disambiguation QA accuracy, ROUGE, citation/evidence recall | [ASQA catalog](https://github.com/tensorflow/datasets/blob/master/docs/catalog/asqa.md) reports 4,353 train and 948 dev examples, 17.86 MiB download; [AmbigQA site](https://ambigqa.github.io/) offers 18 MB ASQA and 575 MB evidence variants. Confirm dataset-specific terms before redistribution. |
| 8 | QAMPARI | answer-cluster-by-candidate support matrix, with $D$ capped to the context budget | Passages covering many valid answers | **Promising coverage stress test** | answer precision, recall, F1, Recall>=0.8 and F1>=0.5 | [Official repository](https://github.com/samsam3232/qampari), CC0 repository; 61,911 train, 1,000 dev, and 1,000 test questions. |
| 9 | NYU-VP | $3\times P$ candidate directions for Manhattan scenes; variable $D$ only after extending the fixed-cardinality interface | Unoriented vanishing directions | **Projective, but some scenes are non-Manhattan** | angular VP error and model-count accuracy | [Official labels/code](https://github.com/fkluger/nyu_vp) for NYU Depth v2; the labels expose vanishing points and line segments. Original NYUv2 images require the separate official download and its terms. |
| 10 | Structured3D / SU3 | $3\times P$ plane-normal or vanishing-direction candidates | Three Manhattan axes | **Direct synthetic projective scale-up** | exact angular frame error and robustness to outliers | [Structured3D](https://structured3d-dataset.org/) has 3.5K house designs and rich plane/normal/Manhattan annotations under its own terms; [SU3 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Learning_to_Reconstruct_3D_Manhattan_Wireframes_From_a_Single_Image_ICCV_2019_paper.pdf) reports 23K rendered images. Large download; use only after York works. |

## 1. York Urban: the clean projective benchmark

The [official dataset description](https://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) gives 102 calibrated images, including 45 indoor and 57 outdoor scenes. It supplies hand-labeled line segments, camera calibration, and the estimated three-axis Manhattan frame. These axes are unoriented: $v$ and $-v$ are the same 3D direction. The method's sign invariance is therefore a feature rather than a semantic approximation.

For image $i$:

1. Detect or use the labeled image lines.
2. Intersect promising line pairs to obtain homogeneous vanishing-point hypotheses.
3. Calibrate each hypothesis with $K_i^{-1}$, normalize it to a unit vector in $\mathbb R^3$, and identify antipodes.
4. Form $X_i\in\mathbb R^{3\times P_i}$ from the resulting candidate directions.
5. Ask every method to select three columns.

The main metric should match estimated axes to the three ground-truth axes over all $3!$ permutations and both signs, then report mean and median angular error:

\[
\min_{\pi\in S_3}\frac{1}{3}\sum_{j=1}^3
\arccos\!\left(\left|\widehat v_j^\top v_{\pi(j)}\right|\right).
\]

Also report the fraction of images for which all three errors are below 2, 5, and 10 degrees, objective value, support size after IP, arithmetic-work proxy, and wall time. Candidate generation must not use the ground-truth labels. It is reasonable to make separate candidate pools from the hand-labeled lines and from an automatic line detector; the former isolates selection quality and the latter measures the full pipeline.

The first pilot should cap $P$ at 100, 250, and 500 using line length and pairwise intersection plausibility. This creates a useful scaling experiment while keeping backward deletion manageable.

## 2. TREC RAG 2025: the strongest retrieval application

The 2025 track is unusually well matched to a facet representation. The [official overview](https://trec.nist.gov/pubs/trec34/papers/Overview_rag.pdf) says that participants received 105 multi-sentence narratives. Each narrative was decomposed into human-reviewed sub-narratives; relevant documents yielded information nuggets that were mapped back to those sub-narratives. Retrieval returned up to 100 segments. Manual assessment was completed for a 22-narrative subset, so that subset should be the confirmatory evaluation and the remaining narratives can be exploratory.

For narrative $q$, let $D_q$ be the number of retained sub-narratives and let the supplied top-100 segments be the candidate set. Construct

\[
X^{(q)}_{dp}=s(\text{sub-narrative }d,\text{ segment }p),
\qquad X^{(q)}\in\mathbb R_+^{D_q\times100},
\]

where $s$ is either an oracle support label derived from the released assessments or a learned cross-encoder/entailment score. The oracle matrix answers whether the selection objective is appropriate. The learned matrix tests a deployable pipeline. Columns are nonnegative, so the unwanted $x\sim -x$ identification never arises.

Selected columns are context segments. Ideally, an isometric subset contains one nearly pure, nonredundant segment for each sub-narrative. Real segments cover multiple facets, so the loss measures whether their support profiles form a well-conditioned basis rather than merely maximizing pairwise textual distance.

The experimental pipeline should be:

1. Use the official reference top-100 list. Do not index the full corpus for the pilot.
2. Retain narratives with $3\le D_q\le10$; if a narrative has more facets than the context budget, merge or rank facets using a rule fixed before evaluation.
3. Compare relevance top-$D_q$, MMR, xQuAD, a DPP-style determinant baseline, forward greedy, backward greedy, and IP-based selection.
4. Measure sub-narrative coverage, vital-nugget recall, support precision, nDCG, and downstream answer completeness with the official scripts where available.
5. Run the same generator with the same prompt and token budget for every selected context set.

The [official corpus page](https://trec-rag.github.io/annoucements/2025-rag25-corpus/) lists 10,960,555 documents (28.1 GB compressed) and 113,520,750 pre-segmented passages (25.1 GB compressed). Full-corpus retrieval is unnecessary for the selection pilot. Download the corpus only if reranking the supplied candidate texts cannot be done from the released AG materials.

Two limitations must be explicit. First, relevance magnitude is not naturally preserved by column normalization, so relevance should enter through the initial top-$P$ screen and be reported separately. Second, row whitening in the current transformation may weaken hand-specified facet importance weights. The first RAG experiment should therefore claim diversity/coverage selection conditional on a relevant candidate pool, not joint retrieval and diversification.

## 3. MuSiQue, HotpotQA, and HoVer: evidence-set selection

These benchmarks supply annotated multi-document evidence, which makes selected-support evaluation much cleaner than generic retrieval datasets.

### MuSiQue

MuSiQue is the easiest end-to-end pilot. The [paper](https://aclanthology.org/2022.tacl-1.31.pdf) constructs 2--4-hop questions and gives each question 20 paragraphs containing the supporting paragraphs plus distractors. The [official repository](https://github.com/stonybrooknlp/musique) is CC BY 4.0 and includes official support and answer evaluation.

For an $h$-hop question, build $X\in\mathbb R_+^{h\times20}$. Row $d$ is the atomic question at hop $d$; column $p$ scores whether paragraph $p$ supports it. In the oracle experiment, use binary gold hop-to-paragraph membership. In the learned experiment, score each atomic question/paragraph pair with a fixed entailment or reranking model. Select $h$ paragraphs and measure support recall/F1 plus answer F1.

The oracle experiment is essential: if the objective cannot recover the gold evidence from an exact nonnegative matrix, representation learning is irrelevant. A caveat is that multi-hop evidence has order and dependency, while orthogonality only models coverage. Report results separately by hop count.

### HotpotQA

[HotpotQA](https://hotpotqa.github.io/) provides 113K multi-hop questions and sentence-level supporting facts under CC BY-SA 4.0. Start with its distractor setting, which already supplies a compact candidate context. A $2\times P$ matrix can represent the two evidence roles, but $D=2$ is a weak stress test of selection. Its main value is downstream validation: supporting-fact EM/F1 and answer EM/F1 are standard and directly available.

### HoVer

[HoVer](https://hover-nlp.github.io/) uses two to four Wikipedia articles to verify claims and distributes compact annotations under CC BY-SA 4.0. It shares HotpotQA's processed Wikipedia corpus. Form one row per evidence hop or atomic claim and one column per top-$P$ retrieved article. Evaluate evidence recall, joint evidence/verification performance, and accuracy. HoVer is more informative than HotpotQA for scaling in $D$, but learned hop decomposition is an additional component.

## 4. Coverage and ambiguity datasets

### TREC Web diversity

The TREC Web diversity tasks are the standard diversification benchmark that fits the method most cleanly. NIST publishes topics, subtopics, diversity qrels, and the `ndeval` evaluator; for example, see the [2012 data page](https://pages.nist.gov/trec-browser/trec21/web/data/) and [2010 data page](https://trec.nist.gov/data/web10.html). Each topic has multiple intents. An oracle matrix uses rows as intents and columns as judged documents, with entries equal to graded intent relevance.

This is meaningful because the desired set covers distinct labeled intents. It is not an evaluation on ordinary text embeddings. Compare with xQuAD/MMR and report alpha-nDCG, ERR-IA, intent-aware precision, and subtopic recall. The practical obstacle is access to the underlying ClueWeb documents; topic and qrel files are public, while the corpus is separately licensed. A first oracle study can use only judged document IDs and relevance labels, but a learned scorer requires document text.

### ASQA / AmbigQA

[ASQA](https://github.com/tensorflow/datasets/blob/master/docs/catalog/asqa.md) contains ambiguous questions, disambiguated QA pairs, evidence, and long-form answers. The TensorFlow catalog reports 4,353 train and 948 dev examples and a 17.86 MiB download. Rows represent disambiguated interpretations, columns represent retrieved passages, and entries score whether a passage answers an interpretation. This supports a direct coverage test, but the passage-to-interpretation scorer must be supplied. Evaluate disambiguation QA accuracy, evidence recall, citation correctness, and the official long-answer measures.

### QAMPARI

[QAMPARI](https://github.com/samsam3232/qampari) asks questions with many valid answers spread across multiple paragraphs. It has 61,911 train, 1,000 dev, and 1,000 test questions; the repository is marked CC0. It is attractive for coverage, but the number of answers can greatly exceed a practical context budget. Cluster or rank answers into exactly $D$ target facets using a predeclared rule, then form answer-cluster-by-passage scores. Official metrics include answer precision, recall, F1, Recall>=0.8, and F1>=0.5.

This is a later-stage experiment because facet construction could dominate the result and make the method hard to interpret.

## 5. Larger projective vision datasets

### NYU-VP

The [NYU-VP repository](https://github.com/fkluger/nyu_vp) adds vanishing-point and line-segment labels to NYU Depth v2. It exposes vanishing points as homogeneous 3-vectors and line candidates directly. Manhattan scenes can use the same $3\times P$ construction as York Urban. Non-Manhattan scenes require either filtering or extending the algorithm to a target cardinality different from the ambient dimension; they should not be silently forced into a three-axis model.

### Structured3D and SU3

[Structured3D](https://structured3d-dataset.org/) contains 3.5K professionally designed houses with plane, Manhattan-structure, normal, depth, and camera annotations under a data-use agreement. The [SU3 wireframe paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Learning_to_Reconstruct_3D_Manhattan_Wireframes_From_a_Single_Image_ICCV_2019_paper.pdf) reports 23K rendered urban images with exact camera-derived vanishing points. Both provide a scalable geometric follow-up with exact or nearly exact ground truth, but their size and preprocessing burden are unnecessary until the York pipeline is validated.

## Approaches that should not be presented as direct fits

### Raw dense text embeddings

Using $X\in\mathbb R^{768\times P}$ from a sentence encoder would select 768 passages and would identify $x$ with $-x$, even though cosine similarity treats them as maximally different. Reducing to $K$ dimensions with PCA makes the requested subset size convenient, but sign equivalence remains a representation artifact and the result depends strongly on the projection. This can be an exploratory baseline, not the main RAG formulation.

### Rank-one projector lifting

The map

\[
\phi(x)=\operatorname{vec}_{\rm sym}(xx^\top)
\]

is genuinely sign invariant and satisfies

\[
\langle\phi(x),\phi(y)\rangle=(x^\top y)^2.
\]

It converts semantic vectors into projective features, but a $d_e$-dimensional embedding lifts to $d_e(d_e+1)/2$ dimensions. The current algorithm would then try to select that many passages. Random compression reduces the dimension but again ties it to the requested selection count. This is mathematically interesting and operationally unsuitable for the present implementation.

### Generic max-sum diversity benchmarks

Benchmarks that reward large pairwise distance are not automatically suitable. A set containing $e_1$ and $-e_1$ has large Euclidean separation but only one projective direction; Isometry Pursuit correctly treats it as redundant. Include a conventional benchmark only when it supplies facets, evidence roles, unoriented axes, or another interpretation under which projective diversity is the desired target.

## Work and storage estimates

The selection matrices themselves are small. A float64 $D\times P$ matrix uses $8DP$ bytes:

| Setting | Shape | Matrix storage | Forward candidate evaluations | Backward candidate evaluations |
|---|---:|---:|---:|---:|
| York, small | $3\times100$ | 2.4 KB | 297 | 5,044 |
| York, medium | $3\times500$ | 12 KB | 1,497 | 125,244 |
| York, large | $3\times1000$ | 24 KB | 2,997 | 500,494 |
| MuSiQue, four-hop | $4\times20$ | 640 B | 74 | 200 |
| RAG, five facets | $5\times100$ | 4 KB | 490 | 5,035 |
| RAG, ten facets | $10\times100$ | 8 KB | 955 | 4,995 |

The counts use

\[
N_{\rm FG}=PD-\frac{D(D-1)}2,
\qquad
N_{\rm BG}=\frac{P(P+1)-D(D+1)}2.
\]

Text scoring, candidate generation, and corpus indexing will dominate storage and often runtime:

- **York:** line-pair hypothesis generation is $O(L^2)$ before deduplication. Capping $P$ makes all selection methods inexpensive except repeated full backward runs.
- **MuSiQue:** at most $4\times20=80$ paragraph/facet scores per example. The full experiment is modest even with a cross-encoder.
- **TREC RAG pilot:** with 105 narratives, ten facets, and 100 candidates, there are at most 105,000 passage/facet scores. The manually judged 22-query subset needs at most 22,000. Batched GPU inference is feasible; LLM API scoring could still be expensive and less reproducible.
- **TREC full corpus:** 25.1 GB compressed for segments and 113.5M records. Index construction, embedding storage, and retrieval dwarf subset selection. Avoid this in the first study.
- **HotpotQA/HoVer full wiki:** the shared processed Wikipedia archive is 7.4 GB compressed. The compact distractor/annotation settings are much cheaper pilots.
- **ASQA:** the annotated dataset is tens of megabytes, but retrieving and scoring external Wikipedia passages adds the main cost.

Selection runtime should be reported separately from representation construction and first-stage retrieval. Otherwise a fast $D\times P$ selection algorithm can appear expensive because it inherits cross-encoder or LLM scoring time.

## Prioritized experiment roadmap

### Phase 0: representation checks

1. Add a test that all methods return the same objective after independent column sign flips.
2. Build synthetic $\mathbb{RP}^2$ instances with three planted axes, antipodal duplicates, angular noise, and outliers.
3. Verify recovery as a function of $P$, noise, duplicated antipodes, and near-collinear distractors.

This isolates the defining projective behavior before involving image or language pipelines.

### Phase 1: York Urban

1. Prepare all 102 images from official calibration and line annotations.
2. Generate candidate pools at $P=100,250,500$ without ground-truth guidance.
3. Run forward greedy, backward greedy, early-stopped IP, and the existing common swap refinement.
4. Compare to a standard Manhattan-frame/RANSAC estimator.
5. Report sign/permutation-invariant angular error, thresholded success, arithmetic work, and controlled runtime.

This should become the primary new real-data experiment if it works: the geometry and evaluation match the method exactly.

### Phase 2: TREC RAG 2025 oracle pilot

1. Use the official top-100 candidate lists rather than retrieving the 113.5M-segment corpus.
2. Start with the 22 manually assessed narratives.
3. Build oracle binary or graded sub-narrative support matrices from the released assessment artifacts.
4. Compare top relevance, MMR, xQuAD, DPP, forward greedy, backward greedy, and IP hybrids.
5. Evaluate facet/nugget coverage before running a generator.

This tests the selection objective without conflating it with a learned scorer.

### Phase 3: learned facet scores and downstream RAG

1. Freeze one passage-to-sub-narrative scorer.
2. Repeat TREC selection on learned nonnegative scores.
3. Generate answers with a fixed model, prompt, ordering rule, and token budget.
4. Use official completeness, support, attribution, and retrieval metrics.
5. Report scorer cost, selection cost, and generation cost separately.

### Phase 4: MuSiQue replication

1. Run the exact oracle hop-by-paragraph matrix first.
2. Replace oracle entries with learned scores.
3. Stratify by 2, 3, and 4 hops.
4. Measure support selection and downstream answer F1.

MuSiQue offers many more instances than manually assessed TREC RAG and can show whether gains persist across thousands of small candidate sets.

### Phase 5: scale only after a positive result

Use HoVer for 2--4-hop verification, ASQA for ambiguous-question coverage, and Structured3D/SU3 for geometric scale. QAMPARI and full-corpus TREC retrieval should wait because their representation and infrastructure costs make negative results difficult to diagnose.

## Claims the resulting experiments can support

York Urban can support a direct claim:

> Isometry Pursuit selects unoriented, approximately orthogonal directions from noisy real candidate sets.

TREC RAG and MuSiQue can support a conditional claim:

> Given a relevant candidate pool and a low-dimensional nonnegative passage-to-facet representation, Isometry Pursuit selects a well-conditioned evidence set with broad facet coverage.

They cannot by themselves establish that raw semantic embedding diversity is projective, that the method jointly optimizes relevance and diversity, or that it improves full-corpus retrieval. Those require different objectives or additional stages.

