---
name: isometry-pursuit-style
description: Write or revise Isometry Pursuit manuscript prose and proposed paper sentences with concrete insight, grounded terminology, and direct attribution, especially when discussing related work.
---

# Isometry Pursuit Writing Style

Every sentence should advance the argument or explain a connection the reader needs. A citation alone is not a reason to add a sentence.

Always use active voice in manuscript prose. Name who or what performs the action; for our constructions and experiments, use “we.” For example, write “We augment a planted identity block with randomly oriented orthonormal blocks,” not “A planted identity block is augmented with randomly oriented orthonormal blocks.”

Write each prose sentence on its own physical line in the LaTeX source. Do not hard-wrap sentences at a fixed column width. Preserve blank lines between paragraphs and the structure of display equations, tables, and algorithms. Source-formatting changes must preserve the rendered paper.

Always use round parentheses `()` for parenthetical grouping and function arguments in prose and mathematics; do not substitute square brackets `[]`. For example, write `\exp(-a(t))`, not `\exp[-a(t)]`. Square brackets retain their distinct mathematical uses, such as indexing and closed intervals.

Keep related-work additions proportional to their role. A citation supporting an already-defined algorithm often needs only a citation or a few words, not a new paragraph. Do not turn a request to add a source into a survey of its methods, qualifications, and guarantees. Use Discussion for optional extensions; keep the algorithm and experiment descriptions centered on what we actually run. Include additional explanation only when it changes how the reader should understand our method or claims. Concise treatment can still cite several relevant sources when the user wants them.

Ground comparisons in concepts already introduced in the surrounding text. State the specific shared mechanism, meaningful difference, or consequence for our method. Do not introduce a list of unfamiliar algorithms or problem settings merely to acknowledge a paper. If explaining the connection requires a distracting detour, omit the sentence or keep a source comment instead, as the task warrants.

Prefer active, direct attribution when describing prior work: name who does what. Avoid passive statements such as “have also been studied” that report the existence of literature without explaining why it matters. Simply changing passive voice to active voice does not fix an empty comparison.

## Example to avoid

> Backward elimination and alternating addition–deletion procedures have also been studied for least-squares subset selection.

This sentence gives the reader no useful insight, introduces several topics without grounding them in the surrounding argument, and hides the contribution behind passive voice. Do not use it or an equivalent generic literature sentence. Before proposing a replacement, identify the exact point the citation supports and explain it using the manuscript's established terms. If there is no useful point to make, leave it out.

Use at most one relational sign per line of displayed mathematics.
Split chains of equalities or inequalities into aligned lines, with one relation on each line; do not place separate relational statements side by side on the same line.

Before a mathematical derivation, write a short declarative sentence stating what the calculation establishes.
For longer proofs, briefly outline the key steps at the beginning; introduce each subsequent calculation by its purpose rather than leaving the reader to infer the argument from algebra.
When quoting a property before proving it, put the quoted statement in its own paragraph, separated from the proof text.

Keep established mathematical expressions explicit rather than introducing temporary names merely to shorten them.
For example, continue writing $n_q^{\times D}(X)$ instead of declaring $N=n_q^{\times D}(X)$ and making the reader translate back.
Introduce new notation only when it represents a distinct mathematical construction needed for the argument, rather than an abbreviation for an expression already defined.

Keep prose concise, but do not confuse a short sentence with an informative one. Follow the user's requested scope; this style guidance does not authorize unrelated rewrites.

## Positive example: local distortion proof

The user approved the following proof style for Proposition
`prop:distortion_is_condition_number` in `archival/supplement.tex`:

> For distinct $\xi',\xi''\in B_{\mathcal M}(\xi,\varepsilon)$, let $u',u''$
> be their local coordinates, chosen so that the coordinate tangent basis at
> $\xi$ is orthonormal. Smoothness of $\mathcal M$ gives
> $d_{\mathcal M}(\xi',\xi'')=\|u'-u''\|_2(1+o(1))$, while continuous
> differentiability of $f_{.S}$ gives
> $f_{.S}(\xi')-f_{.S}(\xi'')=Df_{.S}(\xi)(u'-u'')+o(\|u'-u''\|_2)$,
> uniformly over pairs as $\varepsilon\to0$. Dividing by their separation
> therefore gives
>
> $$r_S(\xi',\xi'')=
> \frac{\|Df_{.S}(\xi)(u'-u'')\|_2}{\|u'-u''\|_2}+o(1).$$
>
> Thus $\sup r_S\to\sigma_1(Df_{.S}(\xi))$ and
> $\inf r_S\to\sigma_D(Df_{.S}(\xi))$. When the differential is invertible,
> taking their quotient gives
> $\operatorname{Distortion}(f_{.S},B_{\mathcal M}(\xi,\varepsilon))
> \to\sigma_1(Df_{.S}(\xi))/\sigma_D(Df_{.S}(\xi))$.

The neighborhood distortion and $r_S$ are defined immediately before this
proof. Follow this example by defining necessary notation, stating the actual
approximation, displaying the decisive algebraic step, and drawing the
conclusion directly. Preserve essential hypotheses and uniform or relative
error statements. Avoid replacing the algebra with vague claims such as
“distance ratios approach those of the differential.” Do not introduce
unexplained machinery such as the exponential map or add commentary that
merely narrates a standard consequence of the displayed formula. This is a
style example, not a requirement to compress every proof into one paragraph.

For the accompanying Table 1 edit, the user preferred the compact
condition-number expression in the existing loss column, with its
infinitesimal-limit qualification and a reference to the supplementary proof.
Keep tables focused on comparable expressions; move necessary derivations to
the supplement rather than crowding cells or replacing the user's column
structure.

## Do not assume the reader knows the cited paper

The reader has the manuscript's preceding text, not our chat history or the contents of every cited source. A citation does not introduce or explain that source's objective, sampling rule, notation, or algorithm. Do not refer to those details as if the reader already knows they exist. This applies to technical qualifications and comparisons as well as standalone sentences. An accurate caveat can still be bad prose if understanding it requires unexplained machinery.

Examples rejected by the user:

> Sampling incoming columns could reduce the cost of swap search, although our isometry loss requires a different sampling rule from their reconstruction-based criterion.

Delete the clause beginning with “although.” The useful point is that sampling can reduce swap-search cost. The qualification introduces another objective and sampling rule without explaining either. The accepted sentence is “Sampling incoming columns could reduce the cost of swap search,” with the citation.

> Wang et al. use swap search for D-optimal design, while Zou et al. sample incoming columns to reduce the number of swaps evaluated for reconstruction-based column selection.

Delete the clause beginning with “while Zou et al.” in Experiments. It pulls the reader away from the just-defined swap algorithm into an unexplained problem and variant. Keep the optional sampling idea in Discussion, expressed using the already-established swap-search vocabulary.

Before adding a comparison or caveat, check what each referenced concept means to a reader at that exact point in the manuscript. If the reader would need to ask “what criterion?” or “what rule?”, either explain the concept when it is essential to the argument or omit the reference to it. Do not add a detour merely to preserve a qualification.
