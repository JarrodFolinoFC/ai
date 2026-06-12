import katex from 'katex';
import { useEffect, useRef } from 'react';

import { color, space, radius, font } from '../theme';
import { FORMULAS } from '../formulas';

function Math({ tex, block = false }: { tex: string; block?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    katex.render(tex, ref.current, { displayMode: block, throwOnError: false });
  }, [tex, block]);
  return (
    <span
      ref={ref}
      style={
        block
          ? { display: 'block', margin: '0.75rem 0', textAlign: 'left' }
          : undefined
      }
    />
  );
}

const SECTION: React.CSSProperties = {
  maxWidth: '70ch',
  lineHeight: 1.6,
  color: color.text.primary,
};
const PROSE: React.CSSProperties = { ...SECTION, marginBottom: space.lg };
const NOTE: React.CSSProperties = {
  ...SECTION,
  background: color.warning.bg,
  border: `1px solid ${color.highlightBg}`,
  padding: '0.75rem 1rem',
  borderRadius: radius.sm,
  marginBottom: space.lg,
};
const CODE: React.CSSProperties = {
  fontFamily: 'monospace',
  fontSize: font.size.sm,
};
const PRE: React.CSSProperties = {
  ...CODE,
  background: color.text.primary,
  color: color.text.onDark,
  padding: '0.75rem 1rem',
  borderRadius: radius.sm,
  overflowX: 'auto',
  maxWidth: '70ch',
  marginBottom: space.lg,
};

export function AttentionConceptsReferencePage() {
  return (
    <div>
      <h1 style={{ fontSize: '1.4rem' }}>
        Attention Concepts Reference (Stage 2 deep dive)
      </h1>
      <p style={PROSE}>
        Self-contained reference for every component on the way to a working
        attention block. Read top-to-bottom; each section assumes the previous
        ones. Symbols are reused consistently:{' '}
        <code style={CODE}>T</code> = sequence length,{' '}
        <code style={CODE}>V</code> = vocabulary size,{' '}
        <code style={CODE}>d</code> = <code style={CODE}>d_model</code> =
        embedding dimension, <code style={CODE}>t</code> = query position,{' '}
        <code style={CODE}>j</code> = key position.
      </p>

      <h2>Table of contents</h2>
      <ol style={PROSE}>
        <li>Lower-Tri Mask</li>
        <li>Normalize Rows</li>
        <li>Unembed Head</li>
        <li>Stage 2c-i overview (hand-weighted attention)</li>
        <li>Causal Score Mask</li>
        <li>Attention-Weighted Sum</li>
        <li>Stage 2c-i Flow (everything chained)</li>
        <li>How these pieces fit into a real transformer</li>
      </ol>

      {/* ============================= 1. Lower-Tri Mask ============================= */}
      <h2 id="lower-tri-mask">1. Lower-Tri Mask</h2>
      <Math
        block
        tex={FORMULAS.attnConcepts01}
      />
      <p style={PROSE}>
        <strong>What it is.</strong> A <code style={CODE}>T × T</code>{' '}
        boolean matrix whose lower triangle (including the diagonal) is 1 and
        whose strictly upper triangle is 0. It is the simplest way to encode
        the rule <em>"position i may attend to position j only if j ≤ i."</em>
      </p>
      <p style={PROSE}>
        <strong>Worked example, T = 4.</strong>
      </p>
      <pre style={PRE}>
{`     j=0  j=1  j=2  j=3
i=0   1    0    0    0      ← row 0 sees only token 0
i=1   1    1    0    0      ← row 1 sees tokens 0..1
i=2   1    1    1    0      ← row 2 sees tokens 0..2
i=3   1    1    1    1      ← row 3 sees the whole prefix`}
      </pre>
      <p style={PROSE}>
        <strong>Why it has this exact shape.</strong> A row is a query, a
        column is a key. We want each query to see itself and every earlier
        key, but never anything later. The diagonal is 1 because a token{' '}
        <em>does</em> see itself; the upper triangle is 0 because that's
        "the future."
      </p>
      <p style={PROSE}>
        <strong>What it isn't.</strong> The mask doesn't tell the model{' '}
        <em>how strongly</em> to attend — only whether attention is{' '}
        <em>allowed</em>. It's a permission matrix, not a weight matrix.
      </p>
      <p style={PROSE}>
        <strong>Two roles it plays.</strong>
      </p>
      <ol style={PROSE}>
        <li>
          As the basis for <em>uniform-over-prefix</em> attention (Stage 2b):
          you row-normalize the mask itself and use the result as the
          attention matrix. No parameters, no softmax — just "average my
          past."
        </li>
        <li>
          As the source of the <em>causal score mask</em> (Stage 2c onward):
          wherever <code style={CODE}>M = 0</code>, you replace the raw
          attention score with <code style={CODE}>−∞</code> before softmax,
          so future positions get exactly zero weight.
        </li>
      </ol>
      <div style={NOTE}>
        <strong>Common confusion.</strong> The lower-tri mask isn't itself
        the attention. It's a <em>shape constraint</em> applied to whatever
        attention mechanism you choose. Bidirectional models (BERT) skip
        this entirely — every position attends everywhere.
      </div>

      {/* ============================= 2. Normalize Rows ============================= */}
      <h2 id="normalize-rows">2. Normalize Rows</h2>
      <Math
        block
        tex={FORMULAS.attnConcepts02}
      />
      <p style={PROSE}>
        <strong>What it is.</strong> Take any non-negative matrix (typically
        the lower-tri mask) and divide each row by its own sum, so every row
        sums to 1. The output is called <em>row-stochastic</em>: each row is
        a valid probability distribution over columns.
      </p>
      <p style={PROSE}>
        <strong>Worked example.</strong> Apply row-normalization to the
        T = 4 lower-tri mask above:
      </p>
      <pre style={PRE}>
{`     j=0     j=1     j=2     j=3      sum
i=0  1.0000  0       0       0        1.00
i=1  0.5000  0.5000  0       0        1.00
i=2  0.3333  0.3333  0.3333  0        1.00
i=3  0.2500  0.2500  0.2500  0.2500   1.00`}
      </pre>
      <p style={PROSE}>
        <strong>Read this matrix as: "how much does query i pay attention
        to key j?"</strong> Row 3 attends uniformly to all four positions;
        row 0 has no choice but to attend entirely to itself.
      </p>
      <p style={PROSE}>
        <strong>What this gives us.</strong> The simplest possible attention
        mechanism — "the new hidden vector at position t is the average of
        all hidden vectors at positions ≤ t." This is Stage 2b. It has
        zero parameters and no softmax, but already produces context-mixing.
      </p>
      <p style={PROSE}>
        <strong>Geometric intuition.</strong> Multiplying H (T rows of
        d-dim vectors) on the left by this row-stochastic W produces a new
        T × d matrix where row t is a convex combination of rows 0..t. Each
        new row lives <em>inside</em> the polytope spanned by the original
        rows — no new directions are invented, only blended.
      </p>
      <div style={NOTE}>
        <strong>Why softmax is "better than" plain row-normalization.</strong>{' '}
        Plain row-normalize gives uniform weights — every past token gets
        equal say. Softmax over learned scores lets the model{' '}
        <em>concentrate</em> attention: a single past token can dominate the
        sum if its score is much larger than the rest. That's the only
        difference, but it's huge — uniform mixing can't disambiguate "he"
        from "she" based on which earlier word the pronoun co-refers with.
      </div>

      {/* ============================= 3. Unembed Head ============================= */}
      <h2 id="unembed-head">3. Unembed Head</h2>
      <Math
        block
        tex={FORMULAS.attnConcepts03}
      />
      <p style={PROSE}>
        <strong>What it is.</strong> A linear layer that maps the model's
        internal hidden vector <code style={CODE}>h ∈ ℝ^d</code> back into
        vocabulary-sized "logits" <code style={CODE}>ℝ^V</code>. The matrix{' '}
        <code style={CODE}>W_head</code> has shape{' '}
        <code style={CODE}>d × V</code>, the bias{' '}
        <code style={CODE}>b ∈ ℝ^V</code>.
      </p>
      <p style={PROSE}>
        <strong>Why this exists.</strong> Everything inside the model
        operates in the <code style={CODE}>d</code>-dimensional embedding
        space — embeddings, attention output, FFN output. The model's job at
        the end is "predict the next token," which means producing one score
        per vocabulary entry. Some operation has to convert from
        d-dimensional internal space to V-dimensional vocab space. Unembed
        is that operation.
      </p>
      <p style={PROSE}>
        <strong>How a single logit is computed.</strong> Each column of{' '}
        <code style={CODE}>W_head</code> is a vector of length{' '}
        <code style={CODE}>d</code> — call it{' '}
        <code style={CODE}>w_v</code>, the "key" for vocab token v.
      </p>
      <Math
        block
        tex={FORMULAS.attnConcepts04}
      />
      <p style={PROSE}>
        Read: "the logit for vocab token v is the dot product of the
        hidden vector with v's column, plus a per-token bias." Tokens whose
        column points in the same direction as <code style={CODE}>h</code>{' '}
        get high logits. This is just cosine-similarity-shaped, with{' '}
        <code style={CODE}>W_head</code> learning a "preferred direction"
        per vocab entry.
      </p>
      <p style={PROSE}>
        <strong>Why the bias exists.</strong> Bias{' '}
        <code style={CODE}>b_v</code> learns a base rate for each token. If
        the word "the" appears in 5% of all positions, its bias drifts
        upward during training so that even a generic <code style={CODE}>h</code>{' '}
        produces a moderately high logit for "the." Without the bias, the
        model would have to encode token frequency through{' '}
        <code style={CODE}>W_head</code> alone, which is wasteful.
      </p>
      <p style={PROSE}>
        <strong>Tied vs untied embeddings.</strong> A common trick (used in
        GPT-2) is to tie the unembed weights to the input embedding matrix:{' '}
        <code style={CODE}>W_head = Eᵀ</code>. This is the same{' '}
        <code style={CODE}>V × d</code> table read in either direction —
        once for token-id → vector, once for vector → token-score. It cuts
        the parameter count by <code style={CODE}>V·d</code> and tends to
        improve sample efficiency. Stage 2 in this project keeps them
        separate so you can see the symmetry breaking.
      </p>
      <p style={PROSE}>
        <strong>Followed by softmax.</strong> Logits are{' '}
        <em>not</em> probabilities. They're unnormalized scores. The very
        next step is softmax over the vocab dimension to produce{' '}
        <code style={CODE}>P(next_token | h) ∈ Δ^V</code>.
      </p>

      {/* ============================= 4. Stage 2c-i overview ============================= */}
      <h2 id="stage-2c-i">4. Stage 2c-i overview (hand-weighted attention)</h2>
      <p style={PROSE}>
        <strong>The point of this stage.</strong> Until Stage 2a, every
        token's hidden vector depends only on itself: <code style={CODE}>
          h_t = E[idx_t] + P[t]
        </code>
        . That means the model has no access to context — it can't learn
        "if 'the cat sat on the' appeared, predict 'mat' or 'sofa'" because{' '}
        <code style={CODE}>h_t</code> at position t literally doesn't know
        what was at positions 0..t−1.
      </p>
      <p style={PROSE}>
        Attention is the operation that fixes this. It builds a new hidden
        vector <code style={CODE}>h'_t</code> at every position by
        <em>blending in</em> information from earlier positions. The weight
        each earlier position gets is called the{' '}
        <em>attention weight</em>.
      </p>
      <p style={PROSE}>
        <strong>Why split 2c into two sub-stages.</strong> Real attention
        (Q/K/V) bundles two distinct ideas:
      </p>
      <ol style={PROSE}>
        <li>
          The <strong>mechanism</strong>: causal mask, softmax, weighted
          sum. This is just plumbing.
        </li>
        <li>
          The <strong>score function</strong>: how the attention weights are
          computed in the first place. This is the genuinely new and
          parameter-laden part of attention.
        </li>
      </ol>
      <p style={PROSE}>
        Stage 2c-i isolates (1) by letting <em>you</em> hand-pick the
        attention scores. The mechanism on either side of those scores is
        identical to what 2c-ii will use; the only swap is replacing your
        hand-typed numbers with{' '}
        <code style={CODE}>q_t · k_j / √d</code>. So if you understand 2c-i
        end to end, you understand roughly 80% of attention.
      </p>
      <p style={PROSE}>
        <strong>Pages in this stage.</strong>
      </p>
      <ul style={PROSE}>
        <li>
          <strong>Causal Score Mask</strong> — the −∞ masking step.
        </li>
        <li>
          <strong>Attention-Weighted Sum</strong> — the weighted blend.
        </li>
        <li>
          <strong>Stage 2c-i Flow</strong> — all the steps chained on a real
          corpus.
        </li>
      </ul>

      {/* ============================= 5. Causal Score Mask ============================= */}
      <h2 id="causal-score-mask">5. Causal Score Mask</h2>
      <Math
        block
        tex={FORMULAS.attnConcepts05}
      />
      <p style={PROSE}>
        <strong>What it is.</strong> Take a row of raw attention scores{' '}
        <code style={CODE}>s_0, s_1, ..., s_{`{T-1}`}</code> for a single
        query position <code style={CODE}>t</code>, and replace every entry
        whose index is strictly greater than <code style={CODE}>t</code>{' '}
        with <code style={CODE}>−∞</code>.
      </p>
      <p style={PROSE}>
        <strong>Worked example.</strong> T = 6, t = 3, raw scores{' '}
        <code style={CODE}>s = [0.5, -0.2, 1.0, 0.0, 0.3, -1.0]</code>:
      </p>
      <pre style={PRE}>
{`j           0     1     2     3     4     5
s_j        0.5  -0.2   1.0   0.0   0.3  -1.0    ← raw
s'_j       0.5  -0.2   1.0   0.0  -inf  -inf    ← after mask
                                  ^^^^  ^^^^
                                  future, killed`}
      </pre>
      <p style={PROSE}>
        <strong>Why −∞ specifically, not 0.</strong> The next step is{' '}
        softmax: <code style={CODE}>a_j = exp(s'_j) / Σ exp(s'_k)</code>.
      </p>
      <ul style={PROSE}>
        <li>
          <code style={CODE}>exp(−∞) = 0</code> exactly, so masked entries
          contribute zero to the numerator and zero to the denominator —
          they vanish from the distribution.
        </li>
        <li>
          <code style={CODE}>exp(0) = 1</code>, which is{' '}
          <em>not</em> small. If you had set masked entries to 0 instead of
          −∞, they'd still get nontrivial weight after softmax.
        </li>
      </ul>
      <p style={PROSE}>
        Mask <em>before</em> softmax — never after. After softmax, a
        zero-weight entry has already had its mass redistributed and you
        can't undo it.
      </p>
      <p style={PROSE}>
        <strong>Why this matters at training time.</strong> Language models
        are trained by predicting the next token from the prefix. The loss
        compares <code style={CODE}>argmax(softmax(logits_t))</code> to{' '}
        <code style={CODE}>token_{`{t+1}`}</code>. If the query at position{' '}
        <code style={CODE}>t</code> could see token{' '}
        <code style={CODE}>t+1</code>, the prediction task collapses to
        "copy the token you can already see" — the model would learn
        nothing useful. The mask <em>defines the supervision signal</em> by
        forcing the model to predict the future from the past only.
      </p>
      <p style={PROSE}>
        <strong>What it does to the corpus.</strong> Nothing. The corpus
        text is unchanged. The mask only edits the score vector that the
        model uses internally for one query at one moment in time. At
        inference, you slide <code style={CODE}>t</code> from 0 to{' '}
        <code style={CODE}>T-1</code> and apply a different mask each time.
      </p>
      <div style={NOTE}>
        <strong>Implementation note.</strong> In a batched implementation
        every query in the batch shares the same lower-tri mask matrix
        (because position t can always see 0..t regardless of which sequence
        we're in). So you precompute one <code style={CODE}>T × T</code>{' '}
        mask once, broadcast it across every batch and head, and add{' '}
        <code style={CODE}>(1 − M) · −∞</code> (or use a torch
        masked_fill) to the score matrix before softmax.
      </div>

      {/* ============================= 6. Attention-Weighted Sum ============================= */}
      <h2 id="attention-weighted-sum">6. Attention-Weighted Sum</h2>
      <Math
        block
        tex={FORMULAS.attnConcepts06}
      />
      <p style={PROSE}>
        <strong>What it is.</strong> Multiply each hidden vector{' '}
        <code style={CODE}>h_j</code> by its scalar attention weight{' '}
        <code style={CODE}>a_j</code>, and sum the resulting vectors
        element-wise. The output is one new vector in the same{' '}
        <code style={CODE}>d</code>-dimensional space as the inputs.
      </p>
      <p style={PROSE}>
        <strong>Worked example.</strong> T = 3, d = 2, weights{' '}
        <code style={CODE}>a = [0.1, 0.6, 0.3]</code>:
      </p>
      <pre style={PRE}>
{`H = [
  [1, 2],     ← h_0
  [3, 4],     ← h_1
  [5, 6],     ← h_2
]

a_0 · h_0 = 0.1 · [1, 2] = [0.1, 0.2]
a_1 · h_1 = 0.6 · [3, 4] = [1.8, 2.4]
a_2 · h_2 = 0.3 · [5, 6] = [1.5, 1.8]
                            ─────────
sum                       = [3.4, 4.4]   ← h'_t`}
      </pre>
      <p style={PROSE}>
        <strong>Convex combination.</strong> When the weights sum to 1 and
        are non-negative (which they are after softmax), the output is a{' '}
        <em>convex combination</em> of the inputs. Geometrically this means{' '}
        <code style={CODE}>h'_t</code> lies inside the polytope spanned by{' '}
        <code style={CODE}>h_0, ..., h_{`{T-1}`}</code>. No new directions
        are invented; the output is literally a blend.
      </p>
      <p style={PROSE}>
        <strong>Information flow.</strong> This is the only operation in
        attention that moves information across positions. Everything else
        — the score function, the mask, softmax — is about computing the
        weights. The weighted sum is the part that actually does the mixing.
        Once <code style={CODE}>h'_t</code> is computed, it knows about the
        prefix in a way <code style={CODE}>h_t</code> didn't.
      </p>
      <p style={PROSE}>
        <strong>Output dimension.</strong> Same as input dimension. Every{' '}
        <code style={CODE}>h_j</code> lives in{' '}
        <code style={CODE}>ℝ^d</code>; their weighted sum also lives in{' '}
        <code style={CODE}>ℝ^d</code>. Attention is not a reshape — it's a
        re-mix at the same dim. The next layer (FFN, or another attention
        block) sees a vector in the same shape it would otherwise expect.
      </p>
      <p style={PROSE}>
        <strong>Why the same dim matters.</strong> Transformer blocks are
        stacked by adding residual connections:{' '}
        <code style={CODE}>x ← x + Attention(x)</code>. Both sides need to
        be the same shape. If attention reshaped, residuals wouldn't add.
        This is one of several conventions that constrains the entire
        architecture.
      </p>
      <div style={NOTE}>
        <strong>The hidden simplicity.</strong> The "weighted sum" is just
        a matrix multiply: <code style={CODE}>h' = aᵀ · H</code> where{' '}
        <code style={CODE}>a</code> is a 1×T vector and{' '}
        <code style={CODE}>H</code> is a T×d matrix. In code it's literally
        one line. The conceptual difficulty isn't the math — it's accepting
        that this innocuous matmul is "attention."
      </div>

      {/* ============================= 7. Stage 2c-i Flow ============================= */}
      <h2 id="stage-2c-i-flow">7. Stage 2c-i Flow (everything chained)</h2>
      <p style={PROSE}>
        Pipeline for one query position <code style={CODE}>t</code>:
      </p>
      <Math
        block
        tex={FORMULAS.attnConcepts07}
      />
      <p style={PROSE}>
        <strong>Read it like a sentence.</strong> "Embed every token plus
        its position. Pick how much each earlier position should
        contribute. Block out the future. Turn the scores into weights that
        sum to 1. Mix the hidden vectors using those weights. The result is
        the context-aware hidden vector at position t."
      </p>
      <p style={PROSE}>
        <strong>What changed compared to Stage 2a.</strong> Stage 2a stops
        after step 1: <code style={CODE}>h_t</code> is the model's "view of
        position t" and it depends only on the token at t and t's location
        in the sentence. Stage 2c-i replaces that with{' '}
        <code style={CODE}>h'_t</code>, which is the same shape but now
        knows about positions 0..t.
      </p>
      <p style={PROSE}>
        <strong>What the unembed head sees.</strong> After attention, the
        full pipeline pushes <code style={CODE}>h'_t</code> through the
        unembed head and softmax to produce a probability distribution over
        the vocab — i.e. the model's prediction for token{' '}
        <code style={CODE}>t+1</code>. With Stage 2a's{' '}
        <code style={CODE}>h_t</code> the prediction can only depend on the
        single current token; with Stage 2c-i's{' '}
        <code style={CODE}>h'_t</code> it can depend on the entire prefix.
        That's the qualitative jump.
      </p>
      <p style={PROSE}>
        <strong>Things to verify by playing with the visualizer.</strong>
      </p>
      <ul style={PROSE}>
        <li>
          Set the score at <code style={CODE}>j = t</code> to a very large
          number (say 100). Softmax collapses to a one-hot at self →{' '}
          <code style={CODE}>h'_t ≈ h_t</code>. That's "no mixing." A
          self-attending token is equivalent to no attention at all.
        </li>
        <li>
          Set <code style={CODE}>t = 0</code>. Only one position is
          unmasked, so softmax is forced to put 1.0 there. No mixing
          possible. The first token of any sequence cannot benefit from
          attention.
        </li>
        <li>
          Make the scores roughly uniform (all zeros). Softmax produces{' '}
          <code style={CODE}>1/(t+1)</code> uniform weights — equivalent to
          row-normalize of the lower-tri mask, i.e. Stage 2b. So Stage 2c-i
          with uniform scores reduces to Stage 2b.
        </li>
        <li>
          Concentrate all the weight on a single past position (say{' '}
          <code style={CODE}>j = 1</code>). Then{' '}
          <code style={CODE}>h'_t ≈ h_1</code> — the query takes on the
          identity of that earlier token. This is the "copy" pattern.
          Trained heads often do exactly this for tasks like name
          coreference.
        </li>
      </ul>

      {/* ============================= 8. Putting it together ============================= */}
      <h2 id="putting-it-together">
        8. How these pieces fit into a real transformer
      </h2>
      <p style={PROSE}>
        Stage 2c-i is missing two things that 2c-ii and beyond add:
      </p>
      <ol style={PROSE}>
        <li>
          <strong>Where the scores come from.</strong> In real attention,
          you don't hand-pick scores. You compute them as
          <Math
            block
            tex={FORMULAS.attnConcepts08}
          />
          The <code style={CODE}>W_Q</code> and{' '}
          <code style={CODE}>W_K</code> matrices are learned. Now scores
          come from the data, not your fingers. Everything else (mask,
          softmax, weighted sum) is unchanged.
        </li>
        <li>
          <strong>What gets mixed.</strong> Hand-weighted attention sums{' '}
          <code style={CODE}>h_j</code> directly. Real attention sums{' '}
          <code style={CODE}>v_j = h_j W_V</code> instead — a
          learned <em>view</em> of each token tuned for what to pass on.
          So <code style={CODE}>h'_t = Σ a_j · v_j</code>.
        </li>
      </ol>
      <p style={PROSE}>
        <strong>One row of the attention matrix at a time.</strong> The
        Stage 2c-i flow page processes one query position. The full
        attention block does this for <em>every</em>{' '}
        <code style={CODE}>t</code> from 0 to{' '}
        <code style={CODE}>T-1</code> in parallel. The scores become a{' '}
        <code style={CODE}>T × T</code> matrix; softmax is row-wise; the
        weighted sum becomes a single matmul{' '}
        <code style={CODE}>A · V</code>. That's why the lower-tri mask is a{' '}
        <code style={CODE}>T × T</code> object: it constrains every row of
        the score matrix at once.
      </p>
      <p style={PROSE}>
        <strong>Stages still ahead.</strong>
      </p>
      <ul style={PROSE}>
        <li>
          <strong>2c-ii:</strong> introduce Q, K, V projections and the
          √d_k scaling.
        </li>
        <li>
          <strong>2d (multi-head):</strong> run several attention heads in
          parallel, each with its own Q/K/V, then concatenate. Lets the
          model attend "different ways at once" — one head for syntax, one
          for coreference, etc.
        </li>
        <li>
          <strong>3 (FFN):</strong> a 2-layer MLP applied independently per
          position. The non-linearity that lets the model do more than
          blend.
        </li>
        <li>
          <strong>4 (residuals + LayerNorm):</strong> the connective tissue
          that makes deep transformer stacks trainable.
        </li>
        <li>
          <strong>5+ (scale up):</strong> stack the block ~12 times
          (GPT-2 small) or ~96 times (GPT-3), bump{' '}
          <code style={CODE}>d_model</code>, train on a real corpus. The
          architecture stays the same.
        </li>
      </ul>
      <div style={NOTE}>
        <strong>Synthesis.</strong> A transformer is "embedding + attention
        + FFN + residual + LayerNorm" stacked many times, with a final
        unembed head. You now understand the embedding lookup (Stage 2a),
        the position table (sinusoidal P), the lower-tri mask, the unembed
        head, and the entire forward path of attention except the score
        function. The score function is the next thing to learn — and the
        only conceptual hurdle left between you and a working GPT-2.
      </div>
    </div>
  );
}
