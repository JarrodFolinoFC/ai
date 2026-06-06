import katex from 'katex';
import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { color, space, radius, font } from '../theme';
import { FORMULAS } from '../formulas';

function Tex({
  latex,
  block = false,
}: {
  latex: string;
  block?: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    katex.render(latex, ref.current, {
      displayMode: block,
      throwOnError: false,
    });
  }, [latex, block]);
  return (
    <span
      ref={ref}
      style={{
        display: block ? 'block' : 'inline-block',
        margin: block ? '0.4rem 0' : 0,
      }}
    />
  );
}

interface Question {
  q: string;
  latex?: string; // optional display-mode formula shown under the question
  options: (string | { text: string; latex?: string })[];
  correct: number;
  explanation: string;
  explanationLatex?: string;
}

const QUESTIONS: Question[] = [
  {
    q: 'In Stage 2a, what does the embedding dimension n_embd represent?',
    options: [
      'The number of tokens in the vocabulary',
      'The maximum sequence length',
      'The width of each embedding vector (E and P rows)',
      'The number of attention heads',
      'The seed used for the random init',
    ],
    correct: 2,
    explanation:
      'n_embd is the chosen width of every embedding row. Both E and P have n_embd columns so that h = E[idx] + P[t] is element-wise.',
  },
  {
    q: 'What is the shape of the token embedding table E?',
    options: [
      'n_embd × n_embd',
      'V × n_embd',
      'T_max × n_embd',
      'n_embd × V',
      'V × V',
    ],
    correct: 1,
    explanation:
      'E has one row per vocab token (V rows), each n_embd wide.',
  },
  {
    q: 'What is the shape of the position embedding table P?',
    options: [
      'V × n_embd',
      'T_max × T_max',
      'T_max × n_embd',
      'n_embd × T_max',
      'V × T_max',
    ],
    correct: 2,
    explanation:
      'P has one row per position slot (T_max rows), each n_embd wide — same width as E so the two can be added.',
  },
  {
    q: 'Why must P have the same width as E?',
    latex: FORMULAS.combinedInput01,
    options: [
      'It is a matrix multiplication requirement',
      'They are concatenated, so the columns line up',
      {
        text: 'The sum is element-wise, so both rows must have identical width',
        latex: FORMULAS.stage2Quiz01,
      },
      'It minimizes the parameter count',
      'It is required by the softmax',
    ],
    correct: 2,
    explanation:
      'Element-wise add forces matching widths. Both E and P rows are length n_embd.',
  },
  {
    q: 'Which dimension of W_head is "forced" rather than chosen?',
    options: [
      'Both — input must match h, output must match the vocab',
      'Neither — both are free hyperparameters',
      'Only the input side, output is chosen',
      'Only the output side, input is chosen',
      'Only when bias b is used',
    ],
    correct: 0,
    explanation:
      'W_head: input width must equal n_embd (to match h), output width must equal V (to produce vocab logits). No design choice.',
  },
  {
    q: 'What is the shape of W_head in Stage 2a?',
    options: [
      'V × V',
      'V × n_embd',
      'n_embd × V',
      'n_embd × n_embd',
      'T_max × V',
    ],
    correct: 2,
    explanation:
      'W_head maps an n_embd-dim hidden vector to V-dim logits.',
  },
  {
    q: 'What is the total parameter count of Stage 2a?',
    options: [
      { text: 'E + P only', latex: FORMULAS.stage2Quiz02 },
      { text: 'E + W_head only', latex: FORMULAS.stage2Quiz03 },
      {
        text: 'E + P + W_head + b',
        latex: FORMULAS.stage2Quiz04,
      },
      { text: 'A single product', latex: FORMULAS.stage2Quiz05 },
      { text: 'A simple sum', latex: FORMULAS.stage2Quiz06 },
    ],
    correct: 2,
    explanation:
      'E (V·n_embd) + P (T_max·n_embd) + W_head (n_embd·V) + b (V).',
    explanationLatex: FORMULAS.stage2Quiz07,
  },
  {
    q: 'Looking up E[idx_t] is mathematically equivalent to what operation?',
    options: [
      'A softmax over E rows',
      'A row index — pure lookup, no math',
      'A dot product with a sinusoidal vector',
      'Matrix multiplication of E with a positional weight',
      'Element-wise multiplication of E with P',
    ],
    correct: 1,
    explanation:
      'idx_t selects one row of E. No arithmetic — just indexing.',
  },
  {
    q: 'In the Stage 2 page, the top section initializes P as:',
    options: [
      'Sinusoidal: alternating sin/cos of t / 10000^(2k/d)',
      'A learnable nn.Embedding',
      'A seeded random matrix scaled to roughly [−0.5, 0.5)',
      'All zeros',
      'A one-hot matrix',
    ],
    correct: 2,
    explanation:
      'The single-position section uses (rand() − 0.5) · 2 · 0.5, a uniform draw in [−0.5, 0.5). The corpus pipeline below uses sinusoidal P.',
  },
  {
    q: 'Why is the seed for P offset (e.g. seed + 17) from the seed for E?',
    options: [
      'It is a security best practice',
      'To decorrelate the two PRNG streams so P and E are independent',
      'To make P sinusoidal',
      'It speeds up the random draws',
      'It has no effect, just convention',
    ],
    correct: 1,
    explanation:
      'Reusing the same seed for both would give correlated draws between E and P. The +17 offset starts a fresh stream.',
  },
  {
    q: 'For the corpus pipeline, what is the shape of the lower-tri mask M?',
    options: [
      'V × V',
      'T × T (T = corpus length)',
      'T × n_embd',
      'n_embd × n_embd',
      'V × T',
    ],
    correct: 1,
    explanation:
      'M is a square T×T matrix encoding "position i can see positions j ≤ i" for every (i, j) pair.',
  },
  {
    q: 'What is M[i, j] equal to?',
    options: [
      { text: 'Identity', latex: FORMULAS.stage2Quiz08 },
      {
        text: 'Lower-tri',
        latex: FORMULAS.lowerTriMask01,
      },
      {
        text: 'Pre-normalized lower-tri',
        latex: FORMULAS.stage2Quiz09,
      },
      {
        text: 'Upper-tri',
        latex: FORMULAS.stage2Quiz10,
      },
      { text: 'Random', latex: FORMULAS.stage2Quiz11 },
    ],
    correct: 1,
    explanation:
      'Lower-triangular: row i has 1s in columns 0..i and 0s elsewhere.',
  },
  {
    q: 'How is W (the row-normalized mask) constructed from M?',
    options: [
      { text: 'Column softmax', latex: FORMULAS.stage2Quiz12 },
      { text: 'Scale by global max', latex: FORMULAS.stage2Quiz13 },
      {
        text: 'Row-divide by row sum',
        latex: FORMULAS.stage2Quiz14,
      },
      { text: 'Self-product', latex: FORMULAS.stage2Quiz15 },
      { text: 'Fresh random matrix', latex: FORMULAS.stage2Quiz16 },
    ],
    correct: 2,
    explanation:
      'Each row of M is divided by its sum, yielding a row-stochastic weight matrix.',
    explanationLatex: FORMULAS.stage2Quiz17,
  },
  {
    q: 'For the lower-tri mask, what does row 3 of W look like (with T ≥ 4)?',
    options: [
      '[1, 0, 0, 0, 0, ...]',
      '[0.25, 0.25, 0.25, 0.25, 0, ...]',
      '[0.33, 0.33, 0.33, 0, 0, ...]',
      '[0, 0, 0, 1, 0, ...]',
      '[0.5, 0.5, 0, 0, 0, ...]',
    ],
    correct: 1,
    explanation:
      'Row i = 3 averages 4 positions (0..3) uniformly, so each weight is 1/4 = 0.25.',
  },
  {
    q: "What does H' = W · H compute, in plain words?",
    latex: FORMULAS.stage2Quiz18,
    options: [
      'A learned attention mix',
      {
        text: "Each row H'[t] is the running uniform average of H[0..t]",
        latex: FORMULAS.stage2Quiz19,
      },
      'A softmax over hidden vectors',
      'A causal mask without averaging',
      'A skip connection',
    ],
    correct: 1,
    explanation:
      'Because W is the row-normalized lower-tri mask, H\'[t] equals the mean of H[0..t].',
  },
  {
    q: "For the corpus pipeline, why does H'[0] equal H[0]?",
    latex: FORMULAS.stage2Quiz20,
    options: [
      'It is initialized that way',
      {
        text: 'Position 0 has only itself to average',
        latex: FORMULAS.stage2Quiz21,
      },
      'It is a coincidence of the seed',
      'Because P[0] is zero',
      'Because the mask is upper-triangular',
    ],
    correct: 1,
    explanation:
      'Causality means position 0 can only see position 0; the only nonzero weight is W[0,0] = 1.',
  },
  {
    q: 'In the corpus pipeline the page uses tied weights. What does that mean?',
    options: [
      { text: 'Initialized to zero', latex: FORMULAS.stage2Quiz22 },
      'Shared across positions (always true)',
      {
        text: 'Unembed reuses the embedding rows',
        latex: FORMULAS.stage2Quiz23,
      },
      'W_head and b share gradients',
      { text: 'Tied to position table', latex: FORMULAS.stage2Quiz24 },
    ],
    correct: 2,
    explanation:
      'Tying means the same vectors that embed tokens also project hidden states back to vocab.',
    explanationLatex: FORMULAS.stage2Quiz25,
  },
  {
    q: 'With tied weights, what is logits[t, v] equal to?',
    latex: FORMULAS.stage2Quiz26,
    options: [
      'A sinusoidal function of t and v',
      {
        text: 'A dot product against vocab token v\'s embedding row',
        latex: FORMULAS.stage2Quiz27,
      },
      {
        text: 'Cross-entropy between rows',
        latex: FORMULAS.stage2Quiz28,
      },
      {
        text: 'Cosine angle',
        latex: FORMULAS.stage2Quiz29,
      },
      'A scalar drawn from the PRNG',
    ],
    correct: 1,
    explanation:
      'With W_head = E^T, the matmul computes a dot product against every embedding row.',
  },
  {
    q: 'After softmax on the logits, what does the argmax of row t represent?',
    options: [
      'The current token at position t',
      'The model\'s predicted next token after position t',
      'The most-attended-to past position',
      'A token that was masked',
      'The position embedding index',
    ],
    correct: 1,
    explanation:
      'Each row predicts a next-token distribution; argmax is the model\'s top guess.',
  },
  {
    q: 'Why does the page show roughly uniform probabilities at the bottom of the corpus pipeline?',
    options: [
      'The vocabulary is too small',
      'The mask is wrong',
      'E and P are random — no training has happened, so the model has no information',
      'Softmax is broken without temperature',
      'Tied weights produce uniform output by construction',
    ],
    correct: 2,
    explanation:
      'With random E (and P), logits carry no learned signal, so softmax flattens to ~uniform.',
  },
  {
    q: 'In the softmax used by the page, why subtract the max logit before exp?',
    latex: FORMULAS.stage2Quiz30,
    options: [
      'To make negative numbers positive',
      {
        text: 'Numerical stability — caps the largest exponent at 0',
        latex: FORMULAS.stage2Quiz31,
      },
      'Required by the chain rule',
      'Changes the resulting probabilities for the better',
      'Only needed when bias b is nonzero',
    ],
    correct: 1,
    explanation:
      'Subtracting the max keeps exp arguments ≤ 0, so exp ≤ 1; the ratio is unchanged.',
  },
  {
    q: 'What happens to the size of M and W if T grows from 30 to 1,000,000?',
    options: [
      'Stay T × T but compute faster',
      {
        text: 'Quadratic blow-up — 1 trillion entries',
        latex: FORMULAS.stage2Quiz32,
      },
      'Only M grows; W stays small',
      'Become diagonal and shrink',
      { text: 'Depend on V, not T', latex: FORMULAS.stage2Quiz33 },
    ],
    correct: 1,
    explanation:
      'Causal mixing is quadratic in T. This is why real LLMs cap context length and use sparse / sliding-window attention.',
  },
  {
    q: 'What was GPT-2\'s max sequence length?',
    options: ['256', '512', '1024', '2048', '4096'],
    correct: 2,
    explanation: 'GPT-2 was trained with context length 1024 tokens.',
  },
  {
    q: 'What was GPT-3\'s max sequence length?',
    options: ['1024', '2048', '4096', '8192', '16384'],
    correct: 1,
    explanation:
      'GPT-3 doubled the context window to 2048 tokens (later models extended further).',
  },
  {
    q: 'What was GPT-2\'s vocabulary size?',
    options: ['10,000', '32,000', '50,257', '65,536', '100,000'],
    correct: 2,
    explanation:
      'GPT-2 used a BPE tokenizer with 50,257 tokens. GPT-3 reused the same tokenizer.',
  },
  {
    q: 'What is the embedding dimension of GPT-2 small?',
    options: ['512', '768', '1024', '1280', '1600'],
    correct: 1,
    explanation:
      'Small: 768. Medium: 1024. Large: 1280. XL: 1600.',
  },
  {
    q: 'What is the embedding dimension of the largest GPT-3 (175B)?',
    options: ['1600', '2048', '4096', '8192', '12,288'],
    correct: 4,
    explanation:
      'GPT-3 175B has d_model = 12,288 across 96 layers and 96 heads.',
  },
  {
    q: 'In the corpus pipeline, clicking a row in E (vocab) changes "selected idx" but also updates t. How?',
    options: [
      'It does not — they are independent',
      'It jumps t to the first occurrence of that vocab id in the tokenized corpus',
      'It sets t = idx directly',
      'It resets t to 0',
      'It sets t to the last position',
    ],
    correct: 1,
    explanation:
      'Clicking E picks a vocab id; the page then jumps t to the first position where that id appears.',
  },
  {
    q: 'Why is sinusoidal P (used in the corpus section) a fixed formula rather than learned?',
    latex: FORMULAS.stage2Quiz34,
    options: [
      'Sin/cos are differentiable, learned versions are not',
      'Parameter-free and generalizes to unseen lengths',
      'Faster on GPUs',
      'Softmax requires it',
      'Eliminates the need for token embeddings',
    ],
    correct: 1,
    explanation:
      'Sinusoidal P is parameter-free and gives a smooth function of position that extrapolates beyond training lengths. (Many modern LLMs still use learned or rotary positions; this is just one classic choice.)',
  },
  {
    q: 'What is missing in Stage 2a that Stage 2c-i adds?',
    options: [
      'A real tokenizer',
      {
        text: 'Context mixing via attention weights',
        latex: FORMULAS.stage2Quiz35,
      },
      'Layer norm',
      'A larger vocab',
      'Multi-head outputs',
    ],
    correct: 1,
    explanation:
      'Stage 2a stops at h_t (token + position only). 2c-i adds the hand-picked attention scores that mix h_j into h_t for j ≤ t.',
  },
];

export function Stage2QuizPage() {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [struck, setStruck] = useState<Record<string, true>>({});
  const [submitted, setSubmitted] = useState(false);

  const toggleStrike = (qi: number, oi: number) => {
    const key = `${qi}-${oi}`;
    setStruck((prev) => {
      const next = { ...prev };
      if (next[key]) delete next[key];
      else next[key] = true;
      return next;
    });
  };

  const score = useMemo(
    () =>
      QUESTIONS.reduce(
        (acc, q, i) => acc + (answers[i] === q.correct ? 1 : 0),
        0
      ),
    [answers]
  );

  const downloadResults = () => {
    const ts = new Date();
    const isoTimestamp = ts.toISOString();
    const fileTimestamp = isoTimestamp.replace(/[:.]/g, '-');
    const optionText = (
      opt: Question['options'][number]
    ): string => (typeof opt === 'string' ? opt : opt.text);

    const payload = {
      quiz: 'stage2-quiz',
      timestamp: isoTimestamp,
      score,
      total: QUESTIONS.length,
      percent: Math.round((score / QUESTIONS.length) * 1000) / 10,
      answers: QUESTIONS.map((q, i) => {
        const selected = answers[i];
        const struckIndices = q.options
          .map((_, j) => j)
          .filter((j) => struck[`${i}-${j}`]);
        return {
          index: i,
          question: q.q,
          options: q.options.map(optionText),
          correctIndex: q.correct,
          correctAnswer: optionText(q.options[q.correct]),
          selectedIndex: selected ?? null,
          selectedAnswer:
            selected !== undefined ? optionText(q.options[selected]) : null,
          isCorrect: selected === q.correct,
          struckIndices,
        };
      }),
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stage2-quiz-${fileTimestamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <h2>Stage 2a Quiz — 30 questions</h2>
      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Multi-choice quiz on the content of{' '}
        <Link to="/stage2-flow">/stage2-flow</Link>. One correct answer per
        question. Pick all 30, then click <strong>Check answers</strong>.
      </p>

      <ol style={{ paddingLeft: '1.25rem' }}>
        {QUESTIONS.map((q, i) => {
          const selected = answers[i];
          const isCorrect = submitted && selected === q.correct;
          const isWrong =
            submitted && selected !== undefined && selected !== q.correct;
          return (
            <li
              key={i}
              style={{
                margin: '1.5rem 0',
                padding: '0.75rem 1rem',
                background: isCorrect
                  ? color.success.bg
                  : isWrong
                    ? color.error.bg
                    : color.bg.subtle,
                border: `1px solid ${color.border.default}`,
                borderRadius: radius.sm,
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: space.sm }}>
                {q.q}
              </div>
              {q.latex && <Tex latex={q.latex} block />}
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.35rem',
                }}
              >
                {q.options.map((opt, j) => {
                  const chosen = selected === j;
                  const correct = submitted && j === q.correct;
                  const text = typeof opt === 'string' ? opt : opt.text;
                  const optLatex = typeof opt === 'string' ? undefined : opt.latex;
                  const isStruck = !!struck[`${i}-${j}`];
                  return (
                    <label
                      key={j}
                      style={{
                        display: 'flex',
                        gap: space.sm,
                        alignItems: 'flex-start',
                        cursor: submitted ? 'default' : 'pointer',
                        padding: '0.25rem 0.5rem',
                        background: correct
                          ? color.success.border
                          : submitted && chosen
                            ? color.error.border
                            : 'transparent',
                        borderRadius: '3px',
                      }}
                    >
                      <input
                        type="radio"
                        name={`q-${i}`}
                        checked={chosen}
                        disabled={submitted || isStruck}
                        onChange={() =>
                          setAnswers((prev) => ({ ...prev, [i]: j }))
                        }
                        style={{ marginTop: '0.2rem' }}
                      />
                      <span
                        style={{
                          flex: 1,
                          textDecoration: isStruck ? 'line-through' : 'none',
                          color: isStruck ? color.text.muted : 'inherit',
                          opacity: isStruck ? 0.7 : 1,
                        }}
                      >
                        <span>{text}</span>
                        {optLatex && (
                          <span style={{ marginLeft: space.sm }}>
                            <Tex latex={optLatex} />
                          </span>
                        )}
                      </span>
                      {!submitted && (
                        <button
                          type="button"
                          aria-label={isStruck ? 'Un-strike' : 'Strike out'}
                          title={isStruck ? 'Un-strike' : 'Strike out (eliminate)'}
                          onClick={(e) => {
                            e.preventDefault();
                            // strike-out should also clear the radio if it was selected
                            if (chosen) {
                              setAnswers((prev) => {
                                const { [i]: _drop, ...rest } = prev;
                                return rest;
                              });
                            }
                            toggleStrike(i, j);
                          }}
                          style={{
                            minWidth: '1.6rem',
                            height: '1.6rem',
                            border: `1px solid ${color.text.muted}`,
                            borderRadius: radius.sm,
                            background: isStruck ? color.bg.disabled : color.bg.page,
                            cursor: 'pointer',
                            fontSize: font.size.base,
                            fontWeight: 700,
                            lineHeight: 1,
                            padding: 0,
                            color: color.text.primary,
                            fontFamily:
                              '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            alignSelf: 'flex-start',
                            flex: '0 0 auto',
                          }}
                        >
                          {isStruck ? '↺' : '×'}
                        </button>
                      )}
                    </label>
                  );
                })}
              </div>
              {submitted && (
                <div
                  style={{
                    marginTop: space.sm,
                    fontSize: font.size.sm,
                    color: color.text.secondary,
                  }}
                >
                  <strong>Why:</strong> {q.explanation}
                  {q.explanationLatex && (
                    <Tex latex={q.explanationLatex} block />
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>

      <div
        style={{
          position: 'sticky',
          bottom: 0,
          background: color.bg.surface,
          borderTop: `1px solid ${color.border.default}`,
          padding: '0.75rem 0',
          marginTop: space.lg,
          display: 'flex',
          gap: space.lg,
          alignItems: 'center',
        }}
      >
        {!submitted ? (
          <button
            type="button"
            onClick={() => {
              setSubmitted(true);
              downloadResults();
            }}
            disabled={Object.keys(answers).length < QUESTIONS.length}
            style={{
              padding: '0.5rem 1rem',
              fontSize: '0.95rem',
              cursor:
                Object.keys(answers).length < QUESTIONS.length
                  ? 'not-allowed'
                  : 'pointer',
            }}
          >
            Check answers
          </button>
        ) : (
          <button
            type="button"
            onClick={() => {
              setAnswers({});
              setStruck({});
              setSubmitted(false);
            }}
            style={{
              padding: '0.5rem 1rem',
              fontSize: '0.95rem',
              cursor: 'pointer',
            }}
          >
            Reset
          </button>
        )}
        <div style={{ color: color.text.secondary }}>
          {submitted ? (
            <strong>
              Score: {score} / {QUESTIONS.length}
            </strong>
          ) : (
            <span>
              Answered: {Object.keys(answers).length} / {QUESTIONS.length}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
