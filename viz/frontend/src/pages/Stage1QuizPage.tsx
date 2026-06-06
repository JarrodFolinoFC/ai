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
  latex?: string;
  options: (string | { text: string; latex?: string })[];
  correct: number;
  explanation: string;
  explanationLatex?: string;
}

const QUESTIONS: Question[] = [
  {
    q: 'In the bigram model, what does "bigram" mean?',
    options: [
      'A token represented by two embedding vectors',
      'A model that predicts the next token from the single previous token',
      'Any model with two hidden layers',
      'A pair of tokens always treated as one unit',
      'A two-stage softmax',
    ],
    correct: 1,
    explanation:
      'A bigram model conditions only on prev → next. No history beyond the immediately preceding token.',
  },
  {
    q: 'On the page, the vocabulary V has how many tokens?',
    options: ['5', '6', '7', '8', '50,257'],
    correct: 2,
    explanation:
      "Stage 1's toy vocab is `the, cat, dog, in, on, hat, sofa` — 7 tokens.",
  },
  {
    q: 'What is the shape of the bigram count matrix on the page?',
    options: [
      { text: 'V × n_embd', latex: FORMULAS.stage1Quiz01 },
      { text: 'V × V', latex: FORMULAS.stage1Quiz02 },
      { text: 'T × V (T = corpus length)', latex: FORMULAS.stage1Quiz03 },
      { text: 'V', latex: FORMULAS.stage1Quiz04 },
      { text: 'T × T', latex: FORMULAS.stage1Quiz05 },
    ],
    correct: 1,
    explanation:
      'Counts[prev, next] is V × V. Each cell is "how often does next follow prev in the corpus".',
  },
  {
    q: 'How is the empirical probability table built from the count matrix?',
    options: [
      { text: 'Subtract column means', latex: FORMULAS.stage1Quiz06 },
      {
        text: 'Divide each row by its row sum',
        latex: FORMULAS.stage1Quiz07,
      },
      { text: 'Apply softmax to each column', latex: FORMULAS.stage1Quiz08 },
      { text: 'Divide every cell by total tokens', latex: FORMULAS.stage1Quiz09 },
      { text: 'Take log of each cell', latex: FORMULAS.stage1Quiz10 },
    ],
    correct: 1,
    explanation:
      'The empirical row is "given prev=i, what fraction of times does next=j appear" — that is the row sum divided into each entry.',
  },
  {
    q: 'What is the model parameter W in the trainer?',
    options: [
      { text: 'A V × V matrix of logits', latex: FORMULAS.stage1Quiz11 },
      { text: 'A V-vector of biases', latex: FORMULAS.stage1Quiz12 },
      { text: 'A V × n_embd embedding', latex: FORMULAS.stage1Quiz13 },
      'A scalar learning rate',
      'A T × V table of probabilities',
    ],
    correct: 0,
    explanation:
      "Each row W[prev] is a 7-vector of logits — one logit per possible next token. So W is V × V.",
  },
  {
    q: 'How is the next-token distribution p computed in the forward pass?',
    latex: FORMULAS.stage1Quiz14,
    options: [
      { text: 'Row of W, divided by its sum', latex: FORMULAS.stage1Quiz15 },
      { text: 'Row-wise softmax', latex: FORMULAS.stage1Quiz16 },
      { text: 'sigmoid of the row', latex: FORMULAS.stage1Quiz17 },
      { text: 'argmax of the row', latex: FORMULAS.stage1Quiz18 },
      { text: 'Scaled tanh', latex: FORMULAS.stage1Quiz19 },
    ],
    correct: 1,
    explanation:
      'Bigram forward = pick row W[prev], apply softmax to get a proper distribution over next tokens.',
  },
  {
    q: 'What is y in the gradient expression p − y?',
    options: [
      'The empirical bigram row',
      { text: 'A one-hot vector at the target token', latex: FORMULAS.stage1Quiz20 },
      'The previous-step prediction p',
      'A learning-rate-scaled noise term',
      'The softmax of the previous row',
    ],
    correct: 1,
    explanation:
      'y is the one-hot indicator of the actual next token in the corpus. p − y is the gradient of cross-entropy w.r.t. the logits of W[prev].',
  },
  {
    q: 'Why is the gradient of cross-entropy w.r.t. logits exactly p − y?',
    latex: FORMULAS.stage1Quiz21,
    options: [
      'It is an approximation; the true gradient has extra terms',
      'Because cross-entropy with softmax cancels out the softmax derivative neatly',
      'Because y is always one-hot',
      'Because p is normalized',
      'Because the gradient of any loss is the residual',
    ],
    correct: 1,
    explanation:
      "When loss = −log p_target and p is softmax of the logits, the chain rule collapses to a clean (p − y) — the softmax Jacobian and −1/p_target cancel.",
  },
  {
    q: 'In SGD, which row of W does each (prev, target) pair update?',
    options: [
      'The whole matrix W',
      { text: 'Only row W[prev]', latex: FORMULAS.stage1Quiz22 },
      'Only column W[:, target]',
      'A random row chosen by the lr',
      'Both row prev and column target',
    ],
    correct: 1,
    explanation:
      'The example only carries information about prev → target, so only W[prev] receives a gradient.',
  },
  {
    q: 'A row of W with no observations in the corpus has what fate during training?',
    options: [
      'It is zeroed out',
      'It receives a small uniform update',
      'It stays at its random init — there\'s no gradient signal for it',
      'It is filled in by interpolating neighbours',
      'It is deleted from W',
    ],
    correct: 2,
    explanation:
      'No (prev=i, target=*) pairs in the corpus → no gradient → row i never moves from its random init.',
  },
  {
    q: 'After many epochs, what should softmax(W) approach?',
    options: [
      'A uniform distribution',
      'The empirical bigram probability table',
      { text: 'The identity', latex: FORMULAS.stage1Quiz23 },
      'The one-hot of the most common token',
      'The transpose of the bigram counts',
    ],
    correct: 1,
    explanation:
      'SGD on cross-entropy converges softmax(W) to the empirical conditional distribution — the bigram model rediscovers count-and-normalize.',
  },
  {
    q: 'On the page, the "row error" column is total variation distance. How is it defined?',
    latex: FORMULAS.stage1Quiz24,
    options: [
      { text: 'L1 distance', latex: FORMULAS.stage1Quiz25 },
      { text: 'Half the L1 distance', latex: FORMULAS.stage1Quiz26 },
      { text: 'L2 distance', latex: FORMULAS.stage1Quiz27 },
      { text: 'KL divergence', latex: FORMULAS.stage1Quiz28 },
      { text: 'Max element-wise diff', latex: FORMULAS.stage1Quiz29 },
    ],
    correct: 1,
    explanation:
      'TV is half the L1 distance between two distributions. Range [0, 1]: 0 = identical, 1 = disjoint support.',
  },
  {
    q: 'Why does the row "on" converge fastest in the trainer?',
    options: [
      'It is the alphabetically first row',
      'Its empirical distribution has only one non-zero target — one logit needs to dominate',
      'lr is higher for that row',
      'It has the smallest random init',
      'It receives more gradient updates than other rows',
    ],
    correct: 1,
    explanation:
      'In the toy corpus "on" is always followed by "the", so its target is one-hot — only one logit needs to grow.',
  },
  {
    q: 'Why does the row "the" converge slowest?',
    options: [
      'It has the longest token spelling',
      'Its empirical distribution spreads mass across multiple tokens, so multiple logits must balance',
      'It is updated less often',
      'It is initialized further from optimum',
      'softmax has a bias against frequent rows',
    ],
    correct: 1,
    explanation:
      'When the target distribution is spread (e.g. ~4 tokens), several logits must converge in concert — slower than a one-hot target.',
  },
  {
    q: 'Softmax is invariant under what transformation of the logits?',
    options: [
      'Multiplying every logit by a positive constant',
      'Adding a constant to every logit in a row',
      'Taking absolute value',
      'Sorting the logits',
      'Replacing each logit with its sign',
    ],
    correct: 1,
    explanation:
      'softmax(x + c) = softmax(x) for any per-row constant c. So W_final logits are not unique — only their per-row differences are.',
  },
  {
    q: 'What is the cross-entropy loss for a single (prev, target) example?',
    options: [
      { text: 'Mean squared error', latex: FORMULAS.stage1Quiz30 },
      { text: 'Negative log of the probability assigned to the target', latex: FORMULAS.stage1Quiz31 },
      { text: 'Sum of all logits', latex: FORMULAS.stage1Quiz32 },
      { text: 'L1 of (p − y)', latex: FORMULAS.stage1Quiz33 },
      { text: 'Max logit minus target logit', latex: FORMULAS.stage1Quiz34 },
    ],
    correct: 1,
    explanation:
      'CE for a single one-hot target reduces to −log p_target. Lower means the model placed more probability on the right answer.',
  },
  {
    q: 'On step 0 (random init, no training), what does p look like?',
    options: [
      'A one-hot at the target',
      'Roughly uniform — the model has no information yet',
      'Exactly the empirical distribution',
      'All zeros',
      'A delta at vocab id 0',
    ],
    correct: 1,
    explanation:
      'Random small logits in [-1, 1] map under softmax to a near-uniform distribution.',
  },
  {
    q: 'What does the SGD update rule look like for one (prev, target) example?',
    latex: FORMULAS.stage1Quiz22,
    options: [
      { text: 'Add the gradient', latex: FORMULAS.stage1Quiz35 },
      { text: 'Subtract the gradient times lr', latex: FORMULAS.stage1Quiz22 },
      { text: 'Replace the row with the target one-hot', latex: FORMULAS.stage1Quiz36 },
      { text: 'Multiply by gradient', latex: FORMULAS.stage1Quiz37 },
      { text: 'Divide by gradient', latex: FORMULAS.stage1Quiz38 },
    ],
    correct: 1,
    explanation:
      'Standard gradient descent: step opposite the gradient, scaled by the learning rate η.',
  },
  {
    q: 'What is one epoch in the trainer?',
    options: [
      'One step on a single (prev, target) pair',
      'One full pass over every (prev, target) pair in the corpus',
      'A fixed number of seconds',
      'When loss reaches zero',
      'A reset of the W matrix',
    ],
    correct: 1,
    explanation:
      'An epoch = visiting every training pair exactly once.',
  },
  {
    q: 'Why is the bigram model an upper bound on what Stage 2+ can do?',
    options: [
      'It is not — Stage 2 is strictly worse',
      "It only conditions on the previous token, so any model with more context can match or beat it",
      'Stage 2 has fewer parameters',
      'Bigrams use exact arithmetic; Stage 2 is approximate',
      'Stage 2 cannot represent count-and-normalize',
    ],
    correct: 1,
    explanation:
      'A model with broader context has at least the bigram\'s information — it can always condition on prev and ignore the rest.',
  },
];

export function Stage1QuizPage() {
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
      quiz: 'stage1-quiz',
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
    a.download = `stage1-quiz-${fileTimestamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <h2>Stage 1 Quiz — 20 questions</h2>
      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Multi-choice quiz on the content of{' '}
        <Link to="/stage1-flow">/stage1-flow</Link>. One correct answer per
        question. Use <strong>×</strong> to strike out options you've
        eliminated. Pick all 20, then click <strong>Check answers</strong> —
        a JSON file with your results will auto-download.
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
