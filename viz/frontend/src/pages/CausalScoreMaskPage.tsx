import { useEffect, useMemo, useState } from 'react';

import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { TrainingCorpus } from '../components/TrainingCorpus';
import { buildLegend } from '../legendEntries';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';

import { color, space } from '../theme';
import { tokenize, chunkIndices } from '../funcs';
import { shadeDiverging } from '../presentationFunctions';
import { FORMULAS } from '../formulas';

const VOCAB = ['the', 'cat', 'dog', 'in', 'on', 'hat', 'sofa'] as const;

const LEGEND = buildLegend('rawScoreSj', 'maskedScoreSjPrime', 'queryPosT', 'keyPosJ', 'negInfMasked');

const WRAP_COLS = 25;

function ScoreInputRow({
  rowLabel,
  tokens,
  values,
  highlightT,
  hideAfterT,
  maxAbs,
  onChange,
}: {
  rowLabel: string;
  tokens: string[];
  values: number[];
  highlightT: number;
  hideAfterT?: boolean;
  maxAbs: number;
  onChange?: (j: number, v: number) => void;
}) {
  const chunks = chunkIndices(values.length, WRAP_COLS);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
      {chunks.map((indices, ci) => (
        <table
          key={ci}
          style={{
            borderCollapse: 'collapse',
            fontFamily: 'monospace',
            fontSize: '0.8rem',
          }}
        >
          <thead>
            <tr>
              <th style={{ padding: '0.15rem 0.4rem' }}>j</th>
              {indices.map((j) => (
                <th
                  key={j}
                  style={{
                    padding: '0.15rem 0.4rem',
                    borderBottom: `1px solid ${color.border.strong}`,
                    background: j === highlightT ? color.highlightBg : undefined,
                    color:
                      hideAfterT && j > highlightT ? color.text.muted : undefined,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {j} {tokens[j]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th
                style={{
                  padding: '0.15rem 0.4rem',
                  borderRight: `1px solid ${color.border.strong}`,
                  textAlign: 'right',
                  whiteSpace: 'nowrap',
                }}
              >
                {rowLabel}
              </th>
              {indices.map((j) => {
                const v = values[j];
                const masked = !Number.isFinite(v);
                if (onChange && !masked) {
                  return (
                    <td
                      key={j}
                      style={{
                        padding: '0.15rem 0.2rem',
                        textAlign: 'right',
                        background: shadeDiverging(v, maxAbs),
                      }}
                    >
                      <input
                        type="number"
                        step="0.1"
                        value={v}
                        onChange={(e) => onChange(j, Number(e.target.value))}
                        style={{
                          width: '3.2rem',
                          fontFamily: 'monospace',
                          fontSize: '0.8rem',
                          textAlign: 'right',
                          background: 'transparent',
                          border: 'none',
                        }}
                      />
                    </td>
                  );
                }
                return (
                  <td
                    key={j}
                    style={{
                      padding: '0.15rem 0.4rem',
                      textAlign: 'right',
                      background: masked ? color.text.primary : shadeDiverging(v, maxAbs),
                      color: masked ? color.text.muted : color.text.primary,
                    }}
                  >
                    {masked ? '−∞' : v.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          </tbody>
        </table>
      ))}
    </div>
  );
}

export function CausalScoreMaskPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const [t, setT] = useState<number>(3);
  const [scores, setScores] = useState<number[]>(() => {
    const init = Array<number>(25).fill(0);
    init[0] = 0.5;
    init[3] = 1.0;
    return init;
  });

  const corpus = CORPUS_OPTIONS[corpusKey];
  const tokens = useMemo(() => tokenize(corpus), [corpus]);
  const T_LEN = tokens.length;

  // Persist the canonical state on next paint, but render uses a derived
  // version that is ALWAYS exactly T_LEN long so cell arrays stay consistent
  // with the corpus on the very first render after a size change.
  useEffect(() => {
    setScores((prev) => {
      if (prev.length === T_LEN) return prev;
      const next = Array<number>(T_LEN).fill(0);
      for (let i = 0; i < Math.min(prev.length, T_LEN); i++) next[i] = prev[i];
      return next;
    });
  }, [T_LEN]);

  const safeT = Math.max(0, Math.min(T_LEN - 1, t));
  const safeScores = useMemo(() => {
    if (scores.length === T_LEN) return scores;
    const next = Array<number>(T_LEN).fill(0);
    for (let i = 0; i < Math.min(scores.length, T_LEN); i++) next[i] = scores[i];
    return next;
  }, [scores, T_LEN]);

  const masked = useMemo(
    () =>
      safeScores.map((s, j) => (j <= safeT ? s : Number.NEGATIVE_INFINITY)),
    [safeScores, safeT]
  );
  const finite = masked.filter((v) => Number.isFinite(v));
  const maxAbs = Math.max(
    ...safeScores.map((v) => Math.abs(v)),
    ...finite.map((v) => Math.abs(v)),
    1e-9
  );

  function setScoreAt(j: number, v: number) {
    setScores((prev) => {
      const next = prev.slice();
      next[j] = v;
      return next;
    });
  }

  const visible = T_LEN - (safeT + 1);
  const queryWord = tokens[safeT] ?? '?';

  return (
    <div>
      <h2>Causal Score Mask</h2>
      <FormulaDisplay
        latex={FORMULAS.attnConcepts05}
      />
      <Legend entries={LEGEND} />

      <h3>Controls</h3>
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'center',
          margin: '0.5rem 0',
        }}
      >
        <label>
          query t:&nbsp;
          <input
            type="number"
            min={0}
            max={T_LEN - 1}
            value={safeT}
            onChange={(e) =>
              setT(
                Math.max(0, Math.min(T_LEN - 1, Number(e.target.value)))
              )
            }
            style={{ width: '5rem' }}
          />
        </label>
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          → token = <strong>{queryWord}</strong>
        </span>
      </div>

      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        Vocab V={VOCAB.length}, T={T_LEN} tokens. Each <code>j</code> is one
        position in the corpus. Set <code>t</code> = the query position; the
        mask blocks every token that comes <em>after</em> it.
      </p>

      <TrainingCorpus corpusKey={corpusKey} onChange={setCorpusKey} />

      <h3>Inputs</h3>
      <h4>raw scores s_j (one per corpus position)</h4>
      <ScoreInputRow
        rowLabel="s_j"
        tokens={tokens}
        values={safeScores}
        highlightT={safeT}
        maxAbs={maxAbs}
        onChange={setScoreAt}
      />

      <h3>Result — masked scores s'_j</h3>
      <ScoreInputRow
        rowLabel="s'_j"
        tokens={tokens}
        values={masked}
        highlightT={safeT}
        hideAfterT
        maxAbs={maxAbs}
      />

      <h3>What this does to the corpus</h3>
      <ul style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.6 }}>
        <li>
          <strong>Doesn't touch the tokens themselves.</strong> The corpus is
          unchanged; only the <em>score vector indexing into it</em> is
          modified.
        </li>
        <li>
          <strong>Blacks out the future.</strong> At <code>t={safeT}</code>{' '}
          ("{queryWord}"), positions <code>{safeT + 1}..{T_LEN - 1}</code> become{' '}
          <code>−∞</code>. That's <strong>{visible}</strong> of {T_LEN}{' '}
          tokens hidden — every token to the right of the highlighted one.
        </li>
        <li>
          <strong>Preserves the past unchanged.</strong> Positions{' '}
          <code>0..{safeT}</code> keep their raw <code>s_j</code> values
          untouched.
        </li>
        <li>
          <strong>Ensures softmax gives them 0 weight.</strong>{' '}
          <code>exp(−∞) = 0</code> exactly. Setting the masked entries to 0
          would still leak weight after softmax — masking happens{' '}
          <em>before</em> softmax for this reason.
        </li>
        <li>
          <strong>Why it matters at training time:</strong> language models
          predict the next token. If the query at position <code>t</code>{' '}
          could see token <code>t+1</code>, the prediction task is trivial
          (just copy). The mask enforces "predict the future from the past
          only," so the model has to learn real patterns.
        </li>
      </ul>
    </div>
  );
}
