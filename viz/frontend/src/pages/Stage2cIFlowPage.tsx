import { Select } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { TrainingCorpus } from '../components/TrainingCorpus';
import { buildLegend } from '../legendEntries';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';

import { color, space, font } from '../theme';
import { FORMULAS } from '../formulas';
import {
  randomMatrix,
  sinusoidalP,
  tokenize,
  tokenIds,
  softmaxRow,
  addVec,
  scaleVec,
  zeros,
  chunkIndices,
} from '../funcs';
import { shadeDiverging, shadeProb } from '../presentationFunctions';

const VOCAB = ['the', 'cat', 'dog', 'in', 'on', 'hat', 'sofa'] as const;

const D_MODEL_OPTIONS = [4, 8, 16] as const;
type DModel = (typeof D_MODEL_OPTIONS)[number];

const LEGEND = buildLegend('combinedInputH', 'mixedHiddenHPrime', 'embeddingMatrixE', 'tokenIdx', 'positionMatrixP', 'queryPosT', 'keyPosJ', 'rawScoreSj', 'attnWeightAj');

const WRAP_COLS = 25;

function ScoresAndMaskTable({
  posLabels,
  scores,
  maskedScores,
  highlightT,
  scoreMaxAbs,
  onChange,
}: {
  posLabels: string[];
  scores: number[];
  maskedScores: number[];
  highlightT: number;
  scoreMaxAbs: number;
  onChange: (j: number, v: number) => void;
}) {
  const chunks = chunkIndices(scores.length, WRAP_COLS);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm, marginBottom: space.md }}>
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
              <th style={{ padding: '0.15rem 0.4rem', textAlign: 'left' }}>
                j
              </th>
              {indices.map((j) => (
                <th
                  key={j}
                  style={{
                    padding: '0.15rem 0.4rem',
                    borderBottom: `1px solid ${color.border.strong}`,
                    background: j === highlightT ? color.highlightBg : undefined,
                    color: j > highlightT ? color.text.muted : undefined,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {posLabels[j]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th
                style={{
                  padding: '0.15rem 0.4rem',
                  textAlign: 'right',
                  borderRight: `1px solid ${color.border.strong}`,
                  whiteSpace: 'nowrap',
                }}
              >
                s_j (raw)
              </th>
              {indices.map((j) => {
                const s = scores[j];
                const masked = j > highlightT;
                return (
                  <td
                    key={j}
                    style={{
                      padding: '0.15rem 0.2rem',
                      textAlign: 'right',
                      background: masked
                        ? color.text.primary
                        : shadeDiverging(s, scoreMaxAbs),
                    }}
                  >
                    <input
                      type="number"
                      step="0.1"
                      value={masked ? '' : s}
                      disabled={masked}
                      onChange={(e) => onChange(j, Number(e.target.value))}
                      style={{
                        width: '3.2rem',
                        fontFamily: 'monospace',
                        fontSize: '0.8rem',
                        textAlign: 'right',
                        background: 'transparent',
                        border: 'none',
                        color: masked ? color.text.muted : 'inherit',
                      }}
                      placeholder={masked ? '−∞' : undefined}
                    />
                  </td>
                );
              })}
            </tr>
            <tr>
              <th
                style={{
                  padding: '0.15rem 0.4rem',
                  textAlign: 'right',
                  borderRight: `1px solid ${color.border.strong}`,
                  whiteSpace: 'nowrap',
                }}
              >
                after mask
              </th>
              {indices.map((j) => {
                const s = maskedScores[j];
                return (
                  <td
                    key={j}
                    style={{
                      padding: '0.15rem 0.4rem',
                      textAlign: 'right',
                      background: Number.isFinite(s)
                        ? shadeDiverging(s, scoreMaxAbs)
                        : color.text.primary,
                      color: Number.isFinite(s) ? color.text.primary : color.text.muted,
                    }}
                  >
                    {Number.isFinite(s) ? s.toFixed(2) : '−∞'}
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

function WeightsRow({
  posLabels,
  weights,
  highlightT,
}: {
  posLabels: string[];
  weights: number[];
  highlightT: number;
}) {
  const chunks = chunkIndices(weights.length, WRAP_COLS);
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
              {indices.map((j) => (
                <th
                  key={j}
                  style={{
                    padding: '0.15rem 0.4rem',
                    borderBottom: `1px solid ${color.border.strong}`,
                    background: j === highlightT ? color.highlightBg : undefined,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {posLabels[j]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              {indices.map((j) => {
                const v = weights[j];
                return (
                  <td
                    key={j}
                    style={{
                      padding: '0.15rem 0.4rem',
                      textAlign: 'right',
                      background: shadeProb(v),
                    }}
                  >
                    {v.toFixed(2)}
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

function CorpusRow({
  data,
  labels,
  shade,
  highlightIdx,
}: {
  data: number[];
  labels?: string[];
  shade: (v: number) => string;
  highlightIdx?: number;
}) {
  return (
    <table
      style={{
        borderCollapse: 'collapse',
        fontFamily: 'monospace',
        fontSize: font.size.sm,
      }}
    >
      {labels && (
        <thead>
          <tr>
            {labels.map((l, j) => (
              <th
                key={j}
                style={{
                  padding: '0.2rem 0.4rem',
                  borderBottom: `1px solid ${color.border.strong}`,
                }}
              >
                {l}
              </th>
            ))}
          </tr>
        </thead>
      )}
      <tbody>
        <tr>
          {data.map((v, j) => (
            <td
              key={j}
              style={{
                padding: '0.2rem 0.4rem',
                textAlign: 'right',
                background: shade(v),
                fontWeight: highlightIdx === j ? 'bold' : 'normal',
                outline: highlightIdx === j ? `2px solid ${color.highlight}` : undefined,
                color: Number.isFinite(v) ? color.text.primary : color.text.muted,
              }}
            >
              {Number.isFinite(v) ? v.toFixed(2) : '−∞'}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

function CorpusMatrix({
  data,
  rowLabels,
  colLabels,
  highlightRow,
  shade,
}: {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightRow?: number;
  shade: (v: number) => string;
}) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table
        style={{
          borderCollapse: 'collapse',
          fontFamily: 'monospace',
          fontSize: '0.8rem',
        }}
      >
        <thead>
          <tr>
            <th style={{ padding: '0.15rem 0.4rem' }}></th>
            {colLabels.map((c, j) => (
              <th
                key={j}
                style={{
                  padding: '0.15rem 0.4rem',
                  borderBottom: `1px solid ${color.border.strong}`,
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={i}
              style={{
                outline: highlightRow === i ? `2px solid ${color.highlight}` : undefined,
              }}
            >
              <th
                style={{
                  padding: '0.15rem 0.4rem',
                  borderRight: `1px solid ${color.border.strong}`,
                  textAlign: 'right',
                  background: highlightRow === i ? color.highlightBg : undefined,
                  whiteSpace: 'nowrap',
                }}
              >
                {rowLabels[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '0.15rem 0.4rem',
                    textAlign: 'right',
                    background: shade(v),
                    color: Number.isFinite(v) ? color.text.primary : color.text.muted,
                  }}
                >
                  {Number.isFinite(v) ? v.toFixed(2) : '−∞'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Stage2cIFlowPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const [dModel, setDModel] = useState<DModel>(8);
  const [seed, setSeed] = useState(1);
  const [t, setT] = useState(3);
  const [scores, setScores] = useState<number[]>(() => {
    const init = zeros(25);
    init[0] = 0.5;
    init[3] = 1.0;
    return init;
  });

  const corpus = CORPUS_OPTIONS[corpusKey];
  const tokens = useMemo(() => tokenize(corpus), [corpus]);
  const T_LEN = tokens.length;
  const ids = useMemo(() => tokenIds(tokens, VOCAB), [tokens]);

  const E = useMemo(
    () => randomMatrix(seed, VOCAB.length, dModel, 1),
    [seed, dModel]
  );
  const P = useMemo(() => sinusoidalP(T_LEN, dModel), [T_LEN, dModel]);

  // Per-position hidden vectors h_i = E[idx_i] + P[i]
  const H = useMemo(
    () => ids.map((id, i) => addVec(E[id], P[i])),
    [ids, E, P]
  );

  // Resize / clamp scores when corpus changes; preserve overlap.
  useEffect(() => {
    setScores((prev) => {
      if (prev.length === T_LEN) return prev;
      const next = zeros(T_LEN);
      for (let i = 0; i < Math.min(prev.length, T_LEN); i++) next[i] = prev[i];
      return next;
    });
    setT((prevT) => Math.min(prevT, T_LEN - 1));
  }, [T_LEN]);

  const maskedScores = useMemo(
    () =>
      scores.map((s, j) => (j <= t ? s : Number.NEGATIVE_INFINITY)),
    [scores, t]
  );

  const attnWeights = useMemo(
    () => softmaxRow(maskedScores),
    [maskedScores]
  );

  const hMixed = useMemo(() => {
    const out = zeros(dModel);
    for (let j = 0; j < T_LEN; j++) {
      const wj = attnWeights[j];
      if (wj === 0) continue;
      const contrib = scaleVec(H[j], wj);
      for (let k = 0; k < dModel; k++) out[k] += contrib[k];
    }
    return out;
  }, [attnWeights, H, T_LEN, dModel]);

  const hRaw = H[t] ?? zeros(dModel);

  const hMaxAbs = Math.max(
    ...H.flat().map((v) => Math.abs(v)),
    ...hMixed.map((v) => Math.abs(v)),
    1e-9
  );
  const scoreFinite = maskedScores.filter((v) => Number.isFinite(v));
  const scoreMaxAbs = Math.max(
    ...scoreFinite.map((v) => Math.abs(v)),
    1e-9
  );

  const dimLabels = Array.from({ length: dModel }, (_, k) => `d${k}`);
  const posLabels = tokens.map((w, i) => `${i} ${w}`);

  function setScoreAt(j: number, v: number) {
    setScores((prev) => {
      const next = prev.slice();
      next[j] = v;
      return next;
    });
  }

  const queryWord = tokens[t] ?? '?';
  const queryIdx = ids[t] ?? -1;

  return (
    <div>
      <h2>Stage 2c-i Flow — Hand-Weighted Attention (one query)</h2>
      <FormulaDisplay
        latex={FORMULAS.stage2ci01}
      />
      <Legend entries={LEGEND} />

      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        First time a token's hidden vector depends on{' '}
        <strong>other tokens</strong>. Pick a query position <code>t</code>{' '}
        and hand-pick the raw score <code>s<sub>j</sub></code> for each
        earlier (or current) key position. Stage{' '}
        <Link to="/stage2-flow">2a</Link> stops at <code>h<sub>t</sub></code>{' '}
        — no mixing. Here we add the mix step.{' '}
        <strong>What's missing:</strong> in 2c-i scores are <em>your</em>{' '}
        guess. 2c-ii will compute them as <code>q<sub>t</sub> · k<sub>j</sub> / √d</code>.
      </p>

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
        <label htmlFor="d-model-select-stage2ci">
          d_model:&nbsp;
          <Select
            id="d-model-select-stage2ci"
            value={dModel}
            onChange={(d) => setDModel(d as DModel)}
            style={{ minWidth: 80 }}
            options={D_MODEL_OPTIONS.map((d) => ({ label: String(d), value: d }))}
          />
        </label>
        <label>
          seed:&nbsp;
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{ width: '5rem' }}
          />
        </label>
        <label>
          query t:&nbsp;
          <input
            type="number"
            min={0}
            max={T_LEN - 1}
            value={t}
            onChange={(e) =>
              setT(
                Math.max(0, Math.min(T_LEN - 1, Number(e.target.value)))
              )
            }
            style={{ width: '5rem' }}
          />
        </label>
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          → token = <strong>{queryWord}</strong> (idx {queryIdx})
        </span>
      </div>

      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Vocab V={VOCAB.length}, T={T_LEN} tokens, d_model={dModel}. E is a
        seeded random [{VOCAB.length}, {dModel}] matrix. P is a sinusoidal [
        {T_LEN}, {dModel}] matrix. h<sub>i</sub> = E[idx<sub>i</sub>] + P[i].
      </p>

      <TrainingCorpus corpusKey={corpusKey} onChange={setCorpusKey} />

      <h3>1. Per-position hidden vectors</h3>
      <FormulaDisplay
        latex={FORMULAS.stage2ci02}
      />
      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        One row per position. The query row <code>h<sub>{t}</sub></code> is
        highlighted. These are the inputs to attention.
      </p>
      <CorpusMatrix
        data={H}
        rowLabels={posLabels}
        colLabels={dimLabels}
        highlightRow={t}
        shade={(v) => shadeDiverging(v, hMaxAbs)}
      />

      <h3>2. Hand-pick raw scores, then causally mask</h3>
      <FormulaDisplay
        latex={FORMULAS.stage2ci03}
      />
      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        Bigger = "pay more attention." Positions <code>j &gt; {t}</code> are{' '}
        <strong>causally masked</strong> to <code>−∞</code> — the query never
        peeks at the future. See{' '}
        <Link to="/causal-score-mask">/causal-score-mask</Link> for the mask
        in isolation. Editable cells scroll horizontally for large corpora.
      </p>
      <ScoresAndMaskTable
        posLabels={posLabels}
        scores={scores}
        maskedScores={maskedScores}
        highlightT={t}
        scoreMaxAbs={scoreMaxAbs}
        onChange={setScoreAt}
      />

      <h3>3. Softmax → attention weights (sum to 1)</h3>
      <FormulaDisplay
        latex={FORMULAS.stage2ci04}
      />
      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        Row-wise softmax over the masked scores. <code>−∞</code> entries map
        to 0, so weight only flows to <code>j ≤ {t}</code>. Atomic page:{' '}
        <Link to="/softmax">/softmax</Link>.
      </p>
      <WeightsRow posLabels={posLabels} weights={attnWeights} highlightT={t} />
      <p style={{ maxWidth: '70ch', color: color.text.secondary, fontSize: font.size.sm }}>
        Sum = {attnWeights.reduce((a, b) => a + b, 0).toFixed(4)}.
      </p>

      <h3>4. Weighted sum → mixed hidden vector</h3>
      <FormulaDisplay
        latex={FORMULAS.stage2ci05}
      />
      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        Each weight scales an entire row of H, then rows are summed
        element-wise. Masked positions contribute 0 because{' '}
        <code>a<sub>j</sub> = 0</code> there. Atomic page:{' '}
        <Link to="/attention-weighted-sum">/attention-weighted-sum</Link>.
      </p>
      <h4>Per-position contributions</h4>
      <FormulaDisplay
        latex={FORMULAS.stage2ci06}
      />
      <CorpusMatrix
        data={H.map((row, j) => scaleVec(row, attnWeights[j]))}
        rowLabels={posLabels.map(
          (l, j) => `${l}  (×${attnWeights[j].toFixed(2)})`
        )}
        colLabels={dimLabels}
        shade={(v) => shadeDiverging(v, hMaxAbs)}
      />
      <h4>Sum → h'<sub>{t}</sub></h4>
      <FormulaDisplay
        latex={String.raw`h'_{${t}} = \sum_{j=0}^{${t}} a_j \cdot h_j`}
      />
      <CorpusRow
        data={hMixed}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, hMaxAbs)}
      />

      <h3>5. Compare: no mix vs mixed</h3>
      <FormulaDisplay
        latex={String.raw`h_{${t}} = E[\text{idx}_{${t}}] + P[${t}] \qquad\text{vs}\qquad h'_{${t}} = \sum_{j \le ${t}} a_j \cdot h_j`}
      />
      <p style={{ maxWidth: '70ch', color: color.text.secondary }}>
        Same query position, two different hidden vectors. Set the score at{' '}
        <code>j={t}</code> to a huge number like 100 → softmax collapses to a
        one-hot at self → <code>h'<sub>{t}</sub> ≈ h<sub>{t}</sub></code>.
        That's "no mixing."
      </p>
      <h4>h<sub>{t}</sub> = E[{queryIdx}] + P[{t}]</h4>
      <CorpusRow
        data={hRaw}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, hMaxAbs)}
      />
      <h4>h'<sub>{t}</sub> = Σ a<sub>j</sub> · h<sub>j</sub></h4>
      <CorpusRow
        data={hMixed}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, hMaxAbs)}
      />

      <h3>What 2c-ii will add</h3>
      <ul style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        <li>
          <strong>Q, K, V projections.</strong> Three trainable matrices
          turn each <code>h<sub>i</sub></code> into a query, key, and value.
          The score becomes <code>q<sub>t</sub> · k<sub>j</sub></code>{' '}
          instead of being hand-picked.
        </li>
        <li>
          <strong>Scale by √d.</strong> Divide the dot product by{' '}
          <code>√d_k</code> to keep softmax temperatures sane as dim grows.
        </li>
        <li>
          <strong>Mix V, not h.</strong> Sum runs over{' '}
          <code>v<sub>j</sub> = h<sub>j</sub> · W_V</code> — values are a
          learned view of each token tuned for what to pass on.
        </li>
        <li>
          The mask + softmax + weighted-sum machinery on this page is{' '}
          <em>identical</em> in 2c-ii. Only the source of the scores changes.
        </li>
      </ul>
    </div>
  );
}
