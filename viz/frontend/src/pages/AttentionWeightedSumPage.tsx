import { Select } from 'antd';
import { useEffect, useMemo, useState } from 'react';

import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { TrainingCorpus } from '../components/TrainingCorpus';
import { buildLegend } from '../legendEntries';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';

import { color, space, radius, font } from '../theme';
import { FORMULAS } from '../formulas';
import {
  randomMatrix,
  sinusoidalP,
  tokenize,
  tokenIds,
  chunkIndices,
} from '../funcs';
import { shadeDiverging, shadeProb } from '../presentationFunctions';

const VOCAB = ['the', 'cat', 'dog', 'in', 'on', 'hat', 'sofa'] as const;

const D_MODEL_OPTIONS = [4, 8, 16] as const;
type DModel = (typeof D_MODEL_OPTIONS)[number];

const LEGEND = buildLegend('mixedHiddenHtPrime', 'attnWeightAj', 'hiddenHj', 'embedDimD');

const WEIGHT_PRESETS: { label: string; build: (T: number, t: number) => number[] }[] = [
  {
    label: 'all on self',
    build: (T, t) => Array.from({ length: T }, (_, j) => (j === t ? 1 : 0)),
  },
  {
    label: 'all on first',
    build: (T) => Array.from({ length: T }, (_, j) => (j === 0 ? 1 : 0)),
  },
  {
    label: 'uniform over j ≤ t',
    build: (T, t) => {
      const w = 1 / (t + 1);
      return Array.from({ length: T }, (_, j) => (j <= t ? w : 0));
    },
  },
  {
    label: 'recency-biased (j ≤ t)',
    build: (T, t) => {
      const raw = Array.from({ length: T }, (_, j) =>
        j <= t ? Math.exp((j - t) / 2) : 0
      );
      const s = raw.reduce((a, b) => a + b, 0);
      return raw.map((x) => (s > 0 ? x / s : 0));
    },
  },
];

const WRAP_COLS = 25;

function WeightInputRow({
  tokens,
  weights,
  highlightT,
  onChange,
}: {
  tokens: string[];
  weights: number[];
  highlightT: number;
  onChange: (j: number, v: number) => void;
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
              <th style={{ padding: '0.15rem 0.4rem' }}>j</th>
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
                a_j
              </th>
              {indices.map((j) => {
                const w = weights[j];
                return (
                  <td
                    key={j}
                    style={{
                      padding: '0.15rem 0.2rem',
                      textAlign: 'right',
                      background: shadeProb(Math.max(0, Math.min(1, w))),
                    }}
                  >
                    <input
                      type="number"
                      step="0.05"
                      value={w}
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
              })}
            </tr>
          </tbody>
        </table>
      ))}
    </div>
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

function CorpusRow({
  data,
  labels,
  shade,
}: {
  data: number[];
  labels?: string[];
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
        {labels && (
          <thead>
            <tr>
              {labels.map((l, j) => (
                <th
                  key={j}
                  style={{
                    padding: '0.15rem 0.4rem',
                    borderBottom: `1px solid ${color.border.strong}`,
                    whiteSpace: 'nowrap',
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
        </tbody>
      </table>
    </div>
  );
}

export function AttentionWeightedSumPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const [dModel, setDModel] = useState<DModel>(8);
  const [seed, setSeed] = useState(1);
  const [t, setT] = useState(3);
  const [weights, setWeights] = useState<number[]>(() => {
    const init = Array<number>(25).fill(0);
    init[0] = 0.4;
    init[3] = 0.6;
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
  const H = useMemo(
    () => ids.map((id, i) => E[id].map((v, k) => v + P[i][k])),
    [ids, E, P]
  );

  useEffect(() => {
    setWeights((prev) => {
      if (prev.length === T_LEN) return prev;
      const next = Array<number>(T_LEN).fill(0);
      for (let i = 0; i < Math.min(prev.length, T_LEN); i++) next[i] = prev[i];
      return next;
    });
  }, [T_LEN]);

  // Derived render-safe versions: ALWAYS exactly T_LEN long, so the very
  // first render after a corpus-size change can't index past the new arrays.
  const safeT = Math.max(0, Math.min(T_LEN - 1, t));
  const safeWeights = useMemo(() => {
    if (weights.length === T_LEN) return weights;
    const next = Array<number>(T_LEN).fill(0);
    for (let i = 0; i < Math.min(weights.length, T_LEN); i++) next[i] = weights[i];
    return next;
  }, [weights, T_LEN]);

  const contributions = useMemo(
    () => H.map((row, j) => row.map((v) => v * safeWeights[j])),
    [H, safeWeights]
  );
  const result = useMemo(() => {
    const out = Array<number>(dModel).fill(0);
    for (let j = 0; j < T_LEN; j++) {
      for (let d = 0; d < dModel; d++) {
        out[d] += safeWeights[j] * H[j][d];
      }
    }
    return out;
  }, [H, safeWeights, T_LEN, dModel]);

  const sumW = safeWeights.reduce((a, b) => a + b, 0);
  const hAbs = Math.max(
    ...H.flat().map((v) => Math.abs(v)),
    ...contributions.flat().map((v) => Math.abs(v)),
    ...result.map((v) => Math.abs(v)),
    1e-9
  );

  function setWeightAt(j: number, v: number) {
    setWeights((prev) => {
      const next = prev.slice();
      next[j] = v;
      return next;
    });
  }

  const dimLabels = Array.from({ length: dModel }, (_, k) => `d${k}`);
  const posLabels = tokens.map((w, i) => `${i} ${w}`);
  const queryWord = tokens[safeT] ?? '?';
  const nonzero = safeWeights
    .map((w, j) => ({ j, w, tok: tokens[j] ?? '?' }))
    .filter((x) => Math.abs(x.w) > 1e-9);

  return (
    <div>
      <h2>Attention-Weighted Sum</h2>
      <FormulaDisplay
        latex={FORMULAS.attnWeightedSum01}
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
        <label htmlFor="d-model-select-aws">
          d_model:&nbsp;
          <Select
            id="d-model-select-aws"
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

      <h3>Weight presets</h3>
      <div style={{ display: 'flex', gap: space.sm, flexWrap: 'wrap', margin: '0.5rem 0' }}>
        {WEIGHT_PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => setWeights(p.build(T_LEN, safeT))}
            style={{
              padding: '0.25rem 0.6rem',
              fontSize: font.size.sm,
              border: `1px solid ${color.border.strong}`,
              borderRadius: radius.sm,
              background: color.bg.page,
              cursor: 'pointer',
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      <TrainingCorpus corpusKey={corpusKey} onChange={setCorpusKey} />

      <h3>Inputs</h3>
      <h4>weights a_j (one per corpus position)</h4>
      <WeightInputRow
        tokens={tokens}
        weights={safeWeights}
        highlightT={safeT}
        onChange={setWeightAt}
      />
      <p style={{ fontFamily: 'monospace', color: sumW > 1.0001 || sumW < 0.9999 ? color.error.fg : color.text.secondary }}>
        Σ a_j = {sumW.toFixed(4)}
        {sumW > 1.0001 || sumW < 0.9999
          ? ' ⚠ not 1 (in the real flow, softmax guarantees this)'
          : ''}
      </p>

      <h4>H = E[idx_i] + P[i] (T × d_model)</h4>
      <CorpusMatrix
        data={H}
        rowLabels={posLabels}
        colLabels={dimLabels}
        highlightRow={safeT}
        shade={(v) => shadeDiverging(v, hAbs)}
      />

      <h3>Result — per-row contributions and the sum</h3>
      <FormulaDisplay
        latex={FORMULAS.attnWeightedSum02}
      />
      <CorpusMatrix
        data={contributions}
        rowLabels={posLabels.map(
          (l, j) => `${l} (×${safeWeights[j].toFixed(2)})`
        )}
        colLabels={dimLabels}
        shade={(v) => shadeDiverging(v, hAbs)}
      />
      <h4>h'<sub>{safeT}</sub> = sum of rows above</h4>
      <CorpusRow
        data={result}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, hAbs)}
      />

      <h3>What this does to the corpus</h3>
      <ul style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.6 }}>
        <li>
          <strong>Doesn't touch the tokens themselves.</strong> The corpus
          stays the same. Only the hidden vector at position{' '}
          <code>t={safeT}</code> changes — from <code>h<sub>{safeT}</sub></code>{' '}
          (just "{queryWord}" + position {safeT}) to <code>h'<sub>{safeT}</sub></code>{' '}
          (a blend of the rows you weighted).
        </li>
        <li>
          <strong>Output dimension is the same as input.</strong>{' '}
          <code>h'<sub>{t}</sub> ∈ ℝ<sup>{dModel}</sup></code> — same shape as
          any single <code>h<sub>j</sub></code>. Attention isn't a
          dimensionality change; it's a re-mix at the same dim.
        </li>
        <li>
          <strong>You picked {nonzero.length} contributor(s):</strong>{' '}
          {nonzero.length === 0 ? (
            <em>none — output is all zeros</em>
          ) : (
            nonzero
              .map(
                (x) =>
                  `j=${x.j} ("${x.tok}")×${x.w.toFixed(2)}`
              )
              .join(', ')
          )}
          .
        </li>
        <li>
          <strong>When weights sum to 1</strong> (Σ a_j ={' '}
          {sumW.toFixed(2)}), the result is a <em>convex combination</em> —
          a point inside the polytope spanned by the contributing{' '}
          <code>h<sub>j</sub></code>'s. No single token's "meaning" is added
          or invented; <code>h'<sub>t</sub></code> is a literal blend of the
          rows you allowed.
        </li>
        <li>
          <strong>Why this is "attention":</strong> from <em>one</em> input
          row plus <em>T</em> weights, you produce a single output row that{' '}
          <em>knows about other positions</em>. That's the only operation
          that introduces cross-position information flow. Stage 2c-ii will
          replace the hand-picked weights with{' '}
          <code>softmax(q<sub>t</sub> · k<sub>j</sub> / √d)</code>, but this
          weighted-sum mechanism stays untouched.
        </li>
      </ul>
    </div>
  );
}
