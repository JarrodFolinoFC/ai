import { Select } from 'antd';
import katex from 'katex';
import { useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../api';
import { CodePanel } from '../components/CodePanel';
import { ErrorBanner } from '../components/ErrorBanner';
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
  maxAbs,
} from '../funcs';

function MathHeading({
  latex,
  hint,
  level = 'h4',
}: {
  latex: string;
  hint?: string;
  level?: 'h4' | 'div';
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    katex.render(latex, ref.current, { displayMode: false, throwOnError: false });
  }, [latex]);
  const Wrapper = level;
  return (
    <Wrapper
      style={{
        margin: '0 0 0.5rem',
        fontWeight: 'normal',
        fontSize: '0.95rem',
        color: color.text.secondary,
      }}
    >
      <span ref={ref} />
      {hint ? (
        <span style={{ marginLeft: space.sm, fontSize: font.size.sm, color: color.text.muted }}>
          {hint}
        </span>
      ) : null}
    </Wrapper>
  );
}

const VOCAB = ['the', 'cat', 'dog', 'in', 'on', 'hat', 'sofa'] as const;

const D_MODEL_OPTIONS = [4, 8, 16] as const;
type DModel = (typeof D_MODEL_OPTIONS)[number];

const LEGEND = buildLegend('combinedInputH', 'embeddingMatrixE', 'tokenIdx', 'positionMatrixP', 'timeStepT');

function CorpusMatrix({
  titleLatex,
  titleHint,
  testId,
  tokens,
  matrix,
  maxAbs,
  dModel,
  indexLabel = 't',
  onRowClick,
  selectedRow,
  onRowHover,
  externalHighlightRows,
}: {
  titleLatex: string;
  titleHint?: string;
  testId?: string;
  tokens: string[];
  matrix: number[][];
  maxAbs: number;
  dModel: number;
  indexLabel?: string;
  onRowClick?: (rowIndex: number) => void;
  selectedRow?: number | null;
  onRowHover?: (rowIndex: number | null) => void;
  externalHighlightRows?: number[];
}) {
  const externalSet = new Set(externalHighlightRows ?? []);
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const rowKeys = matrix.map((row) => row.map((v) => v.toFixed(2)).join(','));

  return (
    <div data-testid={testId}>
      <MathHeading latex={titleLatex} hint={titleHint} />
      <table style={{ borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: font.size.sm }}>
        <thead>
          <tr>
            <th style={{ padding: '0.25rem 0.5rem', borderBottom: `1px solid ${color.border.strong}` }}>{indexLabel}</th>
            <th style={{ padding: '0.25rem 0.5rem', borderBottom: `1px solid ${color.border.strong}` }}>token</th>
            {Array.from({ length: dModel }, (_, i) => (
              <th
                key={i}
                style={{ padding: '0.25rem 0.5rem', borderBottom: `1px solid ${color.border.strong}` }}
              >
                {i}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, t) => {
            const highlighted = hoverKey !== null && rowKeys[t] === hoverKey;
            const selected = selectedRow === t;
            const externallyHighlighted = externalSet.has(t);
            const labelBg = selected
              ? color.warning.border
              : externallyHighlighted
                ? color.highlightBg
                : highlighted
                  ? color.highlightBg
                  : undefined;
            const showOutline = highlighted || externallyHighlighted;
            return (
              <tr
                key={t}
                onMouseEnter={() => {
                  setHoverKey(rowKeys[t]);
                  onRowHover?.(t);
                }}
                onMouseLeave={() => {
                  setHoverKey(null);
                  onRowHover?.(null);
                }}
                onClick={onRowClick ? () => onRowClick(t) : undefined}
                style={{ cursor: onRowClick ? 'pointer' : undefined }}
              >
                <th
                  scope="row"
                  style={{
                    textAlign: 'right',
                    padding: '0.25rem 0.5rem',
                    borderRight: `1px solid ${color.border.strong}`,
                    background: labelBg,
                  }}
                >
                  {t}
                </th>
                <th
                  scope="row"
                  style={{
                    textAlign: 'left',
                    padding: '0.25rem 0.5rem',
                    borderRight: `1px solid ${color.border.strong}`,
                    background: labelBg,
                  }}
                >
                  {tokens[t]}
                </th>
                {row.map((v, i) => {
                  const ratio = v / maxAbs;
                  const bg =
                    ratio >= 0
                      ? `rgba(59, 130, 246, ${0.1 + 0.5 * ratio})`
                      : `rgba(239, 68, 68, ${0.1 + 0.5 * -ratio})`;
                  return (
                    <td
                      key={i}
                      style={{
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: bg,
                        outline: showOutline ? `2px solid ${color.highlight}` : undefined,
                        outlineOffset: showOutline ? '-2px' : undefined,
                      }}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PositionMath({
  t,
  dModel,
  P,
}: {
  t: number;
  dModel: number;
  P: number[][];
}) {
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        padding: space.md,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <MathHeading level="div" latex={`P[t=${t}]`} />
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>i</th>
              <th style={{ textAlign: 'left', padding: '0.15rem 0.5rem' }}>fn</th>
              <th style={{ textAlign: 'left', padding: '0.15rem 0.5rem' }}>arg = t / 10000^(2k/d)</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>value</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: dModel }, (_, i) => {
              const evenI = i % 2 === 0 ? i : i - 1;
              const denom = Math.pow(10000, evenI / dModel);
              const arg = t / denom;
              const fn = i % 2 === 0 ? 'sin' : 'cos';
              return (
                <tr key={i}>
                  <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>{i}</td>
                  <td style={{ padding: '0.15rem 0.5rem' }}>{fn}</td>
                  <td style={{ padding: '0.15rem 0.5rem', color: color.text.secondary }}>
                    {t} / 10000^({evenI}/{dModel}) = {arg.toFixed(4)}
                  </td>
                  <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>
                    {P[t][i].toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div style={{ flex: '1 1 32ch', minWidth: '32ch' }}>
          <PositionMathCode t={t} dModel={dModel} />
        </div>
      </div>
    </div>
  );
}

function PositionMathCode({ t, dModel }: { t: number; dModel: number }) {
  const denom0 = Math.pow(10000, 0 / dModel);
  const arg0 = t / denom0;
  const value0 = Math.sin(arg0);

  const lines: string[] = [
    `// P[t=${t}] for d_model=${dModel}`,
    `const t = ${t};`,
    `const d = ${dModel};`,
    `const P_t = [];`,
    ``,
    `// i=0 unrolled — the worked example`,
    `const denom0 = Math.pow(10000, 0/${dModel});  // = ${denom0.toFixed(4)}`,
    `const arg0   = t / denom0;            // = ${arg0.toFixed(4)}`,
    `P_t[0] = Math.sin(arg0);              // = ${value0.toFixed(4)}`,
    ``,
    `// reusable for the rest`,
    `function pDim(i, t, d) {`,
    `  const evenI = i % 2 === 0 ? i : i - 1;`,
    `  const denom = Math.pow(10000, evenI / d);`,
    `  const arg = t / denom;`,
    `  return i % 2 === 0 ? Math.sin(arg) : Math.cos(arg);`,
    `}`,
    ``,
  ];
  for (let i = 1; i < dModel; i++) {
    const evenI = i % 2 === 0 ? i : i - 1;
    const denom = Math.pow(10000, evenI / dModel);
    const arg = t / denom;
    const value = i % 2 === 0 ? Math.sin(arg) : Math.cos(arg);
    lines.push(
      `P_t[${i}] = pDim(${i}, t, d);  // ${i % 2 === 0 ? 'sin' : 'cos'}(${arg.toFixed(4)}) = ${value.toFixed(4)}`
    );
  }
  lines.push(``, `console.log(P_t);`);
  const code = lines.join('\n');
  return (
    <details open>
      <summary style={{ cursor: 'pointer', color: color.text.emphasis, fontSize: '0.8rem' }}>
        JavaScript — paste into the browser console
      </summary>
      <pre
        style={{
          marginTop: space.sm,
          padding: space.md,
          background: color.text.primary,
          color: color.text.onDark,
          border: `1px solid ${color.text.primary}`,
          borderRadius: radius.sm,
          fontSize: '0.78rem',
          overflowX: 'auto',
          lineHeight: 1.45,
        }}
      >
        <code>{code}</code>
      </pre>
    </details>
  );
}

function EVocabMath({
  idx,
  token,
  E,
  seed,
  dModel,
}: {
  idx: number;
  token: string;
  E: number[][];
  seed: number;
  dModel: number;
}) {
  const row = E[idx] ?? [];
  const draws = idx * dModel; // # of PRNG draws skipped before this row
  const codeLines = [
    `// E is a [V=${E.length}, d_model=${dModel}] random matrix`,
    `// Each row is independent — token "${token}" is row ${idx}.`,
    `// E has no formula; the values come from a seeded PRNG.`,
    ``,
    `function mulberry32(seed) {`,
    `  let t = seed >>> 0;`,
    `  return () => {`,
    `    t = (t + 0x6d2b79f5) >>> 0;`,
    `    let r = t;`,
    `    r = Math.imul(r ^ (r >>> 15), r | 1);`,
    `    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);`,
    `    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;`,
    `  };`,
    `}`,
    ``,
    `const seed = ${seed};`,
    `const rand = mulberry32(seed);`,
    `// skip the first ${draws} draws — those filled rows 0..${idx - 1}`,
    `for (let k = 0; k < ${draws}; k++) rand();`,
    ``,
    `// the next ${dModel} draws fill E[idx=${idx}] (${token})`,
    `const E_row = [];`,
    `for (let i = 0; i < ${dModel}; i++) {`,
    `  E_row.push((rand() - 0.5) * 2);  // map [0,1) → [-1, 1)`,
    `}`,
    ``,
    `console.log(E_row);`,
    `// = [${row.map((v) => v.toFixed(4)).join(', ')}]`,
  ];
  const code = codeLines.join('\n');
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        padding: space.md,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <MathHeading
        level="div"
        latex={`E[\\text{idx}=${idx}]`}
        hint={`= "${token}" row`}
      />
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>i</th>
              <th style={{ textAlign: 'left', padding: '0.15rem 0.5rem' }}>source</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>value</th>
            </tr>
          </thead>
          <tbody>
            {row.map((v, i) => (
              <tr key={i}>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>{i}</td>
                <td style={{ padding: '0.15rem 0.5rem', color: color.text.secondary }}>
                  rand draw #{draws + i + 1} from mulberry32({seed})
                </td>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>
                  {v.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ flex: '1 1 32ch', minWidth: '32ch' }}>
          <details open>
            <summary style={{ cursor: 'pointer', color: color.text.emphasis, fontSize: '0.8rem' }}>
              JavaScript — paste into the browser console
            </summary>
            <pre
              style={{
                marginTop: space.sm,
                padding: space.md,
                background: color.text.primary,
                color: color.text.onDark,
                border: `1px solid ${color.text.primary}`,
                borderRadius: radius.sm,
                fontSize: '0.78rem',
                overflowX: 'auto',
                lineHeight: 1.45,
              }}
            >
              <code>{code}</code>
            </pre>
          </details>
        </div>
      </div>
      <p style={{ marginTop: space.md, color: color.text.secondary, fontSize: '0.8rem', maxWidth: '70ch' }}>
        Today these values are random noise. After training, gradients flow into row{' '}
        {idx} every time "{token}" appears in a batch — the row drifts toward a
        meaningful embedding. P[t] never gets that gradient (it's a fixed
        formula).
      </p>
    </div>
  );
}

function ELookupMath({
  t,
  token,
  idx,
  eRow,
  dModel,
}: {
  t: number;
  token: string;
  idx: number;
  eRow: number[];
  dModel: number;
}) {
  const codeLines = [
    `// E[idx_t] is just a row lookup — no formula, no computation.`,
    `// At t=${t} the token is "${token}", whose vocab id is ${idx}.`,
    ``,
    `const t = ${t};`,
    `const idx_t = ${idx};  // = ids[${t}], the vocab id of "${token}"`,
    ``,
    `// E is a [V, d_model] matrix. The lookup is one indexing operation:`,
    `const E_row_at_t = E[idx_t];`,
    ``,
    `// hard-coded values from this seed/d_model:`,
    `// E_row_at_t = [${eRow.map((v) => v.toFixed(4)).join(', ')}]`,
    ``,
    `console.log(E_row_at_t);`,
  ];
  const code = codeLines.join('\n');
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        padding: space.md,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <MathHeading
        level="div"
        latex={`E[\\text{idx}_t=${t}] = E[${idx}]`}
        hint={`= "${token}" row`}
      />
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>i</th>
              <th style={{ textAlign: 'left', padding: '0.15rem 0.5rem' }}>source</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>value</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: dModel }, (_, i) => (
              <tr key={i}>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>{i}</td>
                <td style={{ padding: '0.15rem 0.5rem', color: color.text.secondary }}>
                  E[{idx}][{i}]
                </td>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>
                  {eRow[i].toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ flex: '1 1 32ch', minWidth: '32ch' }}>
          <details open>
            <summary style={{ cursor: 'pointer', color: color.text.emphasis, fontSize: '0.8rem' }}>
              JavaScript — paste into the browser console
            </summary>
            <pre
              style={{
                marginTop: space.sm,
                padding: space.md,
                background: color.text.primary,
                color: color.text.onDark,
                border: `1px solid ${color.text.primary}`,
                borderRadius: radius.sm,
                fontSize: '0.78rem',
                overflowX: 'auto',
                lineHeight: 1.45,
              }}
            >
              <code>{code}</code>
            </pre>
          </details>
        </div>
      </div>
      <p style={{ marginTop: space.md, color: color.text.secondary, fontSize: '0.8rem', maxWidth: '70ch' }}>
        Same vocab id → same row. Every "{token}" in the corpus shares this
        exact vector. The position info comes from P[t], added in the next step.
      </p>
    </div>
  );
}

function HMath({
  t,
  token,
  idx,
  eRow,
  pRow,
  hRow,
  dModel,
}: {
  t: number;
  token: string;
  idx: number;
  eRow: number[];
  pRow: number[];
  hRow: number[];
  dModel: number;
}) {
  const codeLines = [
    `// h[t] = E[idx_t] + P[t]  (element-wise add)`,
    `// At t=${t}, token "${token}" (vocab idx ${idx}).`,
    ``,
    `const E_row = [${eRow.map((v) => v.toFixed(4)).join(', ')}];  // E[${idx}]`,
    `const P_row = [${pRow.map((v) => v.toFixed(4)).join(', ')}];  // P[${t}]`,
    ``,
    `const h = E_row.map((e, i) => e + P_row[i]);`,
    ``,
    `console.log(h);`,
    `// = [${hRow.map((v) => v.toFixed(4)).join(', ')}]`,
  ];
  const code = codeLines.join('\n');
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        padding: space.md,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      <MathHeading
        level="div"
        latex={`h[t=${t}] = E[${idx}] + P[${t}]`}
        hint={`("${token}" at position ${t})`}
      />
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <table style={{ borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>i</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>E[{idx}][i]</th>
              <th style={{ padding: '0.15rem 0.5rem' }}>+</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>P[{t}][i]</th>
              <th style={{ padding: '0.15rem 0.5rem' }}>=</th>
              <th style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>h[i]</th>
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: dModel }, (_, i) => (
              <tr key={i}>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>{i}</td>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>
                  {eRow[i].toFixed(4)}
                </td>
                <td style={{ padding: '0.15rem 0.5rem', color: color.text.secondary }}>+</td>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem' }}>
                  {pRow[i].toFixed(4)}
                </td>
                <td style={{ padding: '0.15rem 0.5rem', color: color.text.secondary }}>=</td>
                <td style={{ textAlign: 'right', padding: '0.15rem 0.5rem', fontWeight: 'bold' }}>
                  {hRow[i].toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ flex: '1 1 32ch', minWidth: '32ch' }}>
          <details open>
            <summary style={{ cursor: 'pointer', color: color.text.emphasis, fontSize: '0.8rem' }}>
              JavaScript — paste into the browser console
            </summary>
            <pre
              style={{
                marginTop: space.sm,
                padding: space.md,
                background: color.text.primary,
                color: color.text.onDark,
                border: `1px solid ${color.text.primary}`,
                borderRadius: radius.sm,
                fontSize: '0.78rem',
                overflowX: 'auto',
                lineHeight: 1.45,
              }}
            >
              <code>{code}</code>
            </pre>
          </details>
        </div>
      </div>
      <p style={{ marginTop: space.md, color: color.text.secondary, fontSize: '0.8rem', maxWidth: '70ch' }}>
        Same vocab + different t → different h. The "what" comes from E, the
        "where" comes from P, and the model sees them fused as one vector.
      </p>
    </div>
  );
}

export function CombinedInputPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const [dModel, setDModel] = useState<DModel>(8);
  const [seed, setSeed] = useState(1);
  const [sourceCode, setSourceCode] = useState<string | undefined>();
  const [error, setError] = useState<string | null>(null);
  const [pSelectedT, setPSelectedT] = useState<number>(0);
  const [eView, setEView] = useState<'vocab' | 'lookup'>('vocab');
  const [hHoverT, setHHoverT] = useState<number | null>(null);
  const [vocabSelectedIdx, setVocabSelectedIdx] = useState<number>(0);
  const [mathFocus, setMathFocus] = useState<'p' | 'eVocab' | 'eLookup' | 'h'>('p');

  const corpus = CORPUS_OPTIONS[corpusKey];
  const tokens = useMemo(() => tokenize(corpus), [corpus]);
  const ids = useMemo(() => tokenIds(tokens, VOCAB), [tokens]);

  const E = useMemo(
    () => randomMatrix(seed, VOCAB.length, dModel, 1),
    [seed, dModel]
  );
  const P = useMemo(() => sinusoidalP(tokens.length, dModel), [tokens.length, dModel]);

  const eRows = useMemo(() => ids.map((id) => E[id]), [ids, E]);
  const hRows = useMemo(
    () => eRows.map((eRow, t) => eRow.map((v, i) => v + P[t][i])),
    [eRows, P]
  );

  const eMax = useMemo(() => maxAbs(eRows), [eRows]);
  const pMax = useMemo(() => maxAbs(P), [P]);
  const hMax = useMemo(() => maxAbs(hRows), [hRows]);
  // One-shot fetch on mount to populate CodePanel source.
  useEffect(() => {
    let cancelled = false;
    api
      .combinedInput({ E, P, idx: 0, t: 0 })
      .then((res) => {
        if (!cancelled) setSourceCode(res.source_code);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
    // intentionally empty deps — fetch once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <h2>Combined Input — corpus view</h2>
      <FormulaDisplay latex={FORMULAS.combinedInput01} />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Controls</h3>
      <div style={{ display: 'flex', gap: space.xl, flexWrap: 'wrap', alignItems: 'center', margin: '0.5rem 0' }}>
        <label htmlFor="d-model-select">
          d_model:&nbsp;
          <Select
            id="d-model-select"
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
      </div>

      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        Vocab V={VOCAB.length}, T={tokens.length} tokens, d_model={dModel}.
        E is a seeded random [{VOCAB.length}, {dModel}] matrix.
        P is a sinusoidal [{tokens.length}, {dModel}] matrix.
        Each row below is one position in the corpus.
      </p>

      <TrainingCorpus corpusKey={corpusKey} onChange={setCorpusKey} />

      <div style={{ marginBottom: space.sm, fontSize: font.size.sm }}>
        <span style={{ marginRight: space.sm, color: color.text.secondary }}>E view:</span>
        <label style={{ marginRight: space.md }}>
          <input
            type="radio"
            name="e-view"
            checked={eView === 'vocab'}
            onChange={() => setEView('vocab')}
          />
          &nbsp;E (vocab, V={VOCAB.length})
        </label>
        <label>
          <input
            type="radio"
            name="e-view"
            checked={eView === 'lookup'}
            onChange={() => setEView('lookup')}
          />
          &nbsp;E[idx_t] for t=0..{tokens.length - 1}
        </label>
      </div>

      <div style={{ display: 'flex', gap: space.xl, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        {eView === 'vocab' ? (
          <CorpusMatrix
            titleLatex={String.raw`E\ (\text{vocab},\ V=${VOCAB.length})`}
            titleHint="click a row"
            testId="matrix-e-vocab"
            tokens={[...VOCAB]}
            matrix={E}
            maxAbs={eMax}
            dModel={dModel}
            indexLabel="idx"
            onRowClick={(idx) => {
              setVocabSelectedIdx(idx);
              setMathFocus('eVocab');
            }}
            selectedRow={mathFocus === 'eVocab' ? vocabSelectedIdx : (ids[pSelectedT] ?? null)}
            externalHighlightRows={hHoverT !== null ? [ids[hHoverT]] : []}
          />
        ) : (
          <CorpusMatrix
            titleLatex={String.raw`E[\text{idx}_t]\ \text{for}\ t=0..${tokens.length - 1}`}
            titleHint="click a row"
            testId="matrix-e-lookup"
            tokens={tokens}
            matrix={eRows}
            maxAbs={eMax}
            dModel={dModel}
            onRowClick={(t) => {
              setPSelectedT(t);
              setMathFocus('eLookup');
            }}
            selectedRow={pSelectedT}
            externalHighlightRows={hHoverT !== null ? [hHoverT] : []}
          />
        )}
        <CorpusMatrix
          titleLatex={FORMULAS.combinedInput02}
          titleHint="click a row"
          testId="matrix-p"
          tokens={tokens}
          matrix={P}
          maxAbs={pMax}
          dModel={dModel}
          onRowClick={(t) => {
            setPSelectedT(t);
            setMathFocus('p');
          }}
          selectedRow={pSelectedT}
          externalHighlightRows={hHoverT !== null ? [hHoverT] : []}
        />
        <CorpusMatrix
          titleLatex={FORMULAS.combinedInput03}
          titleHint="click a row"
          testId="matrix-h"
          tokens={tokens}
          matrix={hRows}
          maxAbs={hMax}
          dModel={dModel}
          onRowHover={setHHoverT}
          onRowClick={(t) => {
            setPSelectedT(t);
            setMathFocus('h');
          }}
          selectedRow={pSelectedT}
        />
      </div>

      <details open style={{ marginTop: space.lg }}>
        <summary style={{ cursor: 'pointer', color: color.text.emphasis, marginBottom: space.sm }}>
          {mathFocus === 'eVocab'
            ? `E[idx=${vocabSelectedIdx}] (${VOCAB[vocabSelectedIdx] ?? '?'}) math`
            : mathFocus === 'eLookup'
              ? `E[idx_t=${Math.min(pSelectedT, P.length - 1)}] (${tokens[Math.min(pSelectedT, P.length - 1)]}) math`
              : mathFocus === 'h'
                ? `h[t=${Math.min(pSelectedT, P.length - 1)}] math`
                : `P[t=${Math.min(pSelectedT, P.length - 1)}] math`}
        </summary>
        {mathFocus === 'eVocab' && (
          <EVocabMath
            idx={vocabSelectedIdx}
            token={VOCAB[vocabSelectedIdx] ?? '?'}
            E={E}
            seed={seed}
            dModel={dModel}
          />
        )}
        {mathFocus === 'eLookup' && (
          <ELookupMath
            t={Math.min(pSelectedT, P.length - 1)}
            token={tokens[Math.min(pSelectedT, P.length - 1)]}
            idx={ids[Math.min(pSelectedT, P.length - 1)]}
            eRow={eRows[Math.min(pSelectedT, P.length - 1)]}
            dModel={dModel}
          />
        )}
        {mathFocus === 'h' && (
          <HMath
            t={Math.min(pSelectedT, P.length - 1)}
            token={tokens[Math.min(pSelectedT, P.length - 1)]}
            idx={ids[Math.min(pSelectedT, P.length - 1)]}
            eRow={eRows[Math.min(pSelectedT, P.length - 1)]}
            pRow={P[Math.min(pSelectedT, P.length - 1)]}
            hRow={hRows[Math.min(pSelectedT, P.length - 1)]}
            dModel={dModel}
          />
        )}
        {mathFocus === 'p' && (
          <PositionMath
            t={Math.min(pSelectedT, P.length - 1)}
            dModel={dModel}
            P={P}
          />
        )}
      </details>

      <CodePanel source={sourceCode} />
    </div>
  );
}
