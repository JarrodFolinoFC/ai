import { Select } from 'antd';
import { useMemo, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { TrainingCorpus } from '../components/TrainingCorpus';

import { color, space, radius, font } from '../theme';
import {
  randomMatrix,
  sinusoidalP,
  tokenize,
  tokenIds,
  lowerTriMask,
  rowNormalize,
  matMul,
  transpose,
  softmaxRow,
  addVec,
  matVecRight,
} from '../funcs';
import { shadeDiverging, shadeProb } from '../presentationFunctions';

const VOCAB = ['the', 'cat', 'dog', 'in', 'on', 'hat', 'sofa', '.'] as const;
const N_EMBD = 4;
const T_MAX = 8;

const SENTENCE = ['the', 'cat', 'on', 'the', 'sofa'] as const;

const D_MODEL_OPTIONS = [4, 8, 16] as const;
type DModel = (typeof D_MODEL_OPTIONS)[number];

function MatrixTable({
  data,
  rowLabels,
  colLabels,
  highlightRow,
  highlightCol,
  shade,
  onRowClick,
  indexLabel,
  rowTokens,
}: {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightRow?: number;
  highlightCol?: number;
  shade: (v: number) => string;
  onRowClick?: (i: number) => void;
  indexLabel?: string;
  rowTokens?: string[];
}) {
  return (
    <table
      style={{
        borderCollapse: 'collapse',
        fontFamily: 'monospace',
        fontSize: font.size.sm,
      }}
    >
      <thead>
        <tr>
          {indexLabel ? (
            <th
              style={{
                padding: '0.2rem 0.4rem',
                borderBottom: `1px solid ${color.border.strong}`,
              }}
            >
              {indexLabel}
            </th>
          ) : null}
          {rowTokens ? (
            <th
              style={{
                padding: '0.2rem 0.4rem',
                borderBottom: `1px solid ${color.border.strong}`,
              }}
            >
              token
            </th>
          ) : (
            <th style={{ padding: '0.2rem 0.4rem' }}></th>
          )}
          {colLabels.map((c, j) => (
            <th
              key={j}
              style={{
                padding: '0.2rem 0.4rem',
                borderBottom: `1px solid ${color.border.strong}`,
                background: highlightCol === j ? color.highlightBg : undefined,
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
            onClick={onRowClick ? () => onRowClick(i) : undefined}
            style={{
              outline: highlightRow === i ? `2px solid ${color.highlight}` : undefined,
              cursor: onRowClick ? 'pointer' : undefined,
            }}
          >
            {indexLabel ? (
              <th
                style={{
                  padding: '0.2rem 0.4rem',
                  borderRight: `1px solid ${color.border.strong}`,
                  textAlign: 'right',
                  background: highlightRow === i ? color.highlightBg : undefined,
                }}
              >
                {i}
              </th>
            ) : null}
            <th
              style={{
                padding: '0.2rem 0.4rem',
                borderRight: `1px solid ${color.border.strong}`,
                textAlign: rowTokens ? 'left' : 'right',
                background: highlightRow === i ? color.highlightBg : undefined,
                whiteSpace: 'nowrap',
              }}
            >
              {rowTokens ? rowTokens[i] : rowLabels[i]}
            </th>
            {row.map((v, j) => (
              <td
                key={j}
                style={{
                  padding: '0.2rem 0.4rem',
                  textAlign: 'right',
                  background: shade(v),
                }}
              >
                {v.toFixed(2)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Sym({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <span
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      tabIndex={0}
      style={{
        position: 'relative',
        borderBottom: `1px dotted ${color.text.secondary}`,
        cursor: 'help',
        outline: 'none',
      }}
    >
      {children}
      {open && (
        <span
          role="tooltip"
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 6px)',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 50,
            padding: '0.4rem 0.6rem',
            background: color.text.primary,
            color: color.text.onDark,
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            fontSize: '0.78rem',
            fontWeight: 'normal',
            lineHeight: 1.35,
            whiteSpace: 'normal',
            width: 'max-content',
            maxWidth: '24ch',
            borderRadius: radius.sm,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)',
            pointerEvents: 'none',
          }}
        >
          {title}
          <span
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '5px solid transparent',
              borderRight: '5px solid transparent',
              borderTop: `5px solid ${color.text.primary}`,
            }}
          />
        </span>
      )}
    </span>
  );
}

function MathPanel({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontFamily: 'monospace',
        fontSize: font.size.sm,
        padding: space.md,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        lineHeight: 1.6,
      }}
    >
      {children}
    </div>
  );
}

function CodePanelInline({ code }: { code: string }) {
  return (
    <pre
      style={{
        margin: 0,
        padding: space.md,
        background: color.text.primary,
        color: color.text.onDark,
        border: `1px solid ${color.text.primary}`,
        borderRadius: radius.sm,
        fontSize: '0.78rem',
        lineHeight: 1.45,
        overflowX: 'auto',
      }}
    >
      <code>{code}</code>
    </pre>
  );
}

function ThreeColPanel({
  table,
  math,
  code,
}: {
  table: ReactNode;
  math: ReactNode;
  code: ReactNode;
}) {
  return (
    <div
      style={{
        display: 'flex',
        gap: space.xl,
        flexWrap: 'wrap',
        alignItems: 'flex-start',
      }}
    >
      <div style={{ flex: '0 0 auto' }}>
        <h4 style={{ margin: '0 0 0.5rem' }}>Table</h4>
        {table}
      </div>
      <div style={{ flex: '0 0 auto', maxWidth: '34ch' }}>
        <h4 style={{ margin: '0 0 0.5rem' }}>Math</h4>
        {math}
      </div>
      <div style={{ flex: '1 1 36ch', minWidth: '32ch' }}>
        <h4 style={{ margin: '0 0 0.5rem' }}>JavaScript</h4>
        {code}
      </div>
    </div>
  );
}

function ProbsTable({
  data,
  rowLabels,
  colLabels,
  highlightRow,
  onRowClick,
}: {
  data: number[][];
  rowLabels: string[];
  colLabels: string[];
  highlightRow?: number;
  onRowClick?: (i: number) => void;
}) {
  return (
    <table
      style={{
        borderCollapse: 'collapse',
        fontFamily: 'monospace',
        fontSize: font.size.sm,
      }}
    >
      <thead>
        <tr>
          <th style={{ padding: '0.2rem 0.4rem' }}></th>
          {colLabels.map((c, j) => (
            <th
              key={j}
              style={{
                padding: '0.2rem 0.4rem',
                borderBottom: `1px solid ${color.border.strong}`,
              }}
            >
              {c}
            </th>
          ))}
          <th
            style={{
              padding: '0.2rem 0.4rem',
              borderBottom: `1px solid ${color.border.strong}`,
              borderLeft: `1px solid ${color.border.strong}`,
            }}
          >
            argmax
          </th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => {
          const maxIdx = row.indexOf(Math.max(...row));
          return (
            <tr
              key={i}
              onClick={onRowClick ? () => onRowClick(i) : undefined}
              style={{
                outline:
                  highlightRow === i ? `2px solid ${color.highlight}` : undefined,
                cursor: onRowClick ? 'pointer' : undefined,
              }}
            >
              <th
                style={{
                  padding: '0.2rem 0.4rem',
                  borderRight: `1px solid ${color.border.strong}`,
                  textAlign: 'right',
                  whiteSpace: 'nowrap',
                  background: highlightRow === i ? color.highlightBg : undefined,
                }}
              >
                {rowLabels[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '0.2rem 0.4rem',
                    textAlign: 'right',
                    background: shadeProb(v),
                    fontWeight: j === maxIdx ? 'bold' : 'normal',
                    outline: j === maxIdx ? `2px solid ${color.highlight}` : undefined,
                    outlineOffset: j === maxIdx ? '-2px' : undefined,
                  }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
              <td
                style={{
                  padding: '0.2rem 0.4rem',
                  borderLeft: `1px solid ${color.border.strong}`,
                  fontWeight: 'bold',
                }}
              >
                {colLabels[maxIdx]}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function VectorRow({
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
              }}
            >
              {v.toFixed(2)}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

export function Stage2FlowPage() {
  const [seed, setSeed] = useState(7);
  const [t, setT] = useState(2);

  // Corpus-wide pipeline state
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const [dModel, setDModel] = useState<DModel>(8);
  const [corpusSeed, setCorpusSeed] = useState(1);
  const [selectedT, setSelectedT] = useState<number>(0);
  const [selectedVocabIdx, setSelectedVocabIdx] = useState<number>(0);

  const tokenWord = SENTENCE[t] ?? SENTENCE[SENTENCE.length - 1];
  const tokenIdx = VOCAB.indexOf(tokenWord);

  const E = useMemo(() => randomMatrix(seed, VOCAB.length, N_EMBD, 1), [seed]);
  const P = useMemo(
    () => randomMatrix(seed + 17, T_MAX, N_EMBD, 0.5),
    [seed]
  );
  const W_head = useMemo(
    () => randomMatrix(seed + 31, N_EMBD, VOCAB.length, 1),
    [seed]
  );
  const b_head = useMemo(
    () => Array<number>(VOCAB.length).fill(0),
    []
  );

  const eRow = E[tokenIdx];
  const pRow = P[t];
  const h = useMemo(() => addVec(eRow, pRow), [eRow, pRow]);
  const logits = useMemo(
    () => matVecRight(W_head, h, b_head),
    [W_head, h, b_head]
  );
  const probs = useMemo(() => softmaxRow(logits), [logits]);

  const eMaxAbs = Math.max(...E.flat().map((v) => Math.abs(v)), 1e-9);
  const pMaxAbs = Math.max(...P.flat().map((v) => Math.abs(v)), 1e-9);
  const wMaxAbs = Math.max(...W_head.flat().map((v) => Math.abs(v)), 1e-9);
  const hMaxAbs = Math.max(...h.map((v) => Math.abs(v)), 1e-9);
  const logitMaxAbs = Math.max(...logits.map((v) => Math.abs(v)), 1e-9);

  const dimLabels = Array.from({ length: N_EMBD }, (_, k) => `d${k}`);
  const posLabels = Array.from({ length: T_MAX }, (_, i) => `t=${i}`);

  // ─── Corpus-wide pipeline ─────────────────────────────────────────────
  const corpus = CORPUS_OPTIONS[corpusKey];
  const cTokens = useMemo(() => tokenize(corpus), [corpus]);
  const cIds = useMemo(() => tokenIds(cTokens, VOCAB), [cTokens]);
  const cT = cTokens.length;

  const cE = useMemo(
    () => randomMatrix(corpusSeed, VOCAB.length, dModel, 1),
    [corpusSeed, dModel]
  );
  const cP = useMemo(() => sinusoidalP(cT, dModel), [cT, dModel]);
  const cH = useMemo(
    () => cIds.map((id, i) => addVec(cE[id], cP[i])),
    [cIds, cE, cP]
  );
  const cMask = useMemo(() => lowerTriMask(cT), [cT]);
  const cW = useMemo(() => rowNormalize(cMask), [cMask]);
  const cHMix = useMemo(() => matMul(cW, cH), [cW, cH]);
  const cWHead = useMemo(() => transpose(cE), [cE]); // tied: W_head = E^T
  const cLogits = useMemo(() => matMul(cHMix, cWHead), [cHMix, cWHead]);
  const cProbs = useMemo(() => cLogits.map((row) => softmaxRow(row)), [cLogits]);

  const cHMaxAbs = Math.max(...cH.flat().map((v) => Math.abs(v)), 1e-9);
  const cHMixMaxAbs = Math.max(
    ...cHMix.flat().map((v) => Math.abs(v)),
    1e-9
  );
  const cLogitMaxAbs = Math.max(
    ...cLogits.flat().map((v) => Math.abs(v)),
    1e-9
  );
  const cDimLabels = Array.from({ length: dModel }, (_, k) => `d${k}`);
  const cPosLabels = cTokens.map((w, i) => `${i} ${w}`);

  return (
    <div>
      <h2>Stage 2a Flow — Token + Position → Logits</h2>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        End-to-end forward pass through the Stage 2a stack on one position.
        Token embedding (<Link to="/token-embed">/token-embed</Link>) and
        position embedding (<Link to="/position-embed">/position-embed</Link>)
        are added (<Link to="/combined-input">/combined-input</Link>) to give a
        per-position hidden vector <code>h</code>, then projected back to vocab
        logits via the unembed head (
        <Link to="/unembed-head">/unembed-head</Link>) and turned into
        probabilities by <Link to="/softmax">softmax</Link>.
      </p>

      <h3>Setup</h3>
      <div style={{ fontFamily: 'monospace', marginBottom: space.sm }}>
        vocab = [{VOCAB.join(', ')}] &nbsp; n_embd = {N_EMBD} &nbsp; T_max ={' '}
        {T_MAX}
      </div>
      <div style={{ fontFamily: 'monospace', marginBottom: space.sm }}>
        sentence = "{SENTENCE.join(' ')}"
      </div>
      <div
        style={{
          margin: '0.5rem 0',
          display: 'flex',
          gap: space.lg,
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
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
          position t:&nbsp;
          <input
            type="number"
            min={0}
            max={SENTENCE.length - 1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
            style={{ width: '4rem' }}
          />
        </label>
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          → token = <strong>{tokenWord}</strong> (id {tokenIdx})
        </span>
      </div>

      <h3>Parameter shapes</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        Three hyperparameters determine every shape in the model:{' '}
        <code>V = {VOCAB.length}</code> (vocab size),{' '}
        <code>n_embd = {N_EMBD}</code> (embedding dimension),{' '}
        <code>T_max = {T_MAX}</code> (max sequence length). Every dimension
        below traces back to one of those three — there are no other free
        choices.
      </p>
      <table
        style={{
          borderCollapse: 'collapse',
          fontFamily: 'monospace',
          fontSize: font.size.sm,
          marginBottom: space.sm,
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                padding: '0.25rem 0.6rem',
                borderBottom: `1px solid ${color.border.strong}`,
                textAlign: 'left',
              }}
            >
              term
            </th>
            <th
              style={{
                padding: '0.25rem 0.6rem',
                borderBottom: `1px solid ${color.border.strong}`,
                textAlign: 'left',
              }}
            >
              shape
            </th>
            <th
              style={{
                padding: '0.25rem 0.6rem',
                borderBottom: `1px solid ${color.border.strong}`,
                textAlign: 'left',
              }}
            >
              what it is
            </th>
            <th
              style={{
                padding: '0.25rem 0.6rem',
                borderBottom: `1px solid ${color.border.strong}`,
                textAlign: 'left',
              }}
            >
              why that count
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '0.25rem 0.6rem' }}>V · n_embd</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              {VOCAB.length} × {N_EMBD}
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              <code>E</code>, the token embedding table
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              one embedding row (n_embd numbers) per vocab token (V of them)
            </td>
          </tr>
          <tr>
            <td style={{ padding: '0.25rem 0.6rem' }}>T · n_embd</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              {T_MAX} × {N_EMBD}
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              <code>P</code>, the position embedding table
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              one embedding row per position slot (T of them)
            </td>
          </tr>
          <tr>
            <td style={{ padding: '0.25rem 0.6rem' }}>n_embd · V</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              {N_EMBD} × {VOCAB.length}
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              <code>W_head</code>, the unembed projection
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              maps a hidden vector (n_embd dims) to vocab logits (V of them)
            </td>
          </tr>
          <tr>
            <td style={{ padding: '0.25rem 0.6rem' }}>V</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>{VOCAB.length}</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              <code>b</code>, the bias added to logits
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              one bias scalar per vocab token
            </td>
          </tr>
          <tr style={{ borderTop: `2px solid ${color.border.strong}`, fontWeight: 'bold' }}>
            <td style={{ padding: '0.25rem 0.6rem' }}>total</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>—</td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              {VOCAB.length}·{N_EMBD} + {T_MAX}·{N_EMBD} + {N_EMBD}·
              {VOCAB.length} + {VOCAB.length}
            </td>
            <td style={{ padding: '0.25rem 0.6rem' }}>
              ={' '}
              {VOCAB.length * N_EMBD +
                T_MAX * N_EMBD +
                N_EMBD * VOCAB.length +
                VOCAB.length}{' '}
              params
            </td>
          </tr>
        </tbody>
      </table>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        How each shape is forced by dimensional algebra:
      </p>
      <ul style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.6 }}>
        <li>
          <code>E</code> width is <code>n_embd</code> because{' '}
          <em>you chose</em> the embedding dim. Height is <code>V</code>{' '}
          because there must be one row per vocab token.
        </li>
        <li>
          <code>P</code> width is also <code>n_embd</code> —{' '}
          <strong>forced</strong>, not chosen. Because <code>h = E[id] + P[t]</code>{' '}
          is element-wise, both vectors must have identical width.
        </li>
        <li>
          <code>W_head</code> shape is{' '}
          <strong>entirely forced</strong> by its surroundings: input width{' '}
          <code>n_embd</code> matches <code>h</code>, output width{' '}
          <code>V</code> matches the required logit shape. No design choice
          here.
        </li>
        <li>
          <code>b</code> shape <code>V</code> is forced — it's added to logits,
          which are <code>V</code>-shaped.
        </li>
      </ul>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, fontSize: font.size.sm }}>
        Bottom line: pick V, n_embd, T_max — and the rest of the model's shape
        graph falls out by dimensional necessity. This is the recurring
        pattern in transformer architectures: a few endpoint dimensions
        determine every layer's shape.
      </p>

      <h3>1. Token embedding table E ({VOCAB.length} × {N_EMBD})</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        Row <code>{tokenIdx}</code> is highlighted — that's{' '}
        <code>E[{tokenIdx}]</code>, the vector representation of token{' '}
        <strong>{tokenWord}</strong>.
      </p>
      <ThreeColPanel
        table={
          <MatrixTable
            data={E}
            rowLabels={VOCAB.map((w, i) => `${i} ${w}`)}
            colLabels={dimLabels}
            highlightRow={tokenIdx}
            shade={(v) => shadeDiverging(v, eMaxAbs)}
          />
        }
        math={
          <MathPanel>
            <div>
              <Sym title="Token embedding matrix"><strong>E</strong></Sym> ∈ ℝ
              <sup>
                <Sym title="Vocab size">{VOCAB.length}</Sym>×
                <Sym title="Embedding dimension (n_embd)">{N_EMBD}</Sym>
              </sup>
            </div>
            <div style={{ marginTop: space.sm }}>
              <Sym title="Token embedding matrix">E</Sym>[
              <Sym title="Vocab id (row index)">idx</Sym>][
              <Sym title="Embedding dimension index">i</Sym>] = (
              <Sym title="Uniform random draw in [0, 1)">u</Sym> − 0.5) · 2 · 1
              = 2<Sym title="Uniform random draw in [0, 1)">u</Sym> − 1
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              where <Sym title="Uniform random draw in [0, 1)">u</Sym> ∼{' '}
              <Sym title="Seeded pseudorandom generator (mulberry32)">
                mulberry32
              </Sym>
              (<Sym title="PRNG seed">seed</Sym>),{' '}
              <Sym title="Uniform random draw in [0, 1)">u</Sym> ∈ [0, 1)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              ⇒ <Sym title="Token embedding matrix">E</Sym>[idx][i] ∈ [−1, 1)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary, fontSize: '0.8rem' }}>
              One row per vocab token. Lookup at position{' '}
              <Sym title="Position in the sequence">t</Sym> is{' '}
              <Sym title="Token embedding matrix">E</Sym>[
              <Sym title="Vocab id of the token at position t">
                idx<sub>t</sub>
              </Sym>
              ] — pure indexing, no math.
            </div>
          </MathPanel>
        }
        code={
          <CodePanelInline
            code={`// (assumes makeSeededRandom is defined as in section 2)

const vocabSize = ${VOCAB.length}, embeddingDim = ${N_EMBD}, baseSeed = ${seed}, tokenScale = 1;

const nextRandom = makeSeededRandom(baseSeed);

const tokenEmbeddings = Array.from(
  { length: vocabSize },
  () =>
    Array.from(
      { length: embeddingDim },
      () => (nextRandom() - 0.5) * 2 * tokenScale
    )
);

const tokenIndex = ${tokenIdx};  // "${tokenWord}"
console.log(tokenEmbeddings[tokenIndex]);
// = [${E[tokenIdx].map((v) => v.toFixed(2)).join(', ')}]`}
          />
        }
      />
      <h4>E[{tokenIdx}]</h4>
      <VectorRow
        data={eRow}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, eMaxAbs)}
      />

      <h3>2. Position embedding table P ({T_MAX} × {N_EMBD})</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        Row <code>{t}</code> is highlighted — that's <code>P[{t}]</code>, the
        "where in the sequence" signal added on top of the token embedding.
        Here <code>P</code> is a seeded random matrix (the sinusoidal variant
        lives in the corpus pipeline below).
      </p>
      <ThreeColPanel
        table={
          <MatrixTable
            data={P}
            rowLabels={posLabels}
            colLabels={dimLabels}
            highlightRow={t}
            shade={(v) => shadeDiverging(v, pMaxAbs)}
          />
        }
        math={
          <MathPanel>
            <div>
              <Sym title="Position embedding matrix"><strong>P</strong></Sym> ∈
              ℝ
              <sup>
                <Sym title="Max sequence length (T_max)">{T_MAX}</Sym>×
                <Sym title="Embedding dimension (n_embd)">{N_EMBD}</Sym>
              </sup>
            </div>
            <div style={{ marginTop: space.sm }}>
              <Sym title="Position embedding matrix">P</Sym>[
              <Sym title="Position in the sequence (row index)">t</Sym>][
              <Sym title="Embedding dimension index">i</Sym>] = (
              <Sym title="Uniform random draw in [0, 1)">u</Sym> − 0.5) · 2 ·
              0.5 = <Sym title="Uniform random draw in [0, 1)">u</Sym> − 0.5
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              where <Sym title="Uniform random draw in [0, 1)">u</Sym> ∼{' '}
              <Sym title="Seeded pseudorandom generator (mulberry32)">
                mulberry32
              </Sym>
              (<Sym title="PRNG seed">seed</Sym> +{' '}
              <Sym title="Offset so P uses a different draw stream than E">17</Sym>
              ), <Sym title="Uniform random draw in [0, 1)">u</Sym> ∈ [0, 1)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              ⇒ <Sym title="Position embedding matrix">P</Sym>[t][i] ∈ [−0.5,
              0.5)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary, fontSize: '0.8rem' }}>
              Each scalar is one PRNG draw, scanned row-major (
              <Sym title="Position in the sequence">t</Sym> then{' '}
              <Sym title="Embedding dimension index">i</Sym>).
            </div>
          </MathPanel>
        }
        code={
          <CodePanelInline
            code={`function makeSeededRandom(seed) {
  let state = seed >>> 0;
  return function nextRandom() {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

const maxSequenceLength = ${T_MAX};
const embeddingDim = ${N_EMBD};
const baseSeed = ${seed};
const positionSeedOffset = 17;
const positionScale = 0.5;

const nextRandom = makeSeededRandom(baseSeed + positionSeedOffset);

const positionEmbeddings = Array.from(
  { length: maxSequenceLength },
  () =>
    Array.from(
      { length: embeddingDim },
      () => (nextRandom() - 0.5) * 2 * positionScale
    )
);

const positionIndex = ${t};
console.log(positionEmbeddings[positionIndex]);
// = [${P[t].map((v) => v.toFixed(2)).join(', ')}]`}
          />
        }
      />

      <h4>P[{t}]</h4>
      <VectorRow
        data={pRow}
        labels={dimLabels}
        shade={(v) => shadeDiverging(v, pMaxAbs)}
      />

      <h3>3. Combined input h = E[{tokenIdx}] + P[{t}]</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        Element-wise sum. The result is the hidden vector for position{' '}
        <code>{t}</code> — what the rest of the model operates on. In a real
        transformer this would feed into attention; here it goes straight to
        unembed.
      </p>
      <ThreeColPanel
        table={
          <VectorRow
            data={h}
            labels={dimLabels}
            shade={(v) => shadeDiverging(v, hMaxAbs)}
          />
        }
        math={
          <MathPanel>
            <div>
              <Sym title="Hidden vector for this position (input to the rest of the model)">
                <strong>h</strong>
              </Sym>{' '}
              ∈ ℝ
              <sup>
                <Sym title="Embedding dimension (n_embd)">{N_EMBD}</Sym>
              </sup>
            </div>
            <div style={{ marginTop: space.sm }}>
              <Sym title="Hidden vector">h</Sym>[
              <Sym title="Embedding dimension index">i</Sym>] ={' '}
              <Sym title="Token embedding matrix">E</Sym>[
              <Sym title="Vocab id of the token at position t">
                idx<sub>t</sub>
              </Sym>
              ][i] + <Sym title="Position embedding matrix">P</Sym>[
              <Sym title="Position in the sequence">t</Sym>][i]
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              for <Sym title="Embedding dimension index">i</Sym> = 0 ..{' '}
              {N_EMBD - 1}
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary, fontSize: '0.8rem' }}>
              Element-wise add. "What" (
              <Sym title="Token embedding matrix">E</Sym>) + "where" (
              <Sym title="Position embedding matrix">P</Sym>) fuse into one
              vector — the model never separates them again.
            </div>
          </MathPanel>
        }
        code={
          <CodePanelInline
            code={`const tokenEmbeddingRow = [${eRow.map((v) => v.toFixed(2)).join(', ')}];  // tokenEmbeddings[${tokenIdx}]
const positionEmbeddingRow = [${pRow.map((v) => v.toFixed(2)).join(', ')}];  // positionEmbeddings[${t}]

const hiddenVector = tokenEmbeddingRow.map(
  (tokenValue, dim) => tokenValue + positionEmbeddingRow[dim]
);

console.log(hiddenVector);
// = [${h.map((v) => v.toFixed(2)).join(', ')}]`}
          />
        }
      />

      <h3>4. Unembed head: logits = h · W_head + b</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        <code>W_head</code> is <code>{N_EMBD} × {VOCAB.length}</code> — it
        projects the {N_EMBD}-dimensional hidden vector back into the
        vocab-sized logit space. <code>b</code> is initialized to zero here.
      </p>
      <ThreeColPanel
        table={
          <div>
            <h5 style={{ margin: '0 0 0.25rem' }}>
              W_head ({N_EMBD} × {VOCAB.length})
            </h5>
            <MatrixTable
              data={W_head}
              rowLabels={dimLabels}
              colLabels={VOCAB.map((w) => w)}
              shade={(v) => shadeDiverging(v, wMaxAbs)}
            />
            <h5 style={{ margin: '0.5rem 0 0.25rem' }}>logits</h5>
            <VectorRow
              data={logits}
              labels={VOCAB.map((w) => w)}
              shade={(v) => shadeDiverging(v, logitMaxAbs)}
            />
          </div>
        }
        math={
          <MathPanel>
            <div>
              <Sym title="Raw scores over the vocabulary (one per token)">
                <strong>logits</strong>
              </Sym>{' '}
              ∈ ℝ
              <sup>
                <Sym title="Vocab size">{VOCAB.length}</Sym>
              </sup>
            </div>
            <div style={{ marginTop: space.sm }}>
              <Sym title="Raw scores over the vocabulary">logits</Sym>[
              <Sym title="Vocab id (output index)">v</Sym>] ={' '}
              <Sym title="Sum over embedding dimensions">Σ</Sym>
              <sub>
                <Sym title="Embedding dimension index">i</Sym>
              </sub>{' '}
              <Sym title="Hidden vector">h</Sym>[i] ·{' '}
              <Sym title="Unembedding (output projection) matrix [n_embd × V]">
                W_head
              </Sym>
              [i][v] +{' '}
              <Sym title="Output bias vector (length V)">b</Sym>[v]
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              for <Sym title="Vocab id">v</Sym> = 0 .. {VOCAB.length - 1}
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary }}>
              <Sym title="Output bias vector">b</Sym>[v] = 0 (initialized to
              zero)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary, fontSize: '0.8rem' }}>
              Vector × matrix → vector. Each output is a dot product of{' '}
              <Sym title="Hidden vector">h</Sym> with one column of{' '}
              <Sym title="Unembedding matrix">W_head</Sym>.
            </div>
          </MathPanel>
        }
        code={
          <CodePanelInline
            code={`const hiddenVector = [${h.map((v) => v.toFixed(2)).join(', ')}];
const vocabSize = ${VOCAB.length};
const embeddingDim = ${N_EMBD};
// unembedWeights is [embeddingDim × vocabSize]; bias starts at zeros
const bias = Array(vocabSize).fill(0);

const logits = Array.from({ length: vocabSize }, (_, vocabIdx) => {
  let dotProduct = 0;
  for (let dim = 0; dim < embeddingDim; dim++) {
    dotProduct += hiddenVector[dim] * unembedWeights[dim][vocabIdx];
  }
  return dotProduct + bias[vocabIdx];
});

console.log(logits);
// = [${logits.map((v) => v.toFixed(2)).join(', ')}]`}
          />
        }
      />

      <h3>5. Softmax → probabilities over next token</h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary }}>
        With random parameters the distribution is roughly uniform — the model
        hasn't learned anything. Training (Stage 1's loop, applied to{' '}
        <code>E</code>, <code>P</code>, <code>W_head</code>, <code>b</code>)
        would shape these probabilities toward the empirical bigram targets.
      </p>
      <ThreeColPanel
        table={
          <div>
            <VectorRow
              data={probs}
              labels={VOCAB.map((w) => w)}
              shade={shadeProb}
            />
            <p style={{ marginTop: space.sm, color: color.text.secondary, fontSize: font.size.sm }}>
              argmax →{' '}
              <strong>{VOCAB[probs.indexOf(Math.max(...probs))]}</strong>{' '}
              (id {probs.indexOf(Math.max(...probs))}). Sum ={' '}
              {probs.reduce((a, b) => a + b, 0).toFixed(6)}.
            </p>
          </div>
        }
        math={
          <MathPanel>
            <div>
              <Sym title="Probability distribution over the vocabulary">
                <strong>probs</strong>
              </Sym>{' '}
              ∈ ℝ
              <sup>
                <Sym title="Vocab size">{VOCAB.length}</Sym>
              </sup>
              , <Sym title="Sum across all vocab tokens">Σ</Sym> probs = 1
            </div>
            <div style={{ marginTop: space.sm }}>
              <Sym title="Probability distribution">probs</Sym>[
              <Sym title="Vocab id">v</Sym>] = exp(
              <Sym title="Raw scores over the vocabulary">logits</Sym>[v] −{' '}
              <Sym title="Max logit (subtracted for numerical stability)">m</Sym>
              ) /{' '}
              <Sym title="Normalizer: sum of all shifted exponentials">Z</Sym>
            </div>
            <div style={{ marginTop: space.xs, color: color.text.secondary }}>
              <Sym title="Max logit (subtracted for numerical stability)">m</Sym>{' '}
              = max
              <sub>
                <Sym title="Index running over all vocab ids">k</Sym>
              </sub>{' '}
              <Sym title="Raw scores">logits</Sym>[k] (numerical stability)
            </div>
            <div style={{ marginTop: space.xs, color: color.text.secondary }}>
              <Sym title="Normalizer: sum of all shifted exponentials">Z</Sym>{' '}
              = <Sym title="Sum over all vocab ids">Σ</Sym>
              <sub>
                <Sym title="Index running over all vocab ids">k</Sym>
              </sub>{' '}
              exp(<Sym title="Raw scores">logits</Sym>[k] −{' '}
              <Sym title="Max logit">m</Sym>)
            </div>
            <div style={{ marginTop: space.sm, color: color.text.secondary, fontSize: '0.8rem' }}>
              Subtracting <Sym title="Max logit">m</Sym> makes the largest
              exponent 0, avoiding overflow. The ratio is unchanged.
            </div>
          </MathPanel>
        }
        code={
          <CodePanelInline
            code={`const logits = [${logits.map((v) => v.toFixed(2)).join(', ')}];

const maxLogit = Math.max(...logits);
const shiftedExps = logits.map((logit) => Math.exp(logit - maxLogit));
const normalizer = shiftedExps.reduce((sum, e) => sum + e, 0);
const probabilities = shiftedExps.map((e) => e / normalizer);

console.log(probabilities);
// = [${probs.map((v) => v.toFixed(3)).join(', ')}]
// sum = ${probs.reduce((a, b) => a + b, 0).toFixed(6)}`}
          />
        }
      />

      <h3>What's missing (Stage 2b and beyond)</h3>
      <ul style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        <li>
          <strong>No context mixing yet.</strong> The hidden vector at position{' '}
          <code>{t}</code> only sees its own token + position. A real
          transformer would mix it with hidden vectors from earlier positions
          via attention — that's Stage 2b.
        </li>
        <li>
          <strong>Lower-tri mask + normalize-rows</strong> (
          <Link to="/lower-tri-mask">/lower-tri-mask</Link>,{' '}
          <Link to="/normalize-rows">/normalize-rows</Link>) build the
          causal-uniform-mixing matrix used in 2b. Stage 2a's stack here
          ignores them — they're prep for the next stage.
        </li>
        <li>
          <strong>Untied embeddings.</strong> Many implementations tie{' '}
          <code>W_head = E^T</code>; here they're independent so the unembed
          step is a separate trainable matrix.
        </li>
      </ul>

      <hr style={{ margin: '2rem 0', border: 0, borderTop: `1px solid ${color.border.strong}` }} />

      <h2>Corpus-wide pipeline</h2>
      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Same Stage 2 stack, but applied to a whole corpus at once and chained
        through the causal-mix matrices. After computing{' '}
        <code>H = E[idx_t] + P[t]</code> for every position (the{' '}
        <Link to="/combined-input">/combined-input</Link> step), build the
        causal-uniform-mixing matrix from{' '}
        <Link to="/lower-tri-mask">/lower-tri-mask</Link> and{' '}
        <Link to="/normalize-rows">/normalize-rows</Link>, mix to get{' '}
        <code>H'</code>, then unembed all rows at once via{' '}
        <Link to="/unembed-head">/unembed-head</Link> with{' '}
        <strong>tied weights</strong> (<code>W_head = E^T</code>).
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
        <label htmlFor="d-model-select-stage2">
          d_model:&nbsp;
          <Select
            id="d-model-select-stage2"
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
            value={corpusSeed}
            onChange={(e) => setCorpusSeed(Number(e.target.value))}
            style={{ width: '5rem' }}
          />
        </label>
      </div>

      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Vocab V={VOCAB.length}, T={cT} tokens, d_model={dModel}.
      </p>

      <TrainingCorpus
        corpusKey={corpusKey}
        onChange={setCorpusKey}
        name="corpus-size-stage2"
      />

      <h3>1. Combined input — pick a position</h3>
      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        Same as <Link to="/combined-input">/combined-input</Link>. Click a row
        in any of the three tables — it sets <strong>t = {selectedT}</strong>
        {' '}(token "<strong>{cTokens[selectedT] ?? '?'}</strong>", vocab id{' '}
        <strong>{cIds[selectedT] ?? '?'}</strong>), and that position is
        highlighted all the way through M → W → H' → logits → probs below.
      </p>
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
          marginBottom: space.lg,
        }}
      >
        <div>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            E (vocab, V={VOCAB.length})
            <span style={{ marginLeft: space.sm, fontSize: font.size.sm, color: color.text.muted }}>
              click a row
            </span>
          </h4>
          <MatrixTable
            data={cE}
            rowTokens={[...VOCAB]}
            indexLabel="idx"
            rowLabels={[...VOCAB]}
            colLabels={cDimLabels}
            shade={(v) => shadeDiverging(v, Math.max(...cE.flat().map((x) => Math.abs(x)), 1e-9))}
            highlightRow={selectedVocabIdx}
            onRowClick={(idx) => {
              setSelectedVocabIdx(idx);
              // jump t to the first occurrence of this vocab id
              const firstT = cIds.indexOf(idx);
              if (firstT >= 0) setSelectedT(firstT);
            }}
          />
        </div>
        <div>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            P[t]
            <span style={{ marginLeft: space.sm, fontSize: font.size.sm, color: color.text.muted }}>
              click a row
            </span>
          </h4>
          <MatrixTable
            data={cP}
            rowTokens={cTokens}
            indexLabel="t"
            rowLabels={cPosLabels}
            colLabels={cDimLabels}
            shade={(v) => shadeDiverging(v, Math.max(...cP.flat().map((x) => Math.abs(x)), 1e-9))}
            highlightRow={selectedT}
            onRowClick={(i) => {
              setSelectedT(i);
              setSelectedVocabIdx(cIds[i] ?? selectedVocabIdx);
            }}
          />
        </div>
        <div>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            h = E[idx<sub>t</sub>] + P[t]
            <span style={{ marginLeft: space.sm, fontSize: font.size.sm, color: color.text.muted }}>
              click a row
            </span>
          </h4>
          <MatrixTable
            data={cH}
            rowTokens={cTokens}
            indexLabel="t"
            rowLabels={cPosLabels}
            colLabels={cDimLabels}
            shade={(v) => shadeDiverging(v, cHMaxAbs)}
            highlightRow={selectedT}
            onRowClick={(i) => {
              setSelectedT(i);
              setSelectedVocabIdx(cIds[i] ?? selectedVocabIdx);
            }}
          />
        </div>
      </div>

      <p style={{ maxWidth: '70ch', color: color.text.secondary, lineHeight: 1.5 }}>
        <strong>2. Causal mix.</strong> Build a lower-tri mask{' '}
        <code>M</code> (<Link to="/lower-tri-mask">/lower-tri-mask</Link>),
        row-normalize it (<Link to="/normalize-rows">/normalize-rows</Link>)
        to get the {cT}×{cT} weight matrix <code>W</code>, then{' '}
        <code>H' = W · H</code>. The <code>T × T</code> matrices aren't shown
        here — they scale quadratically and the atomic pages cover them in
        isolation.
      </p>

      <h3>Steps 3 → 4 → 5: H' → logits → probs</h3>
      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <div style={{ flex: '0 0 auto' }}>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            3. Mixed hidden H' = W · H ({cT} × {dModel})
          </h4>
          <p style={{ maxWidth: '40ch', color: color.text.secondary, fontSize: font.size.sm }}>
            Each row of <code>H'</code> is the running average of{' '}
            <code>H</code> rows up to that position.{' '}
            <code>H'[{selectedT}]</code> mixes{' '}
            <code>h<sub>0</sub>..h<sub>{selectedT}</sub></code> with equal
            weight <code>1/{selectedT + 1}</code>.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <MatrixTable
              data={cHMix}
              rowTokens={cTokens}
              indexLabel="t"
              rowLabels={cPosLabels}
              colLabels={cDimLabels}
              shade={(v) => shadeDiverging(v, cHMixMaxAbs)}
              highlightRow={selectedT}
              onRowClick={(i) => {
                setSelectedT(i);
                setSelectedVocabIdx(cIds[i] ?? selectedVocabIdx);
              }}
            />
          </div>
        </div>

        <div style={{ flex: '0 0 auto' }}>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            4. logits = H' · E<sup>T</sup> ({cT} × {VOCAB.length}) —{' '}
            <Link to="/unembed-head">/unembed-head</Link>
          </h4>
          <p style={{ maxWidth: '40ch', color: color.text.secondary, fontSize: font.size.sm }}>
            Tied: <code>W_head = E<sup>T</sup></code>, so{' '}
            <code>logits[t, v] = H'[t] · E[v]</code>. Each logit is the dot
            product between <code>H'[t]</code> and vocab token{' '}
            <code>v</code>'s embedding row. <code>b = 0</code>.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <MatrixTable
              data={cLogits}
              rowTokens={cTokens}
              indexLabel="t"
              rowLabels={cPosLabels}
              colLabels={VOCAB.map((w) => w)}
              shade={(v) => shadeDiverging(v, cLogitMaxAbs)}
              highlightRow={selectedT}
              onRowClick={(i) => {
                setSelectedT(i);
                setSelectedVocabIdx(cIds[i] ?? selectedVocabIdx);
              }}
            />
          </div>
        </div>

        <div style={{ flex: '0 0 auto' }}>
          <h4 style={{ margin: '0 0 0.5rem' }}>
            5. probs = softmax(logits) ({cT} × {VOCAB.length}) —{' '}
            <Link to="/softmax">/softmax</Link>
          </h4>
          <p style={{ maxWidth: '40ch', color: color.text.secondary, fontSize: font.size.sm }}>
            Row-wise softmax. With random <code>E</code> it's nearly uniform —
            the model hasn't learned. The argmax column on the right is the
            predicted next token at each position.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <ProbsTable
              data={cProbs}
              rowLabels={cPosLabels.map(
                (l, i) => `${l} → ${cTokens[i + 1] ?? '·'}`
              )}
              colLabels={VOCAB.map((w) => w)}
              highlightRow={selectedT}
              onRowClick={(i) => {
                setSelectedT(i);
                setSelectedVocabIdx(cIds[i] ?? selectedVocabIdx);
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
