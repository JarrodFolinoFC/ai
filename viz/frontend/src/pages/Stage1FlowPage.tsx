import { useEffect, useMemo, useState } from 'react';
import { bigramCounts, trainingPairs, randomMatrix, softmaxRow, trainToConvergence } from '../funcs';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { TrainingCorpus } from '../components/TrainingCorpus';
import { ExplainMore } from '../components/ExplainMore';
import { color, space, radius, font } from '../theme';
import {VOCAB} from "../consts"

export function Stage1FlowPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  const corpus = CORPUS_OPTIONS[corpusKey];
  const { counts } = useMemo(
    () => bigramCounts(corpus, VOCAB),
    [corpus]
  );
  const [seed, setSeed] = useState(1);
  const [lr, setLr] = useState(0.5);
  const [W, setW] = useState(() => randomMatrix(1, VOCAB.length, VOCAB.length, 1));
  const [step, setStep] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [flash, setFlash] = useState<
    {
      row: number;
      target: number;
      pairIdx: number;
      tick: number;
      prevRow: number[];
      phase: 'old' | 'new';
    } | null
  >(null);

  const pairs = useMemo(() => trainingPairs(corpus, VOCAB), [corpus]);

  useEffect(() => {
    setW(randomMatrix(seed, VOCAB.length, VOCAB.length, 1));
    setStep(0);
    setLossHistory([]);
  }, [seed, corpusKey]);

  const softmaxW = useMemo(() => W.map(softmaxRow), [W]);

  const empirical = useMemo(
    () =>
      counts.map((row) => {
        const sum = row.reduce((a, b) => a + b, 0);
        return sum === 0 ? row.map(() => 0) : row.map((c) => c / sum);
      }),
    [counts]
  );

  const FINAL_EPOCHS = 500;
  const wFinal = useMemo(
    () =>
      trainToConvergence(
        randomMatrix(seed, VOCAB.length, VOCAB.length, 1),
        pairs,
        0.5,
        FINAL_EPOCHS
      ),
    [seed, pairs]
  );
  const softmaxWFinal = useMemo(() => wFinal.map(softmaxRow), [wFinal]);
  const wFinalMaxAbs = useMemo(
    () => Math.max(...wFinal.flat().map((v) => Math.abs(v)), 1e-9),
    [wFinal]
  );
  const finalRowErrors = useMemo(
    () =>
      softmaxWFinal.map(
        (row, i) =>
          row.reduce((a, b, j) => a + Math.abs(b - empirical[i][j]), 0) / 2
      ),
    [softmaxWFinal, empirical]
  );
  const finalMeanError = useMemo(
    () => finalRowErrors.reduce((a, b) => a + b, 0) / finalRowErrors.length,
    [finalRowErrors]
  );

  const errorMatrix = useMemo(
    () => softmaxW.map((row, i) => row.map((p, j) => p - empirical[i][j])),
    [softmaxW, empirical]
  );

  const rowErrors = useMemo(
    () =>
      errorMatrix.map((row) => row.reduce((a, b) => a + Math.abs(b), 0) / 2),
    [errorMatrix]
  );

  const totalError = useMemo(
    () => rowErrors.reduce((a, b) => a + b, 0) / rowErrors.length,
    [rowErrors]
  );

  const pairIdx = step % pairs.length;
  const currentPair = pairs[pairIdx];
  // Rows that have received at least one gradient update (appeared as `prev` in
  // a processed pair). Untrained rows stay greyed at their random init — rows
  // that never appear as `prev` never get a gradient and so never light up.
  const trainedRows = useMemo(() => {
    const seen = new Set<number>();
    const n = Math.min(step, pairs.length);
    for (let k = 0; k < n; k++) seen.add(pairs[k].prev);
    return seen;
  }, [step, pairs]);
  // Black box around the single cell being updated (trained row × target column).
  const cellBox = (i: number, j: number) =>
    flash?.row === i && flash?.target === j
      ? { border: `2px solid ${color.text.primary}` }
      : null;
  // Header word tints matching the corpus tracker: the trained row's label is
  // "a" (prev, warning) and the target column's label is "b" (next, emphasis).
  // Only while a step is flashed.
  const aHead = (i: number) =>
    flash?.row === i
      ? {
          background: color.highlightBg,
          color: color.text.emphasis,
          fontWeight: 'bold' as const,
        }
      : null;
  const bHead = (j: number) =>
    flash?.target === j
      ? {
          background: color.info.border,
          color: color.text.emphasis,
          fontWeight: 'bold' as const,
        }
      : null;
  const corpusTokens = useMemo(() => corpus.split(/\s+/).filter(Boolean), [corpus]);
  const currentLogits = W[currentPair.prev];
  const currentExps = currentLogits.map((v) => Math.exp(v));
  const currentExpSum = currentExps.reduce((a, b) => a + b, 0);
  const currentProbs = softmaxRow(currentLogits);
  const currentLoss = -Math.log(Math.max(currentProbs[currentPair.target], 1e-12));
  const currentGrad = currentProbs.map((p, i) =>
    i === currentPair.target ? p - 1 : p
  );

  function doStep(n = 1) {
    const next = W.map((r) => r.slice());
    const newLosses: number[] = [];
    for (let k = 0; k < n; k++) {
      const pair = pairs[(step + k) % pairs.length];
      const row = next[pair.prev];
      const probs = softmaxRow(row);
      newLosses.push(-Math.log(Math.max(probs[pair.target], 1e-12)));
      const grad = probs.map((p, i) => (i === pair.target ? p - 1 : p));
      next[pair.prev] = row.map((v, i) => v - lr * grad[i]);
    }
    setW(next);
    setLossHistory((h) => [...h, ...newLosses].slice(-1000));
    setStep((s) => s + n);
    // Only the single-step button animates: flash the row showing the OLD
    // numbers, hold ~2s, then fade to the new numbers. Batch steps would strobe.
    if (n === 1) {
      const idx = step % pairs.length;
      const pair = pairs[idx];
      const prev = pair.prev;
      const tick = Date.now();
      setFlash({ row: prev, target: pair.target, pairIdx: idx, tick, prevRow: W[prev].slice(), phase: 'old' });
      setTimeout(() => {
        setFlash((f) => (f && f.tick === tick ? { ...f, phase: 'new' } : f));
      }, 2000);
    } else {
      setFlash(null);
    }
  }

  function reset() {
    setW(randomMatrix(seed, VOCAB.length, VOCAB.length, 1));
    setStep(0);
    setLossHistory([]);
    setFlash(null);
  }

  return (
    <div>
      <h2>Bigram Forward + Backward</h2>

      <div
        style={{
          display: 'flex',
          gap: space.xxl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <TrainingCorpus
          corpusKey={corpusKey}
          onChange={setCorpusKey}
          note="(changing this resets training)"
        />
      </div>
      <h3>One-step trainer</h3>
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
          lr:&nbsp;
          <input
            type="number"
            step={0.05}
            min={0}
            value={lr}
            onChange={(e) => setLr(Number(e.target.value))}
            style={{ width: '5rem' }}
          />
        </label>
        <button type="button" onClick={() => doStep(1)}>
          step ×1
        </button>
        <button type="button" onClick={() => doStep(10)}>
          step ×10
        </button>
        <button type="button" onClick={() => doStep(pairs.length)}>
          1 epoch ({pairs.length})
        </button>
        <button type="button" onClick={() => doStep(pairs.length * 50)}>
          50 epochs
        </button>
        <button type="button" onClick={reset}>
          reset
        </button>
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          step={step} &nbsp; epoch={Math.floor(step / pairs.length)} &nbsp;
          loss={currentLoss.toFixed(3)}
        </span>
      </div>

      <div style={{ maxWidth: font.prose, margin: '0.25rem 0 0.75rem' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontFamily: 'monospace',
            fontSize: '0.8rem',
            color: color.text.secondary,
            marginBottom: '0.2rem',
          }}
        >
          <span>
            epoch {Math.floor(step / pairs.length)} progress
          </span>
          <span>
            {step % pairs.length}/{pairs.length} pairs
          </span>
        </div>
        <div
          style={{
            height: '0.6rem',
            background: color.bg.disabled,
            borderRadius: '999px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${((step % pairs.length) / pairs.length) * 100}%`,
              height: '100%',
              background: color.positive,
              transition: 'width 0.15s ease',
            }}
          />
        </div>
      </div>

      <div
        style={{
          display: 'flex',
          gap: space.xl,
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
      <div
        style={{
          flex: '0 1 auto',
          padding: space.md,
          background: color.bg.subtle,
          border: `1px solid ${color.border.default}`,
          borderRadius: radius.sm,
          fontFamily: 'monospace',
          fontSize: font.size.sm,
          lineHeight: 1.5,
        }}
      >
        <div
          style={{
            display: 'flex',
            gap: '1.25rem',
            flexWrap: 'wrap',
            alignItems: 'flex-start',
          }}
        >
          <div style={{ whiteSpace: 'nowrap' }}>
            next pair:
            <br />
            prev=<strong style={{ color: color.warning.fg }}>
              {VOCAB[currentPair.prev]}
            </strong>{' '}
            ({currentPair.prev})
            <br />
            target=<strong style={{ color: color.text.emphasis }}>
              {VOCAB[currentPair.target]}
            </strong>{' '}
            ({currentPair.target})
          </div>
          <div style={{ flex: '1 1 28ch', minWidth: '20ch' }}>
            <div style={{ color: color.text.secondary, marginBottom: space.sm }}>
              corpus position (
              {flash ? `pair ${flash.pairIdx} of ${pairs.length}` : 'not started yet'}
              ):{' '}
              <span style={{ color: color.warning.fg, fontWeight: 'bold' }}>
                a = prev (input)
              </span>
              ,{' '}
              <span style={{ color: color.text.emphasis, fontWeight: 'bold' }}>
                b = next (target)
              </span>
            </div>
            <div style={{ lineHeight: 2.4 }}>
              {corpusTokens.map((t, i) => {
                const markIdx = flash ? flash.pairIdx : -1;
                const isPrev = i === markIdx;
                const isTarget = i === markIdx + 1;
                const marked = isPrev || isTarget;
                return (
                  <span
                    key={i}
                    style={{
                      position: 'relative',
                      display: 'inline-block',
                      padding: '0.05rem 0.25rem',
                      borderRadius: '3px',
                      background: isPrev
                        ? color.highlightBg
                        : isTarget
                          ? color.info.border
                          : 'transparent',
                      color: marked ? color.text.emphasis : color.text.muted,
                      fontWeight: marked ? 'bold' : 'normal',
                      marginRight: space.xs,
                    }}
                  >
                    {marked && (
                      <span
                        style={{
                          position: 'absolute',
                          top: '-1rem',
                          left: '50%',
                          transform: 'translateX(-50%)',
                          fontSize: '0.75rem',
                          fontStyle: 'italic',
                          color: isPrev ? color.warning.fg : color.text.emphasis,
                        }}
                      >
                        {isPrev ? 'a' : 'b'}
                      </span>
                    )}
                    {t}
                  </span>
                );
              })}
            </div>
          </div>
          <div
            style={{
              flex: '1 1 30ch',
              minWidth: '26ch',
              ...(step === 0 && { opacity: 0.4, filter: 'grayscale(1)' }),
            }}
          >
        <div style={{ marginTop: 0, color: color.text.secondary }}>
          forward pass: <code>p = softmax(W[prev])</code> — read row{' '}
          <span
            style={{
              background: color.highlightBg,
              color: color.text.emphasis,
              fontWeight: 'bold',
              padding: '0 0.2rem',
              borderRadius: radius.sm,
            }}
          >
            {VOCAB[currentPair.prev]}
          </span>{' '}
          of W, exponentiate each logit, then divide by the total:
        </div>
        <table
          style={{
            borderCollapse: 'collapse',
            marginTop: '0.4rem',
            fontFamily: 'monospace',
            fontSize: font.size.sm,
          }}
        >
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ padding: '0.15rem 0.5rem', textAlign: 'left' }}>
                next
              </th>
              <th style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                x = W[
                <span
                  style={{
                    background: color.highlightBg,
                    color: color.text.emphasis,
                    fontWeight: 'bold',
                    padding: '0 0.2rem',
                    borderRadius: radius.sm,
                  }}
                >
                  {VOCAB[currentPair.prev]}
                </span>
                ]
              </th>
              <th style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                e^x
              </th>
              <th style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                p = e^x / Σ
              </th>
            </tr>
          </thead>
          <tbody>
            {currentLogits.map((x, i) => {
              const isTarget = i === currentPair.target;
              return (
                <tr
                  key={i}
                  style={{
                    background: isTarget ? color.highlightBg : 'transparent',
                    fontWeight: isTarget ? 'bold' : 'normal',
                  }}
                >
                  <td style={{ padding: '0.15rem 0.5rem' }}>{VOCAB[i]}</td>
                  <td style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                    {x >= 0 ? '+' : ''}
                    {x.toFixed(2)}
                  </td>
                  <td style={{ padding: '0.15rem 0.5rem', textAlign: 'right', color: color.text.muted }}>
                    {currentExps[i] >= 100
                      ? currentExps[i].toFixed(0)
                      : currentExps[i].toFixed(2)}
                  </td>
                  <td
                    style={{
                      padding: '0.15rem 0.5rem',
                      textAlign: 'right',
                      color: isTarget ? color.text.emphasis : color.text.secondary,
                    }}
                  >
                    {currentProbs[i].toFixed(2)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ color: color.text.muted, borderTop: `1px solid ${color.border.default}` }}>
              <td style={{ padding: '0.15rem 0.5rem' }}>Σ</td>
              <td style={{ padding: '0.15rem 0.5rem' }} />
              <td style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                {currentExpSum >= 100
                  ? currentExpSum.toFixed(0)
                  : currentExpSum.toFixed(2)}
              </td>
              <td style={{ padding: '0.15rem 0.5rem', textAlign: 'right' }}>
                1.00
              </td>
            </tr>
          </tbody>
        </table>
          </div>
          <div
            style={{
              flex: '1 1 24ch',
              minWidth: '20ch',
              ...(step === 0 && { opacity: 0.4, filter: 'grayscale(1)' }),
            }}
          >
        <div style={{ marginTop: 0, color: color.text.secondary }}>
          <code>y</code> = one-hot at the target (<strong>
            {VOCAB[currentPair.target]}
          </strong>
          ): 1 there, 0 elsewhere:
        </div>
        <div style={{ marginTop: '0.15rem' }}>
          y = [
          {currentProbs.map((_, i) => {
            const isTarget = i === currentPair.target;
            return (
              <span
                key={i}
                style={{
                  fontWeight: isTarget ? 'bold' : 'normal',
                  color: isTarget ? color.text.emphasis : color.text.muted,
                }}
              >
                {isTarget ? '1.00' : '0.00'}
                {i < currentProbs.length - 1 ? ', ' : ''}
              </span>
            );
          })}
          ]
        </div>
        <div style={{ marginTop: '0.15rem' }}>
          p − y = [
          {currentGrad.map((g, i) => (
            <span
              key={i}
              style={{
                color: g < 0 ? color.error.fg : color.text.secondary,
                fontWeight: i === currentPair.target ? 'bold' : 'normal',
              }}
            >
              {g >= 0 ? '+' : ''}
              {g.toFixed(2)}
              {i < currentGrad.length - 1 ? ', ' : ''}
            </span>
          ))}
          ]
        </div>
        <div style={{ marginTop: '0.35rem', color: color.text.secondary, fontSize: '0.8rem' }}>
          The target entry is <code>p − 1</code> (negative → push that logit
          <strong> up</strong>); every other entry is just <code>p</code>{' '}
          (positive → push those <strong>down</strong>).
        </div>
          </div>
        </div>
      </div>

      <div style={{ flex: '1 1 auto' }}>
        <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
        <div>
          <h4>W (raw logits)</h4>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0 0.5rem' }}></th>
                <th
                  colSpan={VOCAB.length}
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 'normal',
                    fontStyle: 'italic',
                    color: color.text.emphasis,
                    paddingBottom: '0.1rem',
                  }}
                >
                  next = b →
                </th>
              </tr>
              <tr>
                <th
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontStyle: 'italic',
                    color: color.warning.fg,
                    textAlign: 'right',
                  }}
                >
                  prev=a ↓
                </th>
                {VOCAB.map((w, j) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                      ...bHead(j),
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {W.map((row, i) => (
                <tr key={VOCAB[i]}>
                  <th
                    style={{
                      textAlign: 'right',
                      padding: '0.25rem 0.5rem',
                      borderRight: `1px solid ${color.border.strong}`,
                      ...aHead(i),
                    }}
                  >
                    {VOCAB[i]}
                  </th>
                  {row.map((v, j) => {
                    const flashing = flash?.row === i;
                    const dir = flashing
                      ? Math.sign(v - flash.prevRow[j])
                      : 0;
                    const showOld = flashing && flash.phase === 'old';
                    return (
                      <td
                        key={
                          flashing ? `${j}-${flash.tick}-${flash.phase}` : j
                        }
                        style={{
                          position: 'relative',
                          padding: '0.25rem 0.5rem',
                          textAlign: 'right',
                          background: trainedRows.has(i) ? undefined : color.bg.disabled,
                          animation: showOld
                            ? 'wflash 2s ease-out'
                            : undefined,
                          ...cellBox(i, j),
                        }}
                      >
                        {flashing && dir !== 0 && (
                          <span
                            style={{
                              position: 'absolute',
                              top: '-0.1rem',
                              left: '0.15rem',
                              fontSize: '0.7rem',
                              fontWeight: 'bold',
                              color: dir > 0 ? color.success.fg : color.error.fg,
                              animation: undefined,
                            }}
                          >
                            {dir > 0 ? '▲' : '▼'}
                          </span>
                        )}
                        {flashing ? (
                          <span
                            style={{
                              display: 'inline-block',
                              animation:
                                flash.phase === 'new'
                                  ? 'numFade 0.5s ease-out'
                                  : undefined,
                            }}
                          >
                            {(showOld ? flash.prevRow[j] : v).toFixed(2)}
                          </span>
                        ) : (
                          v.toFixed(2)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h4>softmax(W) per row</h4>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0 0.5rem' }}></th>
                <th
                  colSpan={VOCAB.length}
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 'normal',
                    fontStyle: 'italic',
                    color: color.text.emphasis,
                    paddingBottom: '0.1rem',
                  }}
                >
                  next = b →
                </th>
              </tr>
              <tr>
                <th
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    fontStyle: 'italic',
                    color: color.warning.fg,
                    textAlign: 'right',
                  }}
                >
                  prev=a ↓
                </th>
                {VOCAB.map((w, j) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                      ...bHead(j),
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {softmaxW.map((row, i) => (
                <tr key={VOCAB[i]}>
                  <th
                    style={{
                      textAlign: 'right',
                      padding: '0.25rem 0.5rem',
                      borderRight: `1px solid ${color.border.strong}`,
                      ...aHead(i),
                    }}
                  >
                    {VOCAB[i]}
                  </th>
                  {row.map((p, j) => {
                    const flashing = flash?.row === i;
                    const oldProb = flashing ? softmaxRow(flash.prevRow)[j] : p;
                    const dir = flashing ? Math.sign(p - oldProb) : 0;
                    const showOld = flashing && flash.phase === 'old';
                    return (
                    <td
                      key={flashing ? `${j}-${flash.tick}-${flash.phase}` : j}
                      style={{
                        position: 'relative',
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: trainedRows.has(i) ? undefined : color.bg.disabled,
                        animation: showOld ? 'wflash 2s ease-out' : undefined,
                        ...cellBox(i, j),
                      }}
                    >
                      {flashing && dir !== 0 && (
                        <span
                          style={{
                            position: 'absolute',
                            top: '-0.1rem',
                            left: '0.15rem',
                            fontSize: '0.7rem',
                            fontWeight: 'bold',
                            color: dir > 0 ? color.success.fg : color.error.fg,
                            animation: undefined,
                          }}
                        >
                          {dir > 0 ? '▲' : '▼'}
                        </span>
                      )}
                      {flashing ? (
                        <span
                          style={{
                            display: 'inline-block',
                            animation:
                              flash.phase === 'new'
                                ? 'numFade 0.5s ease-out'
                                : undefined,
                          }}
                        >
                          {(showOld ? oldProb : p).toFixed(2)}
                        </span>
                      ) : (
                        p.toFixed(2)
                      )}
                    </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h4>Empirical (target)</h4>
          <code style={{ fontSize: font.size.sm, color: color.text.secondary }}>
            pᵢ = countᵢ / Σ count
          </code>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0.25rem 0.5rem' }}></th>
                {VOCAB.map((w) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {counts.map((row, i) => {
                const rowSum = row.reduce((a, b) => a + b, 0);
                return (
                  <tr key={VOCAB[i]}>
                    <th
                      style={{
                        textAlign: 'right',
                        padding: '0.25rem 0.5rem',
                        borderRight: `1px solid ${color.border.strong}`,
                      }}
                    >
                      {VOCAB[i]}
                    </th>
                    {row.map((c, j) => {
                      const p = rowSum === 0 ? 0 : c / rowSum;
                      return (
                        <td
                          key={j}
                          style={{
                            padding: '0.25rem 0.5rem',
                            textAlign: 'right',
                            background:
                              VOCAB[i] === 'cat' && VOCAB[j] === 'on'
                                ? color.highlightBg
                                : p === 0
                                  ? color.bg.surface
                                  : `rgba(34, 197, 94, ${0.1 + 0.6 * p})`,
                          }}
                        >
                          {p.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div>
      <h4>Per-row convergence: |softmax(W) − empirical|</h4>
      <table
        style={{
          borderCollapse: 'collapse',
          fontFamily: 'monospace',
          fontSize: font.size.md,
        }}
      >
        <thead>
          <tr>
            <th style={{ padding: '0 0.5rem' }}></th>
            <th
              colSpan={VOCAB.length}
              style={{
                fontSize: '0.75rem',
                fontWeight: 'normal',
                fontStyle: 'italic',
                color: color.text.emphasis,
                paddingBottom: '0.1rem',
              }}
            >
              next = b →
            </th>
            <th style={{ padding: '0 0.5rem' }}></th>
          </tr>
          <tr>
            <th
              style={{
                padding: '0.25rem 0.5rem',
                fontSize: '0.75rem',
                fontStyle: 'italic',
                color: color.warning.fg,
                textAlign: 'right',
              }}
            >
              prev=a ↓
            </th>
            {VOCAB.map((w, j) => (
              <th
                key={w}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderBottom: `1px solid ${color.border.strong}`,
                  ...bHead(j),
                }}
              >
                {w}
              </th>
            ))}
            <th
              style={{
                padding: '0.25rem 0.5rem',
                borderBottom: `1px solid ${color.border.strong}`,
                borderLeft: `2px solid ${color.border.strong}`,
              }}
            >
              row error
            </th>
          </tr>
        </thead>
        <tbody>
          {errorMatrix.map((row, i) => {
            const flashing = flash?.row === i;
            const oldProbs = flashing ? softmaxRow(flash.prevRow) : null;
            const showOld = flashing && flash.phase === 'old';
            const oldRowError = oldProbs
              ? oldProbs.reduce((acc, op, j) => acc + Math.abs(op - empirical[i][j]), 0) / 2
              : rowErrors[i];
            return (
            <tr key={VOCAB[i]}>
              <th
                style={{
                  textAlign: 'right',
                  padding: '0.25rem 0.5rem',
                  borderRight: `1px solid ${color.border.strong}`,
                  ...aHead(i),
                }}
              >
                {VOCAB[i]}
              </th>
              {row.map((d, j) => {
                const oldD = oldProbs ? oldProbs[j] - empirical[i][j] : d;
                const dir = flashing ? Math.sign(d - oldD) : 0;
                const shown = showOld ? oldD : d;
                return (
                  <td
                    key={flashing ? `${j}-${flash.tick}-${flash.phase}` : j}
                    style={{
                      position: 'relative',
                      padding: '0.25rem 0.5rem',
                      textAlign: 'right',
                      background: trainedRows.has(i) ? undefined : color.bg.disabled,
                      color: color.text.secondary,
                      animation: showOld ? 'wflash 2s ease-out' : undefined,
                      ...cellBox(i, j),
                    }}
                  >
                    {flashing && dir !== 0 && (
                      <span
                        style={{
                          position: 'absolute',
                          top: '-0.1rem',
                          left: '0.15rem',
                          fontSize: '0.7rem',
                          fontWeight: 'bold',
                          color: dir > 0 ? color.success.fg : color.error.fg,
                          animation: undefined,
                        }}
                      >
                        {dir > 0 ? '▲' : '▼'}
                      </span>
                    )}
                    {flashing ? (
                      <span
                        style={{
                          display: 'inline-block',
                          animation:
                            flash.phase === 'new' ? 'numFade 0.5s ease-out' : undefined,
                        }}
                      >
                        {shown >= 0 ? '+' : ''}
                        {shown.toFixed(2)}
                      </span>
                    ) : (
                      <>
                        {d >= 0 ? '+' : ''}
                        {d.toFixed(2)}
                      </>
                    )}
                  </td>
                );
              })}
              <td
                style={{
                  position: 'relative',
                  padding: '0.25rem 0.5rem',
                  textAlign: 'right',
                  borderLeft: `2px solid ${color.border.strong}`,
                  fontWeight: 'bold',
                  background: trainedRows.has(i) ? undefined : color.bg.disabled,
                  color: color.text.secondary,
                  animation: showOld ? 'wflash 2s ease-out' : undefined,
                }}
              >
                {flashing ? (
                  <span
                    style={{
                      display: 'inline-block',
                      animation:
                        flash.phase === 'new' ? 'numFade 0.5s ease-out' : undefined,
                    }}
                  >
                    {(showOld ? oldRowError : rowErrors[i]).toFixed(3)}
                  </span>
                ) : (
                  rowErrors[i].toFixed(3)
                )}
              </td>
            </tr>
            );
          })}
          <tr>
            <th
              style={{
                textAlign: 'right',
                padding: '0.25rem 0.5rem',
                borderTop: `2px solid ${color.border.strong}`,
                borderRight: `1px solid ${color.border.strong}`,
              }}
            >
              mean
            </th>
            <td
              colSpan={VOCAB.length}
              style={{
                padding: '0.25rem 0.5rem',
                borderTop: `2px solid ${color.border.strong}`,
                color: color.text.secondary,
                fontStyle: 'italic',
              }}
            >
              average row error
            </td>
            <td
              style={{
                padding: '0.25rem 0.5rem',
                textAlign: 'right',
                borderTop: `2px solid ${color.border.strong}`,
                borderLeft: `2px solid ${color.border.strong}`,
                fontWeight: 'bold',
              }}
            >
              {totalError.toFixed(3)}
            </td>
          </tr>
        </tbody>
      </table>
        </div>
        </div>
      </div>
      </div>

      <h3 style={{ marginTop: space.xl }}>
        Corpus reference (fixed — derived from the counts above)
      </h3>
      <div
        style={{
          display: 'flex',
          gap: '2rem',
          flexWrap: 'wrap',
          alignItems: 'flex-start',
        }}
      >
        <div>
          <h4 style={{ marginTop: 0 }}>Bigram counts (rows = prev, cols = next)</h4>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0.25rem 0.5rem' }}></th>
                {VOCAB.map((w) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {counts.map((row, i) => (
                <tr key={VOCAB[i]}>
                  <th
                    style={{
                      textAlign: 'right',
                      padding: '0.25rem 0.5rem',
                      borderRight: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {VOCAB[i]}
                  </th>
                  {row.map((c, j) => (
                    <td
                      key={j}
                      style={{
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background:
                          VOCAB[i] === 'cat' && VOCAB[j] === 'on'
                            ? color.highlightBg
                            : c === 0
                              ? color.bg.surface
                              : 'transparent',
                      }}
                    >
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h4 style={{ marginBottom: '0.1rem' }}>Softmax</h4>
          <code style={{ fontSize: font.size.sm, color: color.text.secondary }}>
            pᵢ = e^xᵢ / Σ e^x
          </code>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0.25rem 0.5rem' }}></th>
                {VOCAB.map((w) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {counts.map((row, i) => {
                const sm = softmaxRow(row);
                return (
                  <tr key={VOCAB[i]}>
                    <th
                      style={{
                        textAlign: 'right',
                        padding: '0.25rem 0.5rem',
                        borderRight: `1px solid ${color.border.strong}`,
                      }}
                    >
                      {VOCAB[i]}
                    </th>
                    {sm.map((p, j) => (
                      <td
                        key={j}
                        style={{
                          padding: '0.25rem 0.5rem',
                          textAlign: 'right',
                          background:
                            VOCAB[i] === 'cat' && VOCAB[j] === 'on'
                              ? color.highlightBg
                              : `rgba(34, 197, 94, ${0.1 + 0.6 * p})`,
                        }}
                      >
                        {p.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {lossHistory.length > 0 && (
        <p style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          recent mean loss (last {Math.min(lossHistory.length, 50)}):{' '}
          {(
            lossHistory.slice(-50).reduce((a, b) => a + b, 0) /
            Math.min(lossHistory.length, 50)
          ).toFixed(3)}
        </p>
      )}

      <ExplainMore>
        At step 0 every row of <code>softmax(W)</code> is roughly uniform.
        Click <em>1 epoch</em> a few times and watch row <code>on</code>{' '}
        collapse onto column <code>the</code> (since every <code>on</code> in
        the corpus is followed by <code>the</code>), and row <code>cat</code>{' '}
        spread mass roughly <code>0.6</code> on <code>on</code>,{' '}
        <code>0.2</code> on <code>the</code>, <code>0.2</code> on{' '}
        <code>in</code> — matching the empirical table. The neural model
        rediscovers count-and-normalize one gradient step at a time.
      </ExplainMore>

      <h3>
        Final state: W after {FINAL_EPOCHS} epochs (lr=0.5, seed={seed})
      </h3>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        Reference: what the same random init converges to after{' '}
        {FINAL_EPOCHS} epochs of full-batch SGD. Compare{' '}
        <code>softmax(W_final)</code> below against the empirical table — they
        should match to within {finalMeanError < 0.02 ? '~0.01' : '~0.05'} per
        cell. The logits in <code>W_final</code> aren't unique (softmax is
        invariant to a per-row constant) but their <em>differences</em> within
        each row encode the empirical distribution.
      </p>
      <p style={{ fontFamily: 'monospace', color: color.text.secondary }}>
        final mean row error = {finalMeanError.toFixed(4)} &nbsp; (per row:{' '}
        {finalRowErrors.map((e) => e.toFixed(3)).join(', ')})
      </p>
      <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
        <div>
          <h4>W_final (raw logits)</h4>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0.25rem 0.5rem' }}></th>
                {VOCAB.map((w) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {wFinal.map((row, i) => (
                <tr key={VOCAB[i]}>
                  <th
                    style={{
                      textAlign: 'right',
                      padding: '0.25rem 0.5rem',
                      borderRight: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {VOCAB[i]}
                  </th>
                  {row.map((v, j) => {
                    const t = v / wFinalMaxAbs;
                    const bg =
                      t >= 0
                        ? `rgba(59, 130, 246, ${0.1 + 0.5 * t})`
                        : `rgba(239, 68, 68, ${0.1 + 0.5 * -t})`;
                    return (
                      <td
                        key={j}
                        style={{
                          padding: '0.25rem 0.5rem',
                          textAlign: 'right',
                          background: bg,
                        }}
                      >
                        {v.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h4>softmax(W_final) per row</h4>
          <table
            style={{
              borderCollapse: 'collapse',
              fontFamily: 'monospace',
              fontSize: font.size.md,
            }}
          >
            <thead>
              <tr>
                <th style={{ padding: '0.25rem 0.5rem' }}></th>
                {VOCAB.map((w) => (
                  <th
                    key={w}
                    style={{
                      padding: '0.25rem 0.5rem',
                      borderBottom: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {w}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {softmaxWFinal.map((row, i) => (
                <tr key={VOCAB[i]}>
                  <th
                    style={{
                      textAlign: 'right',
                      padding: '0.25rem 0.5rem',
                      borderRight: `1px solid ${color.border.strong}`,
                    }}
                  >
                    {VOCAB[i]}
                  </th>
                  {row.map((p, j) => (
                    <td
                      key={j}
                      style={{
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: `rgba(34, 197, 94, ${0.1 + 0.6 * p})`,
                      }}
                    >
                      {p.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, fontSize: font.size.sm }}>
        This is the "answer key" for the live trainer above. The per-row
        errors here ({finalRowErrors.every((e) => e < 0.02) ? 'all under 0.02' : 'shown above'}) tell
        you the floor SGD can reach with this corpus, lr, and epoch budget —
        any residual error is just incomplete convergence, not a model-class
        limit.
      </p>
    </div>
  );
}
