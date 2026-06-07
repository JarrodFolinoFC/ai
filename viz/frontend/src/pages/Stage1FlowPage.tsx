import { Flex } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { bigramCounts, trainingPairs, randomMatrix, softmaxRow, trainToConvergence } from '../funcs';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { TrainingCorpus } from '../components/TrainingCorpus';
import { Heatmap, probCellBg } from '../components/Heatmap';
import { OneStepTrainer } from '../components/OneStepTrainer';
import { WorkedExample } from '../components/WorkedExample';
import { RecentLoss } from '../components/RecentLoss';
import { color, space } from '../theme';
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
      prevRow: number[];
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

  const pairIdx = step === 0 ? 0 : (step - 1) % pairs.length;
  const currentPair = pairs[pairIdx];
  const trainedRows = useMemo(() => {
    const seen = new Set<number>();
    const n = Math.min(step, pairs.length);
    for (let k = 0; k < n; k++) seen.add(pairs[k].prev);
    return seen;
  }, [step, pairs]);
  const corpusTokens = useMemo(() => corpus.split(/\s+/).filter(Boolean), [corpus]);
  const currentLogits = W[currentPair.prev];
  const currentExps = currentLogits.map((v) => Math.exp(v));
  const currentExpSum = currentExps.reduce((a, b) => a + b, 0);
  const currentProbs = softmaxRow(currentLogits);
  const currentLoss = -Math.log(Math.max(currentProbs[currentPair.target], 1e-12));
  const prevLogits = flash ? flash.prevRow : currentLogits;
  const prevExps = prevLogits.map((v) => Math.exp(v));
  const prevExpSum = prevExps.reduce((a, b) => a + b, 0);
  const prevProbs = softmaxRow(prevLogits);
  const prevGrad = prevProbs.map((p, i) =>
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
    if (n === 1) {
      const idx = step % pairs.length;
      const pair = pairs[idx];
      const prev = pair.prev;
      setFlash({ row: prev, target: pair.target, pairIdx: idx, prevRow: W[prev].slice() });
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
      <Flex gap={space.xxl} wrap align="flex-start">
        <TrainingCorpus
          corpusKey={corpusKey}
          onChange={setCorpusKey}
          note="(changing this resets training)"
        />
      </Flex>
      <OneStepTrainer
        seed={seed}
        onSeedChange={setSeed}
        lr={lr}
        onLrChange={setLr}
        pairsLength={pairs.length}
        step={step}
        currentLoss={currentLoss}
        onStep={doStep}
        onReset={reset}
      />

      <WorkedExample
        vocab={VOCAB}
        step={step}
        flash={flash}
        pairsLength={pairs.length}
        corpusTokens={corpusTokens}
        currentPair={currentPair}
        prevLogits={prevLogits}
        prevExps={prevExps}
        prevExpSum={prevExpSum}
        prevProbs={prevProbs}
        prevGrad={prevGrad}
        currentLogits={currentLogits}
        currentExps={currentExps}
        currentExpSum={currentExpSum}
        currentProbs={currentProbs}
        W={W}
        softmaxW={softmaxW}
        empirical={empirical}
        errorMatrix={errorMatrix}
        rowErrors={rowErrors}
        totalError={totalError}
        trainedRows={trainedRows}
      />

      <h3 style={{ marginTop: space.xl }}>
        Corpus reference (fixed — derived from the counts above)
      </h3>
      <Flex gap="2rem" wrap align="flex-start">
        <Heatmap
          heading="Bigram counts (rows = prev, cols = next)"
          headingStyle={{ marginTop: 0 }}
          matrix={counts}
          vocab={VOCAB}
          formatValue={(c) => c}
          cellBackground={(c, i, j) =>
            VOCAB[i] === 'cat' && VOCAB[j] === 'on'
              ? color.highlightBg
              : c === 0
                ? color.bg.surface
                : 'transparent'}
        />

        <Heatmap
          heading="Softmax"
          subHeading="pᵢ = e^xᵢ / Σ e^x"
          headingStyle={{ marginBottom: '0.1rem' }}
          matrix={counts.map(softmaxRow)}
          vocab={VOCAB}
          cellBackground={probCellBg}
        />
      </Flex>

      <RecentLoss lossHistory={lossHistory} />

      <Flex gap="2rem" wrap>
        <Heatmap
          heading="W_final (raw logits)"
          matrix={wFinal}
          vocab={VOCAB}
          cellBackground={(v) => {
            const t = v / wFinalMaxAbs;
            return t >= 0
              ? `rgba(59, 130, 246, ${0.1 + 0.5 * t})`
              : `rgba(239, 68, 68, ${0.1 + 0.5 * -t})`;
          }}
        />

        <Heatmap
          heading="softmax(W_final) per row"
          matrix={softmaxWFinal}
          vocab={VOCAB}
          cellBackground={(p) => `rgba(34, 197, 94, ${0.1 + 0.6 * p})`}
        />
      </Flex>
    </div>
  );
}
