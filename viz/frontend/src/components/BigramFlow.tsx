import { Flex } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { randomMatrix, softmaxRow } from '../funcs';
import { TrainingCorpus } from './TrainingCorpus';
import { Heatmap } from './Heatmap';
import { OneStepTrainer } from './OneStepTrainer';
import { GradientColumn } from './GradientColumn';
import { FlashMatrixTable } from './FlashMatrixTable';
import { RowConvergenceTable } from './RowConvergenceTable';
import { RecentLoss } from './RecentLoss';
import { WeightUpdate } from './WeightUpdate';
import { WeightUpdateCode } from './WeightUpdateCode';
import { RowSoftmax } from './RowSoftmax';
import { GapToFinal } from './GapToFinal';
import { DistanceToFinal } from './DistanceToFinal';
import { PredictionsVsFinal } from './PredictionsVsFinal';
import { PreviousWeights } from './PreviousWeights';
import { BigramTrainer, type Flash } from '../bigramTrainer';
import { divergingColormap } from '../colormaps';
import { color, space } from '../theme';
import { StepContext } from '../stepContext';

interface BigramFlowProps {
  vocab: readonly string[];
  corpus: string;
}

export function BigramFlow({ vocab, corpus }: BigramFlowProps) {
  const [seed, setSeed] = useState(1);
  const [lr, setLr] = useState(0.5);
  const [W, setW] = useState(() => randomMatrix(1, vocab.length, vocab.length, 1));
  const [step, setStep] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [flash, setFlash] = useState<Flash | null>(null);

  const { pairs, empirical } = useMemo(
    () => BigramTrainer.corpusStats(corpus, vocab),
    [corpus, vocab]
  );
  const trainer = useMemo(() => new BigramTrainer(W, pairs, step), [W, pairs, step]);
  const { wFinal, softmaxWFinal } = useMemo(
    () => BigramTrainer.trainReference(seed, vocab, pairs),
    [seed, vocab, pairs]
  );
  // Colour scale for the live trained weights shown by the inference card.
  const wMaxAbs = useMemo(
    () => Math.max(1e-6, ...W.flat().map((v) => Math.abs(v))),
    [W]
  );
  // The seeded initial weights — the 0%-converged baseline for distance-to-final.
  const wInitial = useMemo(
    () => randomMatrix(seed, vocab.length, vocab.length, 1),
    [seed, vocab]
  );
  const { softmaxW, errorMatrix, rowErrors, totalError } = useMemo(
    () => trainer.analysis(empirical),
    [trainer, empirical]
  );

  const {
    currentPair, trainedRows, currentProbs, currentLoss,
    prevLogits, prevExps, prevExpSum, prevProbs, prevGrad,
  } = useMemo(
    () => trainer.derive(flash),
    [trainer, flash]
  );

  // Show the most recently recorded training loss so the readout matches the
  // "Recent mean loss" history (both are training-curve values). Before any
  // step has run, fall back to the loss of the initial pair against initial W.
  const displayLoss =
    lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : currentLoss;

  useEffect(() => {
    setW(randomMatrix(seed, vocab.length, vocab.length, 1));
    setStep(0);
    setLossHistory([]);
  }, [seed, corpus, vocab]);

  function doStep(n = 1) {
    const { W: nextW, step: nextStep, lossHistory: nextLossHistory, flash: nextFlash } =
      trainer.doStep(n, lr, lossHistory);
    setW(nextW);
    setStep(nextStep);
    setLossHistory(nextLossHistory);
    setFlash(nextFlash);
  }

  function reset() {
    setW(randomMatrix(seed, vocab.length, vocab.length, 1));
    setStep(0);
    setLossHistory([]);
    setFlash(null);
  }

  return (
    <StepContext.Provider value={step}>
    <Flex wrap align="stretch" gap={space.lg}>
      {/* ── INPUT ── */}
      {/* The corpus, tokenized into (prev → target) pairs. */}
      <TrainingCorpus corpus={corpus}
                      pairIdx={flash ? flash.pairIdx : null}
                      pairsLength={pairs.length} currentPair={currentPair} vocab={vocab} started={step !== 0}
      />
      {/* Static corpus statistics (currently disabled). */}
      {/* <Heatmap heading="Bigram counts (rows = prev, cols = next)" matrix={counts} vocab={vocab} formatValue={(c) => c} */}
      {/*     cellBackground={(c, i, j) => */}
      {/*       vocab[i] === 'cat' && vocab[j] === 'on' */}
      {/*         ? color.highlightBg */}
      {/*         : c === 0 */}
      {/*           ? color.bg.surface */}
      {/*           : 'transparent'} */}
      {/*   /> */}
      {/* <Heatmap heading="Empirical (target)" subHeading={<FormulaDisplay inline latex={`p_i = \\frac{\\text{count}_i}{\\sum \\text{count}}`} />} matrix={empirical} vocab={vocab} cellBackground={probCellBg} /> */}
      {/* <Heatmap heading="Softmax" subHeading={<FormulaDisplay inline latex={`p_i = \\frac{e^{x_i}}{\\sum e^{x}}`} />} matrix={counts.map(softmaxRow)} vocab={vocab} cellBackground={probCellBg} /> */}

      {/* ── PER-STEP CHAIN: prev W → forward → loss → gradient → update → new W ── */}
      {/* Chain 1. W as it stood at the previous step — the chain's first link (badge: step − 1). */}
      <PreviousWeights vocab={vocab} W={W} flash={flash} step={step} />
      {/* Chain 2. Forward pass on the previous-step row (x = W_prev[prev], e^x, p) — the p that drives the gradient. */}
      <RowSoftmax
        vocab={vocab} prevToken={vocab[currentPair.prev]} target={currentPair.target}
        logits={prevLogits} exps={prevExps} expSum={prevExpSum} probs={prevProbs}
      />
      {/* <ForwardPassSection */}
      {/*   vocab={vocab} targetIdx={currentPair.target} prevToken={vocab[currentPair.prev]} */}
      {/*   before={{ logits: prevLogits, exps: prevExps, expSum: prevExpSum, probs: prevProbs }} */}
      {/*   current={{ logits: currentLogits, exps: currentExps, expSum: currentExpSum, probs: currentProbs }} */}
      {/* /> */}
      {/* Chain 3. The full prediction matrix: softmax applied to every row of W. */}
      <FlashMatrixTable heading="softmax(W) per row" matrix={softmaxW} prevTransform={softmaxRow} vocab={vocab} trainedRows={trainedRows} flash={flash} live />
      {/* Chain 4. The loss from the prediction vs the target, averaged over recent steps. */}
      <RecentLoss lossHistory={lossHistory} />
      {/* Chain 5. Backward pass: the gradient p − y that drives the weight update. */}
      <GradientColumn vocab={vocab} target={currentPair.target} prevProbs={prevProbs} prevGrad={prevGrad} dimmed={step === 0} />
      {/* Chain 6. The SGD update applied to the prev row: Wₐ ← Wₐ − η(p − y). */}
      <WeightUpdate vocab={vocab} lr={lr} flash={flash} W={W} prevGrad={prevGrad} dimmed={step === 0} />
      {/* Chain 6b. The same update as runnable JavaScript, expandable to a full-screen runner. */}
      <WeightUpdateCode
        vocab={vocab} lr={lr}
        rowLabel={vocab[flash ? flash.row : currentPair.prev]}
        row={flash ? flash.prevRow : W[currentPair.prev]}
        probs={flash ? prevProbs : currentProbs}
        target={flash ? flash.target : currentPair.target}
        started={step !== 0}
      />
      {/* Chain 7. The resulting weights after this step: W (raw logits) — becomes next step's chain start. */}
      <FlashMatrixTable heading="W (raw logits)" matrix={W} prevTransform={(row) => row} vocab={vocab} trainedRows={trainedRows} flash={flash} live />

      {/* ── EVALUATION / REFERENCE ── */}
      {/* How close the current predictions are to the empirical target. */}
      <RowConvergenceTable errorMatrix={errorMatrix} empirical={empirical} rowErrors={rowErrors} totalError={totalError} vocab={vocab} trainedRows={trainedRows} flash={flash} />
      {/* <Heatmap */}
      {/*     heading="softmax(W_final) per row" */}
      {/*     matrix={softmaxWFinal} */}
      {/*     vocab={vocab} */}
      {/*     cellBackground={sequentialColormap([34, 197, 94])} */}
      {/* /> */}
      {/* Per-step gap each weight still has to close: W_final − W. */}
      <GapToFinal W={W} wFinal={wFinal} vocab={vocab} />
      {/* Per-step scalar distance to the converged model + % converged. */}
      <DistanceToFinal W={W} wFinal={wFinal} wInitial={wInitial} />
      {/* Current predictions side-by-side with the converged predictions. */}
      <PredictionsVsFinal softmaxW={softmaxW} softmaxWFinal={softmaxWFinal} vocab={vocab} />
      {/* The live trained model used at inference; only trained rows are un-greyed. */}
      <Heatmap heading="W (trained — used at inference)" matrix={W} vocab={vocab} cellBackground={divergingColormap(wMaxAbs)} trainedRows={trainedRows} live />
    </Flex>
    {/* Controls float in the bottom-right corner so they stay reachable while scrolling. */}
    <div
      style={{
        position: 'fixed',
        bottom: space.lg,
        right: space.lg,
        zIndex: 1000,
        width: '22rem',
        maxWidth: '90vw',
        maxHeight: '85vh',
        overflowY: 'auto',
        background: color.bg.surface,
        borderRadius: '8px',
        boxShadow: '0 6px 24px rgba(0, 0, 0, 0.18)',
      }}
    >
      <OneStepTrainer
        seed={seed} onSeedChange={setSeed} lr={lr}
        onLrChange={setLr} pairsLength={pairs.length} step={step}
        currentLoss={displayLoss} onStep={doStep} onReset={reset}
      />
    </div>
    </StepContext.Provider>
  );
}
