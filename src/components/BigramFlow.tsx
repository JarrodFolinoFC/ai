import { Flex } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { BigramTrainer, type Flash } from '../bigramTrainer';
import { stepColormap } from '../colormaps';
import { randomMatrix, softmaxRow } from '../funcs';
import { StepContext } from '../stepContext';
import { color, space } from '../theme';
import { AfterSoftmax } from './AfterSoftmax';
import { DistanceToFinal } from './DistanceToFinal';
import { FlashMatrixTable } from './FlashMatrixTable';
import { GapToFinal } from './GapToFinal';
import { GradientColumn } from './GradientColumn';
import { Heatmap } from './Heatmap';
import { OneStepTrainer } from './OneStepTrainer';
import { PredictionsVsFinal } from './PredictionsVsFinal';
import { PreviousSoftmax } from './PreviousSoftmax';
import { PreviousWeights } from './PreviousWeights';
import { RecentLoss } from './RecentLoss';
import { RowConvergenceTable } from './RowConvergenceTable';
import { RowSoftmax } from './RowSoftmax';
import { TrainingCorpus } from './TrainingCorpus';
import { WeightUpdate } from './WeightUpdate';
import { WeightUpdateCode } from './WeightUpdateCode';
import { TokenIdMap } from './TokenIdMap';

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
      <TokenIdMap vocab={vocab} />
      <TrainingCorpus corpus={corpus}
                      pairIdx={flash ? flash.pairIdx : null}
                      pairsLength={pairs.length} currentPair={currentPair} vocab={vocab} started={step !== 0}
      />
      <PreviousWeights vocab={vocab} W={W} flash={flash} step={step} usedRow={currentPair.prev} />
      <PreviousSoftmax vocab={vocab} W={W} flash={flash} step={step} usedRow={currentPair.prev} />
      <RowSoftmax vocab={vocab} currentPair={currentPair} logits={prevLogits} exps={prevExps} expSum={prevExpSum} probs={prevProbs} />
        <AfterSoftmax
          vocab={vocab} prevToken={vocab[currentPair.prev]} target={currentPair.target}
          row={W[currentPair.prev]} oldProbs={prevProbs}
        />
      <FlashMatrixTable heading="softmax(W) per row (after update)" matrix={softmaxW} prevTransform={softmaxRow} vocab={vocab} trainedRows={trainedRows} flash={flash} live />
      <RecentLoss lossHistory={lossHistory} />
      <GradientColumn vocab={vocab} target={currentPair.target} prevProbs={prevProbs} prevGrad={prevGrad} dimmed={step === 0} />
      <WeightUpdate vocab={vocab} lr={lr} flash={flash} W={W} prevGrad={prevGrad} dimmed={step === 0} />
      <WeightUpdateCode
        vocab={vocab} lr={lr}
        rowLabel={vocab[flash ? flash.row : currentPair.prev]}
        row={flash ? flash.prevRow : W[currentPair.prev]}
        probs={flash ? prevProbs : currentProbs}
        target={flash ? flash.target : currentPair.target}
        started={step !== 0}
      />
      <FlashMatrixTable heading="W (raw logits)" matrix={W} prevTransform={(row) => row} vocab={vocab} trainedRows={trainedRows} flash={flash} live />

      <RowConvergenceTable errorMatrix={errorMatrix} empirical={empirical} rowErrors={rowErrors} totalError={totalError} vocab={vocab} trainedRows={trainedRows} flash={flash} />
      <GapToFinal W={W} wFinal={wFinal} vocab={vocab} />
      <DistanceToFinal W={W} wFinal={wFinal} wInitial={wInitial} />
      <PredictionsVsFinal softmaxW={softmaxW} softmaxWFinal={softmaxWFinal} vocab={vocab} />
      <Heatmap heading="W (trained — used at inference)" matrix={W} vocab={vocab} cellBackground={stepColormap(step, W)} trainedRows={trainedRows} live />
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
