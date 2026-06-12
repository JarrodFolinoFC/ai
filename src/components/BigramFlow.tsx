import { Flex } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { BigramTrainer, type Flash } from '../bigramTrainer';
import { randomMatrix } from '../funcs';
import { StepContext } from '../stepContext';
import { color, space } from '../theme';
import { CrossEntropyLoss } from './CrossEntropyLoss';
import { FlashMatrixTable } from './FlashMatrixTable';
import { GradientColumn } from './GradientColumn';
import { OneStepTrainer } from './OneStepTrainer';
import { PreviousSoftmax } from './PreviousSoftmax';
import { WeightMatrix } from './WeightMatrix';
import { RowSoftmax } from './RowSoftmax';
import { TrainingCorpus } from './TrainingCorpus';
import { WeightUpdate } from './WeightUpdate';
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

  const { pairs } = useMemo(
    () => BigramTrainer.corpusStats(corpus, vocab),
    [corpus, vocab]
  );
  const trainer = useMemo(() => new BigramTrainer(W, pairs, step), [W, pairs, step]);
  const {
    currentPair, trainedRows, currentLoss,
    prevLogits, prevExps, prevExpSum, prevProbs, prevGrad,
  } = useMemo(
    () => trainer.derive(flash),
    [trainer, flash]
  );

  const displayLoss =
    lossHistory.length > 0 ? lossHistory[lossHistory.length - 1] : currentLoss;

  useEffect(() => {
    setW(randomMatrix(seed, vocab.length, vocab.length, 1));
    setStep(0);
    setLossHistory([]);
  }, [seed, corpus, vocab]);

  function doStep(n = 1) {
    const { W: nextW, step: nextStep, flash: nextFlash } =
      trainer.doStep(n, lr);
    setW(nextW);
    setStep(nextStep);
    setFlash(nextFlash);
  }

  function reset() {
    setW(randomMatrix(seed, vocab.length, vocab.length, 1));
    setStep(0);
    setLossHistory([]);
    setFlash(null);
  }

  const prevW = flash
    ? W.map((row, i) => (i === flash.row ? flash.prevRow : row))
    : W;

  const prevPair = flash && step >= 2 ? pairs[(step - 2) % pairs.length] : null;
  const prevStepHighlight = prevPair ? { row: prevPair.prev, target: prevPair.target } : null;

  return (
    <StepContext.Provider value={step}>
    <Flex wrap align="stretch" gap={space.lg}>
      <Flex gap={space.lg} align="stretch">
        <TokenIdMap vocab={vocab} />
        <WeightMatrix
          heading={step <= 1 ? 'Untrained weights' : 'W (previous step — chain start)'}
          vocab={vocab} matrix={prevW} step={step}
          flash={prevStepHighlight} badgeStep={Math.max(0, step - 1)}
          borderRow={currentPair.prev !== 0 ? currentPair.prev : undefined}
        />
        <PreviousSoftmax vocab={vocab} W={W} flash={flash} step={step} usedRow={currentPair.prev} />
        <TrainingCorpus corpus={corpus}
                          pairIdx={flash ? flash.pairIdx : null}
                          pairsLength={pairs.length} currentPair={currentPair} vocab={vocab} started={step !== 0}
        />
      </Flex>
      <Flex gap={space.lg} align="stretch">
        <RowSoftmax vocab={vocab} currentPair={currentPair} logits={prevLogits} exps={prevExps} expSum={prevExpSum} probs={prevProbs} />
        <CrossEntropyLoss vocab={vocab} probs={prevProbs} target={currentPair.target} dimmed={step === 0} />
        <GradientColumn vocab={vocab} target={currentPair.target} prevProbs={prevProbs} prevGrad={prevGrad} dimmed={step === 0} />
        <WeightUpdate vocab={vocab} lr={lr} flash={flash} W={W} prevGrad={prevGrad} dimmed={step === 0} />
      </Flex>
      <Flex  gap={space.lg} align="stretch">
        <FlashMatrixTable heading="W (raw logits)" matrix={W} prevTransform={(row) => row} vocab={vocab} trainedRows={trainedRows} flash={flash} live />
        <WeightMatrix
          heading="W (trained — used at inference)"
          vocab={vocab} matrix={W} step={step}
          flash={flash} trainedRows={trainedRows}
        />
      </Flex>
    </Flex>
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
