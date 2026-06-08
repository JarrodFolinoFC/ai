import { Flex } from 'antd';
import { useEffect, useMemo, useState } from 'react';
import { randomMatrix, softmaxRow } from '../funcs';
import { TrainingCorpus } from './TrainingCorpus';
import { Heatmap } from './Heatmap';
import { OneStepTrainer } from './OneStepTrainer';
import { ForwardPassSection } from './ForwardPassSection';
import { GradientColumn } from './GradientColumn';
import { FlashMatrixTable } from './FlashMatrixTable';
import { RowConvergenceTable } from './RowConvergenceTable';
import { RecentLoss } from './RecentLoss';
import { BigramTrainer, type Flash } from '../bigramTrainer';
import { probCellBg, divergingColormap, sequentialColormap } from '../colormaps';
import { color, space } from '../theme';

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

  const { counts, pairs, empirical } = useMemo(
    () => BigramTrainer.corpusStats(corpus, vocab),
    [corpus, vocab]
  );
  const trainer = useMemo(() => new BigramTrainer(W, pairs, step), [W, pairs, step]);
  const { wFinal, softmaxWFinal, wFinalMaxAbs } = useMemo(
    () => BigramTrainer.trainReference(seed, vocab, pairs),
    [seed, vocab, pairs]
  );
  const { softmaxW, errorMatrix, rowErrors, totalError } = useMemo(
    () => trainer.analysis(empirical),
    [trainer, empirical]
  );

  const {
    currentPair, trainedRows,
    currentLogits, currentExps, currentExpSum, currentProbs, currentLoss,
    prevLogits, prevExps, prevExpSum, prevProbs, prevGrad,
  } = useMemo(
    () => trainer.derive(flash),
    [trainer, flash]
  );

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
    <Flex wrap align="stretch" gap={space.lg}>
      <TrainingCorpus corpus={corpus}
                      pairIdx={flash ? flash.pairIdx : null}
                      pairsLength={pairs.length} currentPair={currentPair} vocab={vocab} started={step !== 0}
      />
      <OneStepTrainer
        seed={seed} onSeedChange={setSeed} lr={lr}
        onLrChange={setLr} pairsLength={pairs.length} step={step}
        currentLoss={currentLoss} onStep={doStep} onReset={reset}
      />
      <Heatmap heading="Bigram counts (rows = prev, cols = next)" matrix={counts} vocab={vocab} formatValue={(c) => c}
          cellBackground={(c, i, j) =>
            vocab[i] === 'cat' && vocab[j] === 'on'
              ? color.highlightBg
              : c === 0
                ? color.bg.surface
                : 'transparent'}
        />
      <ForwardPassSection
        vocab={vocab} targetIdx={currentPair.target} prevToken={vocab[currentPair.prev]}
        before={{ logits: prevLogits, exps: prevExps, expSum: prevExpSum, probs: prevProbs }}
        current={{ logits: currentLogits, exps: currentExps, expSum: currentExpSum, probs: currentProbs }}
      />
      <GradientColumn vocab={vocab} target={currentPair.target} prevProbs={prevProbs} prevGrad={prevGrad} dimmed={step === 0} />
      <FlashMatrixTable heading="W (raw logits)" matrix={W} prevTransform={(row) => row} vocab={vocab} trainedRows={trainedRows} flash={flash} />
      <FlashMatrixTable heading="softmax(W) per row" matrix={softmaxW} prevTransform={softmaxRow} vocab={vocab} trainedRows={trainedRows} flash={flash} />
      <Heatmap heading="Empirical (target)" subHeading="pᵢ = countᵢ / Σ count" matrix={empirical} vocab={vocab} cellBackground={probCellBg} />
      <Heatmap heading="Softmax" subHeading="pᵢ = e^xᵢ / Σ e^x" matrix={counts.map(softmaxRow)} vocab={vocab} cellBackground={probCellBg} />
      <RecentLoss lossHistory={lossHistory} />
      <Heatmap heading="W_final (raw logits)" matrix={wFinal} vocab={vocab} cellBackground={divergingColormap(wFinalMaxAbs)} />
      <Heatmap
          heading="softmax(W_final) per row"
          matrix={softmaxWFinal}
          vocab={vocab}
          cellBackground={sequentialColormap([34, 197, 94])}
      />
      <RowConvergenceTable errorMatrix={errorMatrix} empirical={empirical} rowErrors={rowErrors} totalError={totalError} vocab={vocab} trainedRows={trainedRows} flash={flash} />
    </Flex>
  );
}
