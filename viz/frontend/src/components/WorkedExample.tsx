import { Flex } from 'antd';

import { color, space, radius, font } from '../theme';
import { softmaxRow } from '../funcs';
import { CurrentPairSummary } from './CurrentPairSummary';
import { CorpusPositionTracker } from './CorpusPositionTracker';
import { ForwardPassTable } from './ForwardPassTable';
import { GradientColumn } from './GradientColumn';
import { FlashMatrixTable } from './FlashMatrixTable';
import { RowConvergenceTable } from './RowConvergenceTable';
import { Heatmap, probCellBg } from './Heatmap';

interface Flash {
  row: number;
  target: number;
  pairIdx: number;
  prevRow: number[];
}

interface WorkedExampleProps {
  vocab: readonly string[];
  // Total training steps taken (0 = not started; greys the derived columns).
  step: number;
  // The step currently being annotated, or null when nothing is flashed.
  flash: Flash | null;
  pairsLength: number;
  corpusTokens: string[];
  // The training pair in play.
  currentPair: { prev: number; target: number };
  // Forward pass before the step (drives the gradient).
  prevLogits: number[];
  prevExps: number[];
  prevExpSum: number;
  prevProbs: number[];
  prevGrad: number[];
  // Forward pass after the step (result of the update).
  currentLogits: number[];
  currentExps: number[];
  currentExpSum: number;
  currentProbs: number[];
  // Weight / probability matrices shown alongside the worked example.
  W: number[][];
  softmaxW: number[][];
  empirical: number[][];
  errorMatrix: number[][];
  rowErrors: number[];
  totalError: number;
  trainedRows: Set<number>;
}

// One annotated training step (worked example) beside the live W / softmax(W) /
// empirical / error matrices it updates.
export function WorkedExample({
  vocab,
  step,
  flash,
  pairsLength,
  corpusTokens,
  currentPair,
  prevLogits,
  prevExps,
  prevExpSum,
  prevProbs,
  prevGrad,
  currentLogits,
  currentExps,
  currentExpSum,
  currentProbs,
  W,
  softmaxW,
  empirical,
  errorMatrix,
  rowErrors,
  totalError,
  trainedRows,
}: WorkedExampleProps) {
  return (
    <Flex gap={space.xl} wrap align="flex-start">
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
        <Flex gap="1.25rem" wrap align="flex-start">
          <CurrentPairSummary
            vocab={vocab}
            prev={currentPair.prev}
            target={currentPair.target}
            started={step !== 0}
          />
          <CorpusPositionTracker
            pairIdx={flash ? flash.pairIdx : null}
            pairsLength={pairsLength}
            corpusTokens={corpusTokens}
          />
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
                {vocab[currentPair.prev]}
              </span>{' '}
              of W, exponentiate each logit, then divide by the total:
            </div>
            {flash ? (
              <>
                <div
                  style={{
                    marginTop: '0.4rem',
                    fontSize: '0.8rem',
                    fontWeight: 'bold',
                    color: color.text.secondary,
                  }}
                >
                  before this step — drives the gradient
                </div>
                <ForwardPassTable
                  vocab={vocab}
                  logits={prevLogits}
                  exps={prevExps}
                  expSum={prevExpSum}
                  probs={prevProbs}
                  targetIdx={currentPair.target}
                  prevToken={vocab[currentPair.prev]}
                />
                <div
                  style={{
                    marginTop: '0.6rem',
                    fontSize: '0.8rem',
                    fontWeight: 'bold',
                    color: color.text.secondary,
                  }}
                >
                  after this step — result of the update
                </div>
                <ForwardPassTable
                  vocab={vocab}
                  logits={currentLogits}
                  exps={currentExps}
                  expSum={currentExpSum}
                  probs={currentProbs}
                  targetIdx={currentPair.target}
                  prevToken={vocab[currentPair.prev]}
                />
              </>
            ) : (
              <ForwardPassTable
                vocab={vocab}
                logits={currentLogits}
                exps={currentExps}
                expSum={currentExpSum}
                probs={currentProbs}
                targetIdx={currentPair.target}
                prevToken={vocab[currentPair.prev]}
              />
            )}
          </div>
          <GradientColumn
            vocab={vocab}
            target={currentPair.target}
            prevProbs={prevProbs}
            prevGrad={prevGrad}
            dimmed={step === 0}
          />
        </Flex>
      </div>

      <div style={{ flex: '1 1 auto' }}>
        <Flex gap="2rem" wrap>
          <FlashMatrixTable heading="W (raw logits)" matrix={W} prevTransform={(row) => row} vocab={vocab} trainedRows={trainedRows} flash={flash} />
          <FlashMatrixTable heading="softmax(W) per row" matrix={softmaxW} prevTransform={softmaxRow} vocab={vocab} trainedRows={trainedRows} flash={flash} />
          <Heatmap heading="Empirical (target)" subHeading="pᵢ = countᵢ / Σ count" matrix={empirical} vocab={vocab} cellBackground={probCellBg} />
          <RowConvergenceTable errorMatrix={errorMatrix} empirical={empirical} rowErrors={rowErrors} totalError={totalError} vocab={vocab} trainedRows={trainedRows} flash={flash} />
        </Flex>
      </div>
    </Flex>
  );
}
