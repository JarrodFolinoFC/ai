import { stepColormap } from '../colormaps';
import type { Flash } from '../bigramTrainer';
import { Heatmap } from './Heatmap';

interface PreviousWeightsProps {
  vocab: readonly string[];
  // Current live weights.
  W: number[][];
  // The most recently trained row; its prevRow is W before this step's update.
  flash: Flash | null;
  // Current training step (the badge shows step − 1).
  step: number;
  // Row index feeding the Softmax card — drawn with a bottom border.
  usedRow: number;
}

// W as it stood at the previous step — the first link in this step's chain
// (prev W → forward → loss → gradient → update → new W). Reconstructed by
// reverting the just-trained row to its pre-step values.
export function PreviousWeights({ vocab, W, flash, step, usedRow }: PreviousWeightsProps) {
  const prevW = flash
    ? W.map((rowVals, i) => (i === flash.row ? flash.prevRow : rowVals))
    : W;

  return (
    <Heatmap
      heading={step <= 1 ? "Untrained weights" : "W (previous step — chain start)"}
      subHeading={"Each row is a logit"}
      matrix={prevW}
      vocab={vocab}
      cellBackground={stepColormap(step, prevW)}
      live
      badgeStep={Math.max(0, step - 1)}
      borderRow={usedRow !== 0 ? usedRow : null}
    />
  );
}
