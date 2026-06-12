import { stepColormap } from '../colormaps';
import { softmaxRow } from '../funcs';
import type { Flash } from '../bigramTrainer';
import { Heatmap } from './Heatmap';

interface PreviousSoftmaxProps {
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

// softmax(W) per row as it stood at the previous step — the predictions the
// model made before this step's update. Sits next to the current softmax(W)
// per row so the before/after of the whole prediction matrix is visible.
export function PreviousSoftmax({ vocab, W, flash, step, usedRow }: PreviousSoftmaxProps) {
  const prevW = flash
    ? W.map((rowVals, i) => (i === flash.row ? flash.prevRow : rowVals))
    : W;
  const softmaxPrev = prevW.map(softmaxRow);

  return (
    <Heatmap
      heading={step > 1 ? "softmax(W) per row (before update)" : "softmax (untrained weights)"}
      matrix={softmaxPrev}
      vocab={vocab}
      cellBackground={stepColormap(step)}
      live
      badgeStep={Math.max(0, step - 1)}
      borderRow={usedRow}
    />
  );
}
