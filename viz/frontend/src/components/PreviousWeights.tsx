import { divergingColormap } from '../colormaps';
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
}

// W as it stood at the previous step — the first link in this step's chain
// (prev W → forward → loss → gradient → update → new W). Reconstructed by
// reverting the just-trained row to its pre-step values.
export function PreviousWeights({ vocab, W, flash, step }: PreviousWeightsProps) {
  const prevW = flash
    ? W.map((rowVals, i) => (i === flash.row ? flash.prevRow : rowVals))
    : W;
  const maxAbs = Math.max(1e-6, ...prevW.flat().map((v) => Math.abs(v)));

  return (
    <Heatmap
      heading="W (previous step — chain start)"
      matrix={prevW}
      vocab={vocab}
      cellBackground={divergingColormap(maxAbs)}
      live
      badgeStep={Math.max(0, step - 1)}
    />
  );
}
