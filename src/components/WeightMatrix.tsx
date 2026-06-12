import { stepColormap } from '../colormaps';
import type { Flash } from '../bigramTrainer';
import type { HeaderHighlight } from '../flashStyles';
import { Heatmap } from './Heatmap';

interface WeightMatrixProps {
  vocab: readonly string[];
  // Current live weights W.
  W: number[][];
  // The most recently trained row; its prevRow is W before this step's update.
  flash: Flash | null;
  // Current training step (drives the colormap and the header highlight).
  step: number;
  // Which point in the step's chain to render:
  //  'previous' — W as it stood at the chain start, before this step's update
  //               (the trained row reverted), badged one step behind.
  //  'current'  — the live trained W used at inference, badged at the current step.
  variant: 'previous' | 'current';
  // Which row/column the headers highlight. Defaults to the current step's
  // trained pair (`flash`); the chain-start table passes the previous step's
  // pair so each table highlights the step it represents.
  highlightFlash?: HeaderHighlight | null;
  // 'previous' only: the row feeding the Softmax card, drawn with a bottom border.
  usedRow?: number;
  // 'current' only: rows that have been trained; untrained rows are greyed.
  trainedRows?: Set<number>;
}

// The weight matrix W shown at two adjacent points in a training step: the
// chain start ('previous', one step behind) and the post-update inference
// weights ('current'). Both render the same vocab × vocab logit grid.
export function WeightMatrix({ vocab, W, flash, step, variant, highlightFlash, usedRow, trainedRows }: WeightMatrixProps) {
  // Default the header highlight to the current step's trained pair.
  const highlight = highlightFlash === undefined ? flash : highlightFlash;
  if (variant === 'previous') {
    // Reconstruct the pre-step matrix by reverting the just-trained row.
    const prevW = flash
      ? W.map((rowVals, i) => (i === flash.row ? flash.prevRow : rowVals))
      : W;
    return (
      <Heatmap
        heading={step <= 1 ? 'Untrained weights' : 'W (previous step — chain start)'}
        subHeading="Each row is a logit"
        matrix={prevW}
        vocab={vocab}
        cellBackground={stepColormap(step, prevW)}
        flash={highlight}
        live
        badgeStep={Math.max(0, step - 1)}
        borderRow={usedRow !== 0 ? usedRow : undefined}
      />
    );
  }

  return (
    <Heatmap
      heading="W (trained — used at inference)"
      matrix={W}
      vocab={vocab}
      cellBackground={stepColormap(step, W)}
      flash={highlight}
      trainedRows={trainedRows}
      live
    />
  );
}
