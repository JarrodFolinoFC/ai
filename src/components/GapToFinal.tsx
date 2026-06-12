import { stepColormap } from '../colormaps';
import { PRECISION } from '../consts';
import { useStep } from '../stepContext';
import { Heatmap } from './Heatmap';

interface GapToFinalProps {
  // Current live weights.
  W: number[][];
  // Fully-trained reference weights.
  wFinal: number[][];
  vocab: readonly string[];
}

// Per-step view of how far each weight still has to move: W_final − W. Cells
// shrink toward 0 (faint) as the live model converges on the reference.
export function GapToFinal({ W, wFinal, vocab }: GapToFinalProps) {
  const gap = wFinal.map((row, i) => row.map((v, j) => v - W[i][j]));

  return (
    <Heatmap
      heading="Gap to final (W_final − W)"
      matrix={gap}
      vocab={vocab}
      cellBackground={stepColormap(useStep(), gap)}
      formatValue={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(PRECISION)}`}
      live
    />
  );
}
