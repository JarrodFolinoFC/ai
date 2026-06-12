import { stepColormap } from '../colormaps';
import type { HeaderHighlight } from '../flashStyles';
import { Heatmap } from './Heatmap';

interface WeightMatrixProps {
  vocab: readonly string[];
  matrix: number[][];
  heading: string;
  step: number;
  flash?: HeaderHighlight | null;
  badgeStep?: number;
  borderRow?: number;
  trainedRows?: Set<number>;
}

export function WeightMatrix({ vocab, matrix, heading, step, flash, badgeStep, borderRow, trainedRows }: WeightMatrixProps) {
  return (
    <Heatmap
      heading={heading}
      matrix={matrix}
      vocab={vocab}
      cellBackground={stepColormap(step, matrix)}
      flash={flash}
      badgeStep={badgeStep}
      borderRow={borderRow}
      trainedRows={trainedRows}
      live
    />
  );
}
