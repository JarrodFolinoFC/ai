import { color } from './theme';

export interface Flash {
  row: number;
  target: number;
  prevRow: number[];
}

// Style helpers for highlighting the cell, row, and column touched by the
// training step currently being flashed. Each returns null when nothing is
// flashed at the given index, so the result can be spread into a style object.
export function flashStyles(flash: Flash | null) {
  return {
    // Black box around the single cell being updated (trained row × target column).
    cellBox: (i: number, j: number) =>
      flash?.row === i && flash?.target === j
        ? { border: `2px solid ${color.text.primary}` }
        : null,
    // Trained row's label ("a" = prev): warning highlight while flashed.
    aHead: (i: number) =>
      flash?.row === i
        ? {
            background: color.highlightBg,
            color: color.text.emphasis,
            fontWeight: 'bold' as const,
          }
        : null,
    // Target column's label ("b" = next): info highlight while flashed.
    bHead: (j: number) =>
      flash?.target === j
        ? {
            background: color.info.border,
            color: color.text.emphasis,
            fontWeight: 'bold' as const,
          }
        : null,
  };
}
