// Reusable cell-background colormaps for matrix heatmaps. Each factory returns a
// `(value, i, j) => string` compatible with Heatmap's `cellBackground` prop.

import { color } from './theme';
import { VOCAB } from './consts';

export type Colormap = (value: number, i: number, j: number) => string;

// Flat colormap: every cell is white regardless of value.
const whiteColormap: Colormap = () => '#ffffff';

// Value-ramped greyscale: cells darken with |value| (normalized by maxAbs, so a
// value at maxAbs is darkest and 0 is near-white). Used to render the untrained
// step-0 state in neutral grey. Probability tables can omit maxAbs (values lie
// in [0, 1]).
function greyscaleColormap(
  maxAbs = 1,
  { base = 0.05, range = 0.55 }: { base?: number; range?: number } = {}
): Colormap {
  const grey: RGB = [80, 80, 80];
  return (value) => rgba(grey, base + range * Math.min(1, Math.abs(value) / maxAbs));
}

// Background colormap for a live table: a value-ramped greyscale before training
// (step 0), flat white once weights start moving. Pass `matrix` to scale the
// grey ramp to its magnitude; omit it for probability tables (values are already
// in [0, 1]).
export function stepColormap(step: number | null, matrix?: number[][]): Colormap {
  if (step !== 0) return whiteColormap;
  const maxAbs = matrix
    ? Math.max(1e-6, ...matrix.flat().map((v) => Math.abs(v)))
    : 1;
  return greyscaleColormap(maxAbs);
}

type RGB = readonly [number, number, number];

const rgba = ([r, g, b]: RGB, alpha: number) => `rgba(${r}, ${g}, ${b}, ${alpha})`;

// Single-hue ramp: alpha grows with value (expected in [0, 1]).
// alpha = base + range * value.
export function sequentialColormap(
  rgb: RGB,
  { base = 0.1, range = 0.6 }: { base?: number; range?: number } = {}
): Colormap {
  return (value) => rgba(rgb, base + range * value);
}

export function divergingColormap(
  maxAbs: number,
  {
    pos = [59, 130, 246] as RGB,
    neg = [239, 68, 68] as RGB,
    base = 0.1,
    range = 0.5,
  }: { pos?: RGB; neg?: RGB; base?: number; range?: number } = {}
): Colormap {
  return (value) => {
    const t = value / maxAbs;
    return t >= 0 ? rgba(pos, base + range * t) : rgba(neg, base + range * -t);
  };
}

// Probability ramp (green) with two demo-specific special cases: the cat→on
// cell is highlighted, and zero-probability cells use the surface background.
const probRamp = sequentialColormap([34, 197, 94]);
export const probCellBg: Colormap = (p, i, j) =>
  VOCAB[i] === 'cat' && VOCAB[j] === 'on'
    ? color.highlightBg
    : p === 0
      ? color.bg.surface
      : probRamp(p, i, j);
