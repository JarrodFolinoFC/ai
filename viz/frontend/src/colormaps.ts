// Reusable cell-background colormaps for matrix heatmaps. Each factory returns a
// `(value, i, j) => string` compatible with Heatmap's `cellBackground` prop.

import { color } from './theme';
import { VOCAB } from './consts';

export type Colormap = (value: number, i: number, j: number) => string;

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

// Two-hue ramp diverging around zero; magnitude is normalized by `maxAbs`.
// Positive values fade toward `pos`, negatives toward `neg`.
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
