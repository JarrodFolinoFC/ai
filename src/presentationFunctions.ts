// Presentational helpers that map numeric values to cell colours for the
// matrix/vector tables. These return CSS colour strings (not pure math), so they
// live apart from funcs.ts and may depend on the design tokens.
import { color } from './theme';

// Diverging blue↔red scale around zero. Non-finite values (e.g. masked -Infinity
// cells) render as primary text colour rather than a colour artifact.
export function shadeDiverging(v: number, maxAbs: number) {
  if (!Number.isFinite(v)) return color.text.primary;
  const t = v / Math.max(maxAbs, 1e-9);
  if (t >= 0) return `rgba(59, 130, 246, ${0.1 + 0.5 * t})`;
  return `rgba(239, 68, 68, ${0.1 + 0.5 * -t})`;
}

// Sequential green scale for probabilities in [0, 1].
export function shadeProb(p: number) {
  return `rgba(34, 197, 94, ${0.1 + 0.6 * p})`;
}
