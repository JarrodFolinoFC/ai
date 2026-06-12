import { color, font, space } from '../theme';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';

interface DistanceToFinalProps {
  // Current live weights.
  W: number[][];
  // Fully-trained reference weights.
  wFinal: number[][];
  // The seeded initial weights, used as the 0%-converged baseline.
  wInitial: number[][];
}

// Frobenius norm ‖A‖ = sqrt(Σ aᵢⱼ²).
function frobenius(a: number[][], b: number[][]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      const d = a[i][j] - b[i][j];
      sum += d * d;
    }
  }
  return Math.sqrt(sum);
}

// A single scalar per step: how far the live weights are from the converged
// reference, plus a progress bar (0% at the initial weights, 100% at W_final).
export function DistanceToFinal({ W, wFinal, wInitial }: DistanceToFinalProps) {
  const distance = frobenius(wFinal, W);
  const initialDistance = frobenius(wFinal, wInitial);
  const pct =
    initialDistance > 0
      ? Math.min(100, Math.max(0, (1 - distance / initialDistance) * 100))
      : 100;

  return (
    <Panel title="Distance to final" live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`\\lVert W_{\\text{final}} - W \\rVert = \\sqrt{\\textstyle\\sum_{i,j} (W^{\\text{final}}_{ij} - W_{ij})^2}`} />
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          ‖W_final − W‖ = {distance.toFixed(3)} &nbsp; ({pct.toFixed(0)}% converged)
        </span>
        <div
          style={{
            height: '0.6rem',
            background: color.bg.disabled,
            borderRadius: '999px',
            overflow: 'hidden',
            maxWidth: font.prose,
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: '100%',
              background: color.positive,
              transition: 'width 0.15s ease',
            }}
          />
        </div>
      </div>
    </Panel>
  );
}
