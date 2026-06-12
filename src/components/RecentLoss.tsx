import { color, space } from '../theme';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';
import { Term } from './Term';

interface RecentLossProps {
  lossHistory: number[];
}

// Mean of the most recent training losses (last 50 steps). Always visible;
// before any step has run it shows the formula with empty/zero placeholders.
// Shows the full calculation: the averaging formula followed by each
// intermediate step — the values, their sum, and the final division by N.
export function RecentLoss({ lossHistory }: RecentLossProps) {
  const recent = lossHistory.slice(-50);
  const n = recent.length;
  const sum = recent.reduce((a, b) => a + b, 0);
  const mean = n > 0 ? sum / n : 0;

  const monospace = {
    fontFamily: 'monospace',
    fontSize: '0.85rem',
    color: color.text.secondary,
    wordBreak: 'break-word' as const,
  };

  return (
    <Panel title="Recent mean loss" live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay
          latex={`\\bar{\\mathcal{L}} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathcal{L}_i`}
        />
        {n === 0 ? (
          <span style={monospace}>no steps yet — train to populate</span>
        ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: space.xs }}>
          <span style={monospace}>
            <strong>
              1. values (Lᵢ,{' '}
              <Term
                label="i"
                explain="Summation index: counts from 1 to N, selecting each recent loss in turn. L₁ is the oldest loss in the window, L_N the most recent."
                formula="i = 1, 2, \dots, N"
              />{' '}
              = 1…
              <Term
                label="N"
                explain="Number of recent per-step losses being averaged — the loss history capped at the last 50 steps. It equals the step count only until step 50, then stays at 50."
                formula="N = \min(\text{step}, 50)"
              />
              ={n}):
            </strong>{' '}
            [{recent.map((l, idx) => `L${idx + 1}=${l.toFixed(3)}`).join(', ')}]
          </span>
          <span style={monospace}>
            <strong>2. sum:</strong>{' '}
            {recent.map((l) => l.toFixed(3)).join(' + ')} = {sum.toFixed(3)}
          </span>
          <span style={monospace}>
            <strong>3. divide by N:</strong>{' '}
            {sum.toFixed(3)} / {n} = {mean.toFixed(3)}
          </span>
        </div>
        )}
      </div>
    </Panel>
  );
}
