import { probCellBg } from '../colormaps';
import { color, font, space } from '../theme';
import { Panel } from './Panel';

interface PredictionsVsFinalProps {
  // Current per-row predicted distribution, softmax(W).
  softmaxW: number[][];
  // Fully-trained per-row predicted distribution, softmax(W_final).
  softmaxWFinal: number[][];
  vocab: readonly string[];
}

// One labeled probability table (rows = prev, cols = next).
function ProbTable({
  label,
  matrix,
  vocab,
}: {
  label: string;
  matrix: number[][];
  vocab: readonly string[];
}) {
  return (
    <div>
      <div style={{ fontSize: font.size.sm, color: color.text.secondary, marginBottom: space.xs }}>
        {label}
      </div>
      <table style={{ borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: font.size.sm }}>
        <thead>
          <tr>
            <th />
            {vocab.map((w) => (
              <th key={w} style={{ padding: '0.2rem 0.4rem', borderBottom: `1px solid ${color.border.strong}` }}>
                {w}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={vocab[i]}>
              <th style={{ padding: '0.2rem 0.4rem', textAlign: 'right', borderRight: `1px solid ${color.border.strong}` }}>
                {vocab[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '0.2rem 0.4rem',
                    textAlign: 'right',
                    color: color.text.secondary,
                    background: probCellBg(v, i, j),
                  }}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Side-by-side comparison: the current model's predictions next to the
// converged model's, so you watch softmax(W) morph into softmax(W_final).
export function PredictionsVsFinal({ softmaxW, softmaxWFinal, vocab }: PredictionsVsFinalProps) {
  return (
    <Panel title="Predictions: now vs final" live>
      <div style={{ display: 'flex', gap: space.lg, flexWrap: 'wrap' }}>
        <ProbTable label="softmax(W) — now" matrix={softmaxW} vocab={vocab} />
        <ProbTable label="softmax(W_final) — final" matrix={softmaxWFinal} vocab={vocab} />
      </div>
    </Panel>
  );
}
