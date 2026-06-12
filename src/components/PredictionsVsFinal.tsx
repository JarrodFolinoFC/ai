import { stepColormap, type Colormap } from '../colormaps';
import { PRECISION } from '../consts';
import { flashStyles, type Flash } from '../flashStyles';
import { useStep } from '../stepContext';
import { color, font, space } from '../theme';
import { Panel } from './Panel';

interface PredictionsVsFinalProps {
  // Current per-row predicted distribution, softmax(W).
  softmaxW: number[][];
  // Fully-trained per-row predicted distribution, softmax(W_final).
  softmaxWFinal: number[][];
  vocab: readonly string[];
  // The step currently being annotated, used to highlight the prev/target headers.
  flash: Flash | null;
}

// One labeled probability table (rows = prev, cols = next).
function ProbTable({
  label,
  matrix,
  vocab,
  cellBackground,
  flash,
}: {
  label: string;
  matrix: number[][];
  vocab: readonly string[];
  cellBackground: Colormap;
  flash: Flash | null;
}) {
  const { aHead, bHead } = flashStyles(flash);
  return (
    <div>
      <div style={{ fontSize: font.size.sm, color: color.text.secondary, marginBottom: space.xs }}>
        {label}
      </div>
      <table style={{ borderCollapse: 'collapse', fontFamily: 'monospace', fontSize: font.size.sm }}>
        <thead>
          <tr>
            <th style={{ padding: '0.2rem 0.4rem', fontSize: '0.75rem', fontStyle: 'italic', color: color.highlight, textAlign: 'right' }}>
              prev=a ↓
            </th>
            {vocab.map((w, j) => (
              <th key={w} style={{ padding: '0.2rem 0.4rem', borderBottom: `1px solid ${color.border.strong}`, ...bHead(j) }}>
                {w}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={vocab[i]}>
              <th style={{ padding: '0.2rem 0.4rem', textAlign: 'right', borderRight: `1px solid ${color.border.strong}`, ...aHead(i) }}>
                {vocab[i]}
              </th>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '0.2rem 0.4rem',
                    textAlign: 'right',
                    color: color.text.secondary,
                    background: cellBackground(v, i, j),
                  }}
                >
                  {v.toFixed(PRECISION)}
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
export function PredictionsVsFinal({ softmaxW, softmaxWFinal, vocab, flash }: PredictionsVsFinalProps) {
  const cellBackground = stepColormap(useStep());
  return (
    <Panel title="Predictions: now vs final" live>
      <div style={{ display: 'flex', gap: space.lg, flexWrap: 'wrap' }}>
        <ProbTable label="softmax(W) — now" matrix={softmaxW} vocab={vocab} cellBackground={cellBackground} flash={flash} />
        <ProbTable label="softmax(W_final) — final" matrix={softmaxWFinal} vocab={vocab} cellBackground={cellBackground} flash={flash} />
      </div>
    </Panel>
  );
}
