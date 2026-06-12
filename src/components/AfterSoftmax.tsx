import { color, font, space } from '../theme';
import { PRECISION } from '../consts';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';

interface AfterSoftmaxProps {
  vocab: readonly string[];
  // The prev token whose (now updated) row of W is turned into probabilities.
  prevToken: string;
  // Vocab index of the target (next) token.
  target: number;
  // The row of W after this step's update (new logits).
  row: number[];
  // The pre-update probabilities, to show Δp = new − old.
  oldProbs: number[];
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

// The post-update forward pass: softmax of the updated row, with Δp = p − p_old
// so you can see exactly how each next-token probability moved this step.
export function AfterSoftmax({ vocab, prevToken, target, row, oldProbs }: AfterSoftmaxProps) {
  const exps = row.map((x) => Math.exp(x));
  const expSum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / expSum);

  return (
    <Panel title="Softmax (per row) — after update" live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`} />
        <div style={{ color: color.text.secondary, fontSize: font.size.sm }}>
          row a = <strong>{prevToken}</strong>, target = <strong>{vocab[target]}</strong> —
          recomputed on the updated row
        </div>
        <table
          style={{
            borderCollapse: 'collapse',
            fontFamily: 'monospace',
            fontSize: font.size.md,
          }}
        >
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ ...cell, textAlign: 'left' }}>next</th>
              <th style={cell}>x = Wₐ,ᵢ</th>
              <th style={cell}>eˣ</th>
              <th style={cell}>p = eˣ/Σ</th>
              <th style={cell}>Δp = p − p_old</th>
            </tr>
          </thead>
          <tbody>
            {row.map((x, i) => {
              const isTarget = i === target;
              const delta = probs[i] - oldProbs[i];
              return (
                <tr key={vocab[i]} style={{ fontWeight: isTarget ? 'bold' : 'normal' }}>
                  <th style={{ ...cell, textAlign: 'left', color: color.highlight }}>
                    {vocab[i]}
                  </th>
                  <td style={{ ...cell, color: color.text.secondary }}>
                    {x >= 0 ? '+' : ''}
                    {x.toFixed(PRECISION)}
                  </td>
                  <td style={{ ...cell, color: color.text.secondary }}>{exps[i].toFixed(PRECISION)}</td>
                  <td style={{ ...cell, color: color.text.emphasis }}>{probs[i].toFixed(PRECISION)}</td>
                  <td style={{ ...cell, color: delta >= 0 ? color.positive : color.error.fg }}>
                    {delta >= 0 ? '+' : ''}
                    {delta.toFixed(PRECISION)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ fontWeight: 'bold', borderTop: `1px solid ${color.border.strong}` }}>
              <th style={{ ...cell, textAlign: 'left' }}>Σ</th>
              <td style={cell} />
              <td style={{ ...cell, color: color.text.secondary }}>{expSum.toFixed(PRECISION)}</td>
              <td style={{ ...cell, color: color.text.emphasis }}>1.00</td>
              <td style={cell} />
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
