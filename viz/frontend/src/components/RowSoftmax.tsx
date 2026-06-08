import { color, font, space } from '../theme';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';

interface RowSoftmaxProps {
  vocab: readonly string[];
  // The prev token whose row of W is being turned into probabilities.
  prevToken: string;
  // Vocab index of the target (next) token, highlighted in the table.
  target: number;
  // The row's logits x = Wₐ, their exponentials, the sum, and the resulting p.
  logits: number[];
  exps: number[];
  expSum: number;
  probs: number[];
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

// Worked example of the per-row softmax pᵢ = eˣⁱ / Σⱼ eˣʲ applied to the
// current pair's row of W: logit, its exponential, and the normalized prob.
export function RowSoftmax({ vocab, prevToken, target, logits, exps, expSum, probs }: RowSoftmaxProps) {
  return (
    <Panel title="Softmax (per row)" live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`} />
        <div style={{ color: color.text.secondary, fontSize: font.size.sm }}>
          row a = <strong>{prevToken}</strong>, target = <strong>{vocab[target]}</strong>
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
            </tr>
          </thead>
          <tbody>
            {logits.map((x, i) => {
              const isTarget = i === target;
              return (
                <tr key={vocab[i]} style={{ fontWeight: isTarget ? 'bold' : 'normal' }}>
                  <th style={{ ...cell, textAlign: 'left', color: color.highlight }}>
                    {vocab[i]}
                  </th>
                  <td style={{ ...cell, color: color.text.secondary }}>
                    {x >= 0 ? '+' : ''}
                    {x.toFixed(2)}
                  </td>
                  <td style={{ ...cell, color: color.text.secondary }}>
                    {exps[i].toFixed(2)}
                  </td>
                  <td style={{ ...cell, color: color.text.emphasis }}>
                    {probs[i].toFixed(2)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ fontWeight: 'bold', borderTop: `1px solid ${color.border.strong}` }}>
              <th style={{ ...cell, textAlign: 'left' }}>Σ</th>
              <td style={cell} />
              <td style={{ ...cell, color: color.text.secondary }}>{expSum.toFixed(2)}</td>
              <td style={{ ...cell, color: color.text.emphasis }}>1.00</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
