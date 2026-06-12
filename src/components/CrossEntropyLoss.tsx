import { color, font, space } from '../theme';
import { PRECISION } from '../consts';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';
import { TokenChip } from './TokenChip';

interface CrossEntropyLossProps {
  vocab: readonly string[];
  probs: number[];
  target: number;
  dimmed?: boolean;
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

export function CrossEntropyLoss({ vocab, probs, target, dimmed }: CrossEntropyLossProps) {
  return (
    <Panel title="Probabilities p" live dimmed={dimmed}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`} />
        <table
          style={{
            borderCollapse: 'collapse',
            fontFamily: 'monospace',
            fontSize: font.size.md,
          }}
        >
          <thead>
            <tr style={{ color: color.text.secondary }}>
              <th style={{ ...cell, textAlign: 'left' }}>token</th>
              <th style={cell}>p = softmax(x)</th>
            </tr>
          </thead>
          <tbody>
            {probs.map((p, i) => {
              const isTarget = i === target;
              return (
                <tr key={vocab[i]} style={{ fontWeight: isTarget ? 'bold' : 'normal' }}>
                  <th style={{ ...cell, textAlign: 'left' }}>
                    {isTarget ? <TokenChip text={vocab[i]} /> : vocab[i]}
                  </th>
                  <td style={{ ...cell, color: isTarget ? color.text.emphasis : color.text.secondary }}>
                    {p.toFixed(PRECISION)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
