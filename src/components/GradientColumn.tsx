import { color, font, space } from '../theme';
import { PRECISION } from '../consts';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';
import { Term } from './Term';
import { TokenChip } from './TokenChip';

interface GradientColumnProps {
  vocab: readonly string[];
  target: number;
  prevProbs: number[];
  prevGrad: number[];
  dimmed: boolean;
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

export function GradientColumn({ vocab, target, prevProbs, prevGrad, dimmed }: GradientColumnProps) {
  return (
    <Panel title="Gradient (p − y)" dimmed={dimmed} live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`\\frac{\\partial \\mathcal{L}}{\\partial x_i} = p_i - y_i`} />
        <div style={{ color: color.text.secondary, fontSize: font.size.md }}>
          <Term
            label="y"
            explain="The target as a one-hot vector: 1 at the actual next token, 0 everywhere else."
            formula={`y_i = \\begin{cases} 1 & i = \\text{target} \\\\ 0 & \\text{otherwise} \\end{cases}`}
          />{' '}
          is one-hot at <TokenChip text={vocab[target]} />.
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
              <th style={{ ...cell, textAlign: 'left' }}>token</th>
              <th style={cell}>p</th>
              <th style={cell}>y</th>
              <th style={cell}>p − y</th>
            </tr>
          </thead>
          <tbody>
            {prevProbs.map((p, i) => {
              const isTarget = i === target;
              const g = prevGrad[i];
              return (
                <tr key={vocab[i]} style={{ fontWeight: isTarget ? 'bold' : 'normal' }}>
                  <th style={{ ...cell, textAlign: 'left' }}>
                    {isTarget ? <TokenChip text={vocab[i]} /> : vocab[i]}
                  </th>
                  <td style={{ ...cell, color: color.text.secondary }}>
                    {p.toFixed(PRECISION)}
                  </td>
                  <td style={{ ...cell, color: isTarget ? color.text.emphasis : color.text.muted }}>
                    {isTarget ? '1.00' : '0.00'}
                  </td>
                  <td style={{ ...cell, color: g < 0 ? color.error.fg : color.text.secondary }}>
                    {g >= 0 ? '+' : ''}
                    {g.toFixed(PRECISION)}
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
