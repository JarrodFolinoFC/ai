import { color, radius, font } from '../theme';

interface ForwardPassTableProps {
  vocab: readonly string[];
  logits: number[];
  exps: number[];
  expSum: number;
  probs: number[];
  targetIdx: number;
  prevToken: string;
}

// One softmax forward pass over a single W row: logit x, e^x, and the
// normalised probability p, with the target row highlighted.
export function ForwardPassTable({
  vocab,
  logits,
  exps,
  expSum,
  probs,
  targetIdx,
  prevToken,
}: ForwardPassTableProps) {
  const cell = { padding: '0.15rem 0.5rem' } as const;
  const numCell = { ...cell, textAlign: 'right' as const };
  return (
    <table
      style={{
        borderCollapse: 'collapse',
        marginTop: '0.2rem',
        fontFamily: 'monospace',
        fontSize: font.size.sm,
      }}
    >
      <thead>
        <tr style={{ color: color.text.secondary }}>
          <th style={{ ...cell, textAlign: 'left' }}>next</th>
          <th style={numCell}>
            x = W[
            <span
              style={{
                background: color.highlightBg,
                color: color.text.emphasis,
                fontWeight: 'bold',
                padding: '0 0.2rem',
                borderRadius: radius.sm,
              }}
            >
              {prevToken}
            </span>
            ]
          </th>
          <th style={numCell}>e^x</th>
          <th style={numCell}>p = e^x / Σ</th>
        </tr>
      </thead>
      <tbody>
        {logits.map((x, i) => {
          const isTarget = i === targetIdx;
          return (
            <tr
              key={i}
              style={{
                background: isTarget ? color.highlightBg : 'transparent',
                fontWeight: isTarget ? 'bold' : 'normal',
              }}
            >
              <td style={cell}>{vocab[i]}</td>
              <td style={numCell}>
                {x >= 0 ? '+' : ''}
                {x.toFixed(2)}
              </td>
              <td style={{ ...numCell, color: color.text.muted }}>
                {exps[i].toFixed(2)}
              </td>
              <td
                style={{
                  ...numCell,
                  color: isTarget ? color.text.emphasis : color.text.secondary,
                }}
              >
                {probs[i].toFixed(2)}
              </td>
            </tr>
          );
        })}
        <tr style={{ color: color.text.muted, borderTop: `1px solid ${color.border.default}` }}>
          <td style={cell}>Σ</td>
          <td style={cell} />
          <td style={numCell}>{expSum.toFixed(2)}</td>
          <td style={numCell}>1.00</td>
        </tr>
      </tbody>
    </table>
  );
}
