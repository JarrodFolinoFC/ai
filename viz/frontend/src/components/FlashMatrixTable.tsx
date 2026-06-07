import { color, font } from '../theme';
import { flashStyles, type Flash } from '../flashStyles';
import { ChangeBadge } from './ChangeBadge';

interface FlashMatrixTableProps {
  heading: string;
  // Displayed values, vocab × vocab (raw logits W, or softmax(W) per row).
  matrix: number[][];
  // Maps a raw W row to its displayed values, used to recover the pre-step
  // values for the flashed row (identity for raw logits, softmaxRow for probs).
  prevTransform: (row: number[]) => number[];
  vocab: readonly string[];
  // Rows that have received at least one gradient update; untrained rows are greyed.
  trainedRows: Set<number>;
  // The training step currently being annotated, or null when nothing is flashed.
  flash: Flash | null;
}

export function FlashMatrixTable({
  heading,
  matrix,
  prevTransform,
  vocab,
  trainedRows,
  flash,
}: FlashMatrixTableProps) {
  const { aHead, bHead, cellBox } = flashStyles(flash);

  return (
    <div>
      <h4>{heading}</h4>
      <table
        style={{
          borderCollapse: 'collapse',
          fontFamily: 'monospace',
          fontSize: font.size.md,
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                padding: '0.25rem 0.5rem',
                fontSize: '0.75rem',
                fontStyle: 'italic',
                color: color.highlight,
                textAlign: 'right',
              }}
            >
              prev=a ↓
            </th>
            {vocab.map((w, j) => (
              <th
                key={w}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderBottom: `1px solid ${color.border.strong}`,
                  ...bHead(j),
                }}
              >
                {w}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const flashing = flash?.row === i;
            const prevRow = flashing ? prevTransform(flash.prevRow) : null;
            return (
              <tr key={vocab[i]}>
                <th
                  style={{
                    textAlign: 'right',
                    padding: '0.25rem 0.5rem',
                    borderRight: `1px solid ${color.border.strong}`,
                    ...aHead(i),
                  }}
                >
                  {vocab[i]}
                </th>
                {row.map((v, j) => {
                  const prev = prevRow ? prevRow[j] : v;
                  const delta = flashing ? Number((v - prev).toFixed(2)) : 0;
                  return (
                    <td
                      key={j}
                      style={{
                        position: 'relative',
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: trainedRows.has(i) ? undefined : color.bg.disabled,
                        ...cellBox(i, j),
                      }}
                    >
                      {flashing && <ChangeBadge prev={prev} delta={delta} />}
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
