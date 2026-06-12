import { color, font } from '../theme';
import { PRECISION } from '../consts';
import { flashStyles, type Flash } from '../flashStyles';
import { ChangeBadge } from './ChangeBadge';
import { Panel } from './Panel';

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
  // Mark the card as live-updating (adds a step badge to the title).
  live?: boolean;
}

export function FlashMatrixTable({
  heading,
  matrix,
  prevTransform,
  vocab,
  trainedRows,
  flash,
  live = false,
}: FlashMatrixTableProps) {
  const { aHead, bHead, cellBox } = flashStyles(flash);

  return (
    <Panel title={heading} live={live}>
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
                  const delta = flashing ? Number((v - prev).toFixed(PRECISION)) : 0;
                  return (
                    <td
                      key={j}
                      style={{
                        position: 'relative',
                        // Wide enough for the two-number ChangeBadge so the
                        // column doesn't widen when a row flashes.
                        minWidth: '6rem',
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: trainedRows.has(i) ? undefined : color.bg.disabled,
                        ...cellBox(i, j),
                      }}
                    >
                      {flashing ? (
                        <ChangeBadge prev={prev} delta={delta} />
                      ) : (
                        // Reserve the badge's line height so the table doesn't
                        // grow when a row flashes (matches ChangeBadge metrics).
                        <div style={{ fontSize: '0.6rem', lineHeight: 1.1, marginBottom: '0.1rem' }}>
                          &nbsp;
                        </div>
                      )}
                      {v.toFixed(PRECISION)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}
