import { softmaxRow } from '../funcs';
import { color, font } from '../theme';
import { flashStyles, type Flash } from '../flashStyles';
import { ChangeBadge } from './ChangeBadge';
import { Panel } from './Panel';

interface RowConvergenceTableProps {
  // Signed per-cell error: softmax(W) − empirical.
  errorMatrix: number[][];
  // Empirical target probabilities, used to recompute the pre-step error when flashed.
  empirical: number[][];
  // Per-row total error (½·L1) and its mean across rows.
  rowErrors: number[];
  totalError: number;
  vocab: readonly string[];
  // Rows that have received at least one gradient update; untrained rows are greyed.
  trainedRows: Set<number>;
  // The training step currently being annotated, or null when nothing is flashed.
  flash: Flash | null;
}

export function RowConvergenceTable({
  errorMatrix,
  empirical,
  rowErrors,
  totalError,
  vocab,
  trainedRows,
  flash,
}: RowConvergenceTableProps) {
  const { aHead, bHead, cellBox } = flashStyles(flash);

  return (
    <Panel title="Per-row convergence: |softmax(W) − empirical|">
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
            <th
              style={{
                padding: '0.25rem 0.5rem',
                borderBottom: `1px solid ${color.border.strong}`,
                borderLeft: `2px solid ${color.border.strong}`,
              }}
            >
              row error
            </th>
          </tr>
        </thead>
        <tbody>
          {errorMatrix.map((row, i) => {
            const flashing = flash?.row === i;
            const oldProbs = flashing ? softmaxRow(flash.prevRow) : null;
            const oldRowError = oldProbs
              ? oldProbs.reduce((acc, op, j) => acc + Math.abs(op - empirical[i][j]), 0) / 2
              : rowErrors[i];
            const rowErrorDelta = flashing
              ? Number((rowErrors[i] - oldRowError).toFixed(3))
              : 0;
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
                {row.map((d, j) => {
                  const oldD = oldProbs ? oldProbs[j] - empirical[i][j] : d;
                  const delta = flashing ? Number((d - oldD).toFixed(2)) : 0;
                  return (
                    <td
                      key={j}
                      style={{
                        position: 'relative',
                        padding: '0.25rem 0.5rem',
                        textAlign: 'right',
                        background: trainedRows.has(i) ? undefined : color.bg.disabled,
                        color: color.text.secondary,
                        ...cellBox(i, j),
                      }}
                    >
                      {flashing && <ChangeBadge prev={oldD} delta={delta} signedPrev />}
                      {d >= 0 ? '+' : ''}
                      {d.toFixed(2)}
                    </td>
                  );
                })}
                <td
                  style={{
                    position: 'relative',
                    padding: '0.25rem 0.5rem',
                    textAlign: 'right',
                    borderLeft: `2px solid ${color.border.strong}`,
                    fontWeight: 'bold',
                    background: trainedRows.has(i) ? undefined : color.bg.disabled,
                    color: color.text.secondary,
                  }}
                >
                  {flashing && (
                    <ChangeBadge prev={oldRowError} delta={rowErrorDelta} decimals={3} />
                  )}
                  {rowErrors[i].toFixed(3)}
                </td>
              </tr>
            );
          })}
          <tr>
            <th
              style={{
                textAlign: 'right',
                padding: '0.25rem 0.5rem',
                borderTop: `2px solid ${color.border.strong}`,
                borderRight: `1px solid ${color.border.strong}`,
              }}
            >
              mean
            </th>
            <td
              colSpan={vocab.length}
              style={{
                padding: '0.25rem 0.5rem',
                borderTop: `2px solid ${color.border.strong}`,
                color: color.text.secondary,
                fontStyle: 'italic',
              }}
            >
              average row error
            </td>
            <td
              style={{
                padding: '0.25rem 0.5rem',
                textAlign: 'right',
                borderTop: `2px solid ${color.border.strong}`,
                borderLeft: `2px solid ${color.border.strong}`,
                fontWeight: 'bold',
              }}
            >
              {totalError.toFixed(3)}
            </td>
          </tr>
        </tbody>
      </table>
    </Panel>
  );
}
