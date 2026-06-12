import { softmaxRow } from '../funcs';
import { PRECISION } from '../consts';
import { color, font } from '../theme';
import { flashStyles, type Flash } from '../flashStyles';
import { ChangeBadge } from './ChangeBadge';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';
import { Term } from './Term';

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
    <Panel
      live
      title={
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4ch' }}>
          Per-row convergence:
          <FormulaDisplay
            latex={`\\lvert \\text{softmax}(W) - \\text{empirical} \\rvert`}
            inline
          />
        </span>
      }
    >
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
              <Term
                label="row error"
                explain="Total absolute mismatch between the row's predicted distribution and the empirical target, halved so a perfect match is 0 and total disagreement is 1 (½ · L1 distance)."
                formula={`\\tfrac{1}{2}\\sum_j \\lvert \\text{softmax}(W)_{ij} - \\text{empirical}_{ij} \\rvert`}
              />
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
                  const delta = flashing ? Number((d - oldD).toFixed(PRECISION)) : 0;
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
                      {d.toFixed(PRECISION)}
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
              }}
            >
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4ch' }}>
                <span style={{ fontStyle: 'italic' }}>average row error =</span>
                <FormulaDisplay
                  latex={`\\frac{1}{R}\\sum_i \\tfrac{1}{2}\\sum_j \\lvert \\text{softmax}(W)_{ij} - \\text{empirical}_{ij} \\rvert`}
                  inline
                />
              </span>
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
