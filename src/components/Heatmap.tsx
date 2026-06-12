import type { ReactNode } from 'react';
import { PRECISION } from '../consts';
import { flashStyles, type HeaderHighlight } from '../flashStyles';

import { color, font } from '../theme';
import { Panel } from './Panel';

interface HeatmapProps {
  heading: string;
  subHeading?: ReactNode;
  matrix: number[][];
  vocab: readonly string[];
  cellBackground: (value: number, i: number, j: number) => string;
  // How to render a cell's value; defaults to 2-decimal fixed (counts pass an integer formatter).
  formatValue?: (value: number) => ReactNode;
  // Greyed out (e.g. before training has started).
  dimmed?: boolean;
  // When provided, rows whose index is NOT in this set are greyed individually
  // (e.g. rows that haven't received a gradient update yet).
  trainedRows?: Set<number>;
  // Mark the card as live-updating (adds a step badge to the title).
  live?: boolean;
  // Override the step shown in the live badge (e.g. a previous-step card).
  badgeStep?: number;
  // Draw a bottom border under this row (e.g. the row feeding another card).
  borderRow?: number;
  // Which row (prev=a) and column (target=b) to highlight in the headers. When
  // set, the trained row header and target column header are highlighted,
  // matching the flash matrix tables.
  flash?: HeaderHighlight | null;
}

const dimRow = { opacity: 0.4, filter: 'grayscale(1)' };

export function Heatmap({
  heading,
  subHeading,
  matrix,
  vocab,
  cellBackground,
  formatValue = (v) => v.toFixed(PRECISION),
  dimmed = false,
  trainedRows,
  live = false,
  badgeStep,
  borderRow,
  flash,
}: HeatmapProps) {
  const { aHead, bHead } = flashStyles(flash ?? null);
  return (
    <Panel title={heading} dimmed={dimmed} live={live} badgeStep={badgeStep}>
      {subHeading !== undefined && (
        <div style={{ fontSize: font.size.sm, color: color.text.secondary }}>
          {subHeading}
        </div>
      )}
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
            const underline =
              i === borderRow ? `2px solid ${color.emphasis}` : undefined;
            return (
            <tr key={vocab[i]} style={trainedRows && !trainedRows.has(i) ? dimRow : undefined}>
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
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    padding: '0.25rem 0.5rem',
                    textAlign: 'right',
                    background: cellBackground(v, i, j),
                    borderBottom: underline,
                  }}
                >
                  {formatValue(v)}
                </td>
              ))}
            </tr>
            );
          })}
        </tbody>
      </table>
    </Panel>
  );
}
