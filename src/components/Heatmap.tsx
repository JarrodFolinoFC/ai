import type { ReactNode } from 'react';
import { PRECISION } from '../consts';

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
}: HeatmapProps) {
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
            <th style={{ padding: '0.25rem 0.5rem' }}></th>
            {vocab.map((w) => (
              <th
                key={w}
                style={{
                  padding: '0.25rem 0.5rem',
                  borderBottom: `1px solid ${color.border.strong}`,
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
