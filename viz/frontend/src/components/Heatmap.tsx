import type { CSSProperties, ReactNode } from 'react';

import { color, font } from '../theme';
import { VOCAB } from '../consts';

export const probCellBg = (p: number, i: number, j: number) =>
  VOCAB[i] === 'cat' && VOCAB[j] === 'on'
    ? color.highlightBg
    : p === 0
      ? color.bg.surface
      : `rgba(34, 197, 94, ${0.1 + 0.6 * p})`;

interface HeatmapProps {
  heading: string;
  subHeading?: string;
  matrix: number[][];
  vocab: readonly string[];
  cellBackground: (value: number, i: number, j: number) => string;
  // How to render a cell's value; defaults to 2-decimal fixed (counts pass an integer formatter).
  formatValue?: (value: number) => ReactNode;
  headingStyle?: CSSProperties;
}

export function Heatmap({
  heading,
  subHeading,
  matrix,
  vocab,
  cellBackground,
  formatValue = (v) => v.toFixed(2),
  headingStyle,
}: HeatmapProps) {
  return (
    <div>
      <h4 style={headingStyle}>{heading}</h4>
      {subHeading !== undefined && (
        <code style={{ fontSize: font.size.sm, color: color.text.secondary }}>
          {subHeading}
        </code>
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
          {matrix.map((row, i) => (
            <tr key={vocab[i]}>
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
                  }}
                >
                  {formatValue(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
