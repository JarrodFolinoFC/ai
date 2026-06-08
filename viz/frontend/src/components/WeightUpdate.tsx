import { color, font, space } from '../theme';
import type { Flash } from '../bigramTrainer';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';

interface WeightUpdateProps {
  vocab: readonly string[];
  // Learning rate η.
  lr: number;
  // The most recently trained row (prev row + which step), or null when a
  // single step isn't being animated.
  flash: Flash | null;
  // Live weights, so the "after" column matches the W (raw logits) card exactly.
  W: number[][];
  // Gradient p − y for the flashed row.
  prevGrad: number[];
  // Greyed out before training has started.
  dimmed: boolean;
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

// Worked example of the SGD weight update Wₐ ← Wₐ − η·(p − y) applied to the
// row of W for the current pair's prev token: old weight, gradient, step, new.
export function WeightUpdate({ vocab, lr, flash, W, prevGrad, dimmed }: WeightUpdateProps) {
  return (
    <Panel title="Weight update" dimmed={dimmed} live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`W_{a,i} \\leftarrow W_{a,i} - \\eta \\, (p_i - y_i)`} />
        {flash ? (
          <>
            <div style={{ color: color.text.secondary, fontSize: font.size.sm }}>
              row a ={' '}
              <strong style={{ color: color.highlight }}>{vocab[flash.row]}</strong>, η ={' '}
              {lr}, target ={' '}
              <strong style={{ color: color.text.emphasis }}>{vocab[flash.target]}</strong>
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
                  <th style={{ ...cell, textAlign: 'left' }}>next</th>
                  <th style={cell}>Wₐ,ᵢ</th>
                  <th style={cell}>p − y</th>
                  <th style={cell}>−η(p − y)</th>
                  <th style={cell}>Wₐ,ᵢ (new)</th>
                </tr>
              </thead>
              <tbody>
                {flash.prevRow.map((old, i) => {
                  const grad = prevGrad[i];
                  const delta = -lr * grad;
                  const next = W[flash.row][i];
                  const isTarget = i === flash.target;
                  return (
                    <tr
                      key={vocab[i]}
                      style={{ fontWeight: isTarget ? 'bold' : 'normal' }}
                    >
                      <th style={{ ...cell, textAlign: 'left', color: color.highlight }}>
                        {vocab[i]}
                      </th>
                      <td style={{ ...cell, color: color.text.secondary }}>
                        {old.toFixed(2)}
                      </td>
                      <td style={{ ...cell, color: grad < 0 ? color.error.fg : color.text.secondary }}>
                        {grad >= 0 ? '+' : ''}
                        {grad.toFixed(2)}
                      </td>
                      <td style={{ ...cell, color: delta >= 0 ? color.positive : color.error.fg }}>
                        {delta >= 0 ? '+' : ''}
                        {delta.toFixed(2)}
                      </td>
                      <td style={{ ...cell, color: color.text.emphasis }}>
                        {next.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        ) : (
          <div style={{ color: color.text.secondary, fontSize: font.size.sm }}>
            Step ×1 to see the update worked out for a single row.
          </div>
        )}
      </div>
    </Panel>
  );
}
