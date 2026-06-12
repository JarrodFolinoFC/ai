import { color, font, space } from '../theme';
import { PRECISION } from '../consts';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';

import { useStep } from '../stepContext';


interface RowSoftmaxProps {
  vocab: readonly string[];
  prevToken: string;
  logits: number[];
  exps: number[];
  expSum: number;
  probs: number[];
  currentPair: { prev: number; target: number };
}

const cell = {
  padding: '0.25rem 0.5rem',
  textAlign: 'right' as const,
};

// Token chip styled like the highlighted tokens in TrainingCorpus: the prev
// ("a"/input) token gets the highlight background, the target ("b"/next) token
// gets the info background, both with bold emphasis text.
function TokenChip({ text, isPrev }: { text: string; isPrev: boolean }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.05rem 0.25rem',
        borderRadius: '3px',
        background: isPrev ? color.highlightBg : color.info.border,
        color: color.text.emphasis,
        fontWeight: 'bold',
      }}
    >
      {text}
    </span>
  );
}

export function RowSoftmax({ vocab, logits, exps, expSum, probs, currentPair  }: RowSoftmaxProps) {
  let prevToken = vocab[currentPair.prev]
  let targetToken = vocab[currentPair.target]
  const step = useStep();

  let target = currentPair.target
  return (
    <Panel
      title={
        step === 0 ? (
          'No Softmax to calculate'
        ) : (
          <span>
            Softmax <TokenChip text={prevToken} isPrev />
          </span>
        )
      }
      live
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <FormulaDisplay latex={`p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`} />
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
              <th style={cell}>x = Wₐ,ᵢ</th>
              <th style={cell}>eˣ</th>
              <th style={cell}>p = eˣ/Σ</th>
            </tr>
          </thead>
          <tbody>
            {logits.map((x, i) => {
              const isTarget = i === target;
              // After W ← W − η(p − y): the target logit rises, the rest fall.
              const dirColor = isTarget ? color.positive : color.error.fg;
              return (
                <tr key={vocab[i]} style={{ fontWeight: isTarget ? 'bold' : 'normal' }}>
                  <th style={{ ...cell, textAlign: 'left' }}>
                    {vocab[i] === targetToken ? <TokenChip text={vocab[i]} /> : vocab[i] }
                  </th>
                  <td style={{ ...cell, color: dirColor }}>
                    {isTarget ? '↑' : '↓'} {x >= 0 ? '+' : ''}
                    {x.toFixed(PRECISION)}
                  </td>
                  <td style={{ ...cell, color: color.text.secondary }}>
                    {exps[i].toFixed(PRECISION)}
                  </td>
                  <td style={{ ...cell, color: color.text.emphasis }}>
                    {probs[i].toFixed(PRECISION)}
                  </td>
                </tr>
              );
            })}
            <tr style={{ fontWeight: 'bold', borderTop: `1px solid ${color.border.strong}` }}>
              <th style={{ ...cell, textAlign: 'left' }}>Σ</th>
              <td style={cell} />
              <td style={{ ...cell, color: color.text.secondary }}>{expSum.toFixed(PRECISION)}</td>
              <td style={{ ...cell, color: color.text.emphasis }}>1.00</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
