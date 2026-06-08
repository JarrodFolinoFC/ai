import { color } from '../theme';
import { Panel } from './Panel';

interface GradientColumnProps {
  vocab: readonly string[];
  // Vocab index of the target (next) token.
  target: number;
  // Pre-step probabilities (drives the gradient); only its length and the
  // target index matter for the one-hot row.
  prevProbs: number[];
  // Pre-step gradient p − y.
  prevGrad: number[];
  // Greyed out before training has started.
  dimmed: boolean;
}

// The loss/gradient column: the one-hot target vector y and the gradient
// p − y that each weight is nudged against.
export function GradientColumn({ vocab, target, prevProbs, prevGrad, dimmed }: GradientColumnProps) {
  return (
    <Panel title="Gradient (p − y)" dimmed={dimmed}>
      <div style={{ marginTop: 0, color: color.text.secondary }}>
        <code>y</code> = one-hot at the target (<strong>
          {vocab[target]}
        </strong>
        ): 1 there, 0 elsewhere:
      </div>
      <div style={{ marginTop: '0.15rem' }}>
        y = [
        {prevProbs.map((_, i) => {
          const isTarget = i === target;
          return (
            <span
              key={i}
              style={{
                fontWeight: isTarget ? 'bold' : 'normal',
                color: isTarget ? color.text.emphasis : color.text.muted,
              }}
            >
              {isTarget ? '1.00' : '0.00'}
              {i < prevProbs.length - 1 ? ', ' : ''}
            </span>
          );
        })}
        ]
      </div>
      <div style={{ marginTop: '0.15rem' }}>
        p − y = [
        {prevGrad.map((g, i) => (
          <span
            key={i}
            style={{
              color: g < 0 ? color.error.fg : color.text.secondary,
              fontWeight: i === target ? 'bold' : 'normal',
            }}
          >
            {g >= 0 ? '+' : ''}
            {g.toFixed(2)}
            {i < prevGrad.length - 1 ? ', ' : ''}
          </span>
        ))}
        ]
      </div>
    </Panel>
  );
}
