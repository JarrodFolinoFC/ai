import { color, space } from '../theme';
import { PRECISION } from '../consts';
import { FormulaDisplay } from './FormulaDisplay';
import { Panel } from './Panel';
import { Term } from './Term';
import { TokenChip } from './TokenChip';

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
    <Panel title="Gradient (p − y)" dimmed={dimmed} live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.xs }}>
      <FormulaDisplay latex={`\\frac{\\partial \\mathcal{L}}{\\partial x_i} = p_i - y_i`} />
      <div style={{ color: color.text.secondary }}>
        <Term
          label="p"
          explain="The model's predicted next-token distribution: the softmax of the prev token's row of W."
          formula={`p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}`}
        />{' '}
        is the prediction;{' '}
        <Term
          label="y"
          explain="The target as a one-hot vector: 1 at the actual next token, 0 everywhere else."
          formula={`y_i = \\begin{cases} 1 & i = \\text{target} \\\\ 0 & \\text{otherwise} \\end{cases}`}
        />{' '}
        is the truth, one-hot at the target (<TokenChip text={vocab[target]} />).
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
      <div>
        <Term
          label="p − y"
          explain="The gradient of the cross-entropy loss with respect to the logits. Each weight is nudged in the opposite direction; it is positive where the model over-predicts and negative at the target."
          formula={`\\frac{\\partial \\mathcal{L}}{\\partial x_i} = p_i - y_i`}
        />{' '}
        = [
        {prevGrad.map((g, i) => (
          <span
            key={i}
            style={{
              color: g < 0 ? color.error.fg : color.text.secondary,
              fontWeight: i === target ? 'bold' : 'normal',
            }}
          >
            {g >= 0 ? '+' : ''}
            {g.toFixed(PRECISION)}
            {i < prevGrad.length - 1 ? ', ' : ''}
          </span>
        ))}
        ]
      </div>
      </div>
    </Panel>
  );
}
