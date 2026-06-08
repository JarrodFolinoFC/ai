import { Flex } from 'antd';

import { color, space, font } from '../theme';
import { Panel } from './Panel';
import { Term } from './Term';

interface OneStepTrainerProps {
  seed: number;
  onSeedChange: (seed: number) => void;
  lr: number;
  onLrChange: (lr: number) => void;
  // Number of training pairs in one epoch.
  pairsLength: number;
  // Total pairs processed so far.
  step: number;
  // Cross-entropy loss of the current worked example.
  currentLoss: number;
  // Advance training by n pairs.
  onStep: (n: number) => void;
  onReset: () => void;
}

// Controls for the single-step bigram trainer: seed/lr inputs, step buttons,
// a status readout, and an epoch progress bar.
export function OneStepTrainer({
  seed,
  onSeedChange,
  lr,
  onLrChange,
  pairsLength,
  step,
  currentLoss,
  onStep,
  onReset,
}: OneStepTrainerProps) {
  return (
    <Panel title="One-step trainer">
      <Flex vertical gap={space.sm}>
        <Flex gap={space.lg} wrap align="center">
          <label>
            <Term
              label="seed"
              explain="Random seed: fixes the random number generator so weight initialization and any shuffling are reproducible. Same seed → same run; change it to try a different starting point."
            />
            :&nbsp;
            <input
              type="number"
              value={seed}
              onChange={(e) => onSeedChange(Number(e.target.value))}
              style={{ width: '5rem' }}
            />
          </label>
          <label>
            <Term
              label="lr"
              explain="Learning rate (η): how big a step each update takes down the loss gradient. Higher learns faster but can overshoot and diverge; lower is stabler but slower."
              formula="\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}"
            />
            :&nbsp;
            <input
              type="number"
              step={0.05}
              min={0}
              value={lr}
              onChange={(e) => onLrChange(Number(e.target.value))}
              style={{ width: '5rem' }}
            />
          </label>
        </Flex>
        <Flex gap={space.sm} wrap align="center">
          <button type="button" onClick={() => onStep(1)}>
            step ×1
          </button>
          <button type="button" onClick={() => onStep(10)}>
            step ×10
          </button>
          <button type="button" onClick={() => onStep(pairsLength)}>
            1 epoch ({pairsLength})
          </button>
          <button type="button" onClick={() => onStep(pairsLength * 50)}>
            50 epochs
          </button>
          <button type="button" onClick={onReset}>
            reset
          </button>
        </Flex>
        <span style={{ fontFamily: 'monospace', color: color.text.secondary }}>
          <Term
            label="step"
            explain="Total number of training pairs processed so far. Each step does one forward pass + weight update on a single (context → next-token) pair."
          />
          ={step} &nbsp;
          <Term
            label="epoch"
            explain="One full pass over all training pairs. Computed as step ÷ pairs-per-epoch, so it increments once every full sweep through the corpus."
            formula="\text{epoch} = \left\lfloor \frac{\text{step}}{N_{\text{pairs}}} \right\rfloor"
          />
          ={Math.floor(step / pairsLength)} &nbsp;
          <Term
            label="loss"
            explain="Cross-entropy loss of the current worked example: how surprised the model is by the correct next token. Lower is better; it falls as training improves the predictions."
            formula="\mathcal{L} = -\log p_\theta(\text{target} \mid \text{context})"
          />
          ={currentLoss.toFixed(3)}
        </span>
        <div style={{ maxWidth: font.prose }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontFamily: 'monospace',
            fontSize: '0.8rem',
            color: color.text.secondary,
            marginBottom: '0.2rem',
          }}
        >
          <span>epoch {Math.floor(step / pairsLength)} progress</span>
          <span>
            {step % pairsLength}/{pairsLength} pairs
          </span>
        </div>
        <div
          style={{
            height: '0.6rem',
            background: color.bg.disabled,
            borderRadius: '999px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${((step % pairsLength) / pairsLength) * 100}%`,
              height: '100%',
              background: color.positive,
              transition: 'width 0.15s ease',
            }}
          />
        </div>
        </div>
      </Flex>
    </Panel>
  );
}
