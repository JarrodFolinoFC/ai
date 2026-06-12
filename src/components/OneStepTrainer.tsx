import { Flex } from 'antd';
import { useEffect, useRef } from 'react';

import { color, space, font } from '../theme';
import { Panel } from './Panel';
import { Term } from './Term';

interface OneStepTrainerProps {
  seed: number;
  onSeedChange: (seed: number) => void;
  lr: number;
  onLrChange: (lr: number) => void;
  pairsLength: number;
  step: number;
  currentLoss: number;
  onStep: (n: number) => void;
  onReset: () => void;
}

export function OneStepTrainer({
  seed,
  onSeedChange,
  lr,
  onLrChange,
  pairsLength,
  step,
  onStep,
  onReset,
}: OneStepTrainerProps) {
  const onStepRef = useRef(onStep);
  onStepRef.current = onStep;
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== 'n' && e.key !== 'N') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
      e.preventDefault();
      onStepRef.current(1);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

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
          <button type="button" onClick={() => onStep(1)} title="press n">
            step +1 (n)
          </button>
          <button type="button" onClick={() => onStep(10)}>
            step +10
          </button>
          <button type="button" onClick={() => onStep(pairsLength)}>
            1 epoch ({pairsLength})
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
          <span>{step > 0 ? `epoch ${Math.floor(step / pairsLength)} progress` : "(training not started)"}</span>
          <span>
            {step % pairsLength}/{pairsLength} pair
          </span>
        </div>
        </div>
      </Flex>
    </Panel>
  );
}
