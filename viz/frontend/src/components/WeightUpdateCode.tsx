import { useState } from 'react';
import { Button, Modal } from 'antd';

import { color, font, space } from '../theme';
import { JsFormula } from './JsFormula';
import { JsRunner } from './JsRunner';
import { Panel } from './Panel';

interface WeightUpdateCodeProps {
  vocab: readonly string[];
  // Learning rate η.
  lr: number;
  // Prev token whose row is updated.
  rowLabel: string;
  // The row of W being updated (pre-step values).
  row: number[];
  // Predicted distribution p = softmax(row).
  probs: number[];
  // Vocab index of the target (next) token.
  target: number;
  // False before the first step — no update has been applied yet.
  started: boolean;
}

const arr = (xs: number[]) => `[${xs.map((v) => v.toFixed(3)).join(', ')}]`;

// Build a runnable snippet seeded with the real values for the current step.
function buildCode(
  vocab: readonly string[],
  lr: number,
  rowLabel: string,
  row: number[],
  probs: number[],
  target: number
): string {
  const y = vocab.map((_, i) => (i === target ? 1 : 0));
  return `// Real values from the current step (row a = "${rowLabel}")
const lr = ${lr};
const Wa = ${arr(row)};  // W["${rowLabel}"]
const p  = ${arr(probs)};  // softmax(Wa)
const y  = [${y.join(', ')}];  // one-hot at "${vocab[target]}"

// Pure SGD update — returns a new row, no mutation:
const updateRow = (row, p, y, lr) =>
  row.map((w, i) => w - lr * (p[i] - y[i]));

const newRow = updateRow(Wa, p, y, lr);

console.log('Wa     =', Wa.map((v) => v.toFixed(2)).join('  '));
console.log('p - y  =', p.map((pi, i) => (pi - y[i]).toFixed(2)).join('  '));
console.log('newRow =', newRow.map((v) => v.toFixed(2)).join('  '));`;
}

// Companion card to WeightUpdate: the same update as runnable JavaScript seeded
// with the step's real values, expandable to a full-screen runner with console.
export function WeightUpdateCode({ vocab, lr, rowLabel, row, probs, target, started }: WeightUpdateCodeProps) {
  const [open, setOpen] = useState(false);

  if (!started) {
    return (
      <Panel title="Weight update (JavaScript)" live>
        <div style={{ color: color.text.secondary, fontSize: font.size.sm }}>
          Step ×1 to populate the code with this step's real values and run it.
        </div>
      </Panel>
    );
  }

  const code = buildCode(vocab, lr, rowLabel, row, probs, target);

  return (
    <Panel title="Weight update (JavaScript)" live>
      <div style={{ display: 'flex', flexDirection: 'column', gap: space.sm }}>
        <JsFormula code={code} />
        <div>
          <Button size="small" onClick={() => setOpen(true)}>
            Expand &amp; run ⤢
          </Button>
        </div>
      </div>
      <Modal
        title="Weight update — run JavaScript"
        open={open}
        onCancel={() => setOpen(false)}
        footer={null}
        width="90vw"
        style={{ top: 20 }}
        styles={{ body: { maxHeight: '82vh', overflow: 'auto' } }}
      >
        <JsRunner initialCode={code} />
      </Modal>
    </Panel>
  );
}
