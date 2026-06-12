import { useState } from 'react';

import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { buildLegend } from '../legendEntries';
import { VectorBars } from '../viz/VectorBars';

import { color, font } from '../theme';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('oneHotTruthY', 'oneHotYi', 'vocabSizeV', 'targetIndex');

export function OneHotTruthPage() {
  const [vocabSize, setVocabSize] = useState(5);
  const [targetId, setTargetId] = useState(2);

  const safeVocab = Math.max(1, Math.min(64, vocabSize | 0));
  const safeTarget = Math.max(0, Math.min(safeVocab - 1, targetId | 0));

  const y = Array.from({ length: safeVocab }, (_, i) =>
    i === safeTarget ? 1 : 0
  );

  return (
    <div>
      <h2>One-Hot Truth Vector</h2>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        A one-hot vector encodes a categorical label as a vector that's all
        zeros except for a single 1 at the position of the correct class. It's
        the "true distribution" — 100% probability on the right answer, 0% on
        everything else — and it's what cross-entropy loss compares the model's
        predicted distribution against.
      </p>
      <ul style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        <li>
          Used in the general loss form{' '}
          <code>L = -Σᵢ yᵢ · log(pᵢ)</code> — multiplying by zero everywhere
          except the target slot is what collapses it to{' '}
          <code>-log(p_target)</code>.
        </li>
        <li>
          Appears explicitly in the gradient <code>∂L/∂z = p − y</code>, the
          signal the optimizer uses to nudge logits.
        </li>
        <li>
          Rarely materialized in practice — for vocab sizes of 50k+, frameworks
          index directly with the integer target instead of building the full
          zero-filled vector.
        </li>
      </ul>
      <FormulaDisplay
        latex={FORMULAS.oneHot01}
      />
      <Legend entries={LEGEND} />

      <h3>Inputs</h3>
      <label>
        vocab size V:&nbsp;
        <input
          type="number"
          value={vocabSize}
          min={1}
          max={64}
          onChange={(e) => setVocabSize(Number(e.target.value))}
          style={{ width: '4rem' }}
        />
      </label>
      &nbsp;&nbsp;
      <label>
        target:&nbsp;
        <input
          type="number"
          value={targetId}
          min={0}
          max={safeVocab - 1}
          onChange={(e) => setTargetId(Number(e.target.value))}
          style={{ width: '4rem' }}
        />
      </label>

      <h3>y</h3>
      <VectorBars data={y} />
    </div>
  );
}
