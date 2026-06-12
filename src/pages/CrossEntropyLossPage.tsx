import { useCallback, useState } from 'react';

import { api } from '../api';
import { CodePanel } from '../components/CodePanel';
import { ErrorBanner } from '../components/ErrorBanner';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { buildLegend } from '../legendEntries';
import { PresetPicker } from '../components/PresetPicker';
import { VectorEditor } from '../components/VectorEditor';
import { useDebouncedApi } from '../hooks/useDebouncedApi';
import { crossEntropyLossPresets } from '../presets';
import type { CrossEntropyLossRequest, Scalar } from '../types';
import { VectorBars } from '../viz/VectorBars';

import { color, radius, font } from '../theme';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('ceLoss', 'probTarget', 'epsilon', 'oneHotTruthY');

export function CrossEntropyLossPage() {
  const [input, setInput] = useState<CrossEntropyLossRequest>(
    crossEntropyLossPresets[0].value
  );

  const fetcher = useCallback(
    (req: CrossEntropyLossRequest) => api.crossEntropyLoss(req),
    []
  );
  const { data, error } = useDebouncedApi(input, fetcher);

  const loss =
    typeof data?.result === 'number' ? (data.result as Scalar) : null;

  return (
    <div>
      <h2>Cross-Entropy Loss</h2>
      <p style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        Given the model's predicted probabilities and the correct class,
        cross-entropy loss returns a scalar measuring how surprised the model
        was. Its gradient w.r.t. the logits —{' '}
        <code>p − onehot(target)</code> — is what the optimizer uses to nudge
        the logits in the right direction.
      </p>
      <ul style={{ maxWidth: font.prose, color: color.text.secondary, lineHeight: 1.5 }}>
        <li>Model says correct token has prob 1.0 → loss = 0</li>
        <li>Model says correct token has prob 0.5 → loss ≈ 0.69</li>
        <li>Model says correct token has prob 0.01 → loss ≈ 4.6</li>
        <li>Model says correct token has prob 0.0 → loss = ∞</li>
      </ul>
      <PresetPicker presets={crossEntropyLossPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.crossEntropy01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <VectorEditor
        label="probs"
        value={input.probs}
        onChange={(probs) => setInput((s) => ({ ...s, probs }))}
      />
      <label>
        target_id:&nbsp;
        <input
          type="number"
          value={input.target_id}
          onChange={(e) =>
            setInput((s) => ({ ...s, target_id: Number(e.target.value) }))
          }
          style={{ width: '4rem' }}
        />
      </label>

      <h3>probs (target highlighted)</h3>
      <VectorBars data={input.probs} />

      <h3>Loss</h3>
      <div
        style={{
          fontSize: '2rem',
          fontFamily: 'monospace',
          padding: '0.5rem 0.75rem',
          background: color.bg.subtle,
          border: `1px solid ${color.border.default}`,
          borderRadius: radius.sm,
          display: 'inline-block',
        }}
      >
        {loss !== null ? loss.toFixed(6) : '—'}
      </div>

      <CodePanel source={data?.source_code} />
    </div>
  );
}
