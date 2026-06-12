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
import { sgdStepPresets } from '../presets';
import type { SgdStepRequest, Vector } from '../types';
import { VectorBars } from '../viz/VectorBars';

import { space } from '../theme';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('updatedWeightsWPrime', 'currentWeightsW', 'learningRateEta', 'batchSizeN', 'summedGradG');

export function SgdStepPage() {
  const [input, setInput] = useState<SgdStepRequest>(sgdStepPresets[0].value);

  const fetcher = useCallback((req: SgdStepRequest) => api.sgdStep(req), []);
  const { data, error } = useDebouncedApi(input, fetcher);

  const after = (data?.result as Vector | undefined) ?? [];

  return (
    <div>
      <h2>SGD Step</h2>
      <PresetPicker presets={sgdStepPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.sgdStep01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <VectorEditor
        label="weight_row (before)"
        value={input.weight_row}
        onChange={(weight_row) => setInput((s) => ({ ...s, weight_row }))}
      />
      <VectorEditor
        label="grad_row"
        value={input.grad_row}
        onChange={(grad_row) => setInput((s) => ({ ...s, grad_row }))}
      />
      <div style={{ display: 'flex', gap: space.lg }}>
        <label>
          lr:&nbsp;
          <input
            type="number"
            step="any"
            value={input.lr}
            onChange={(e) =>
              setInput((s) => ({ ...s, lr: Number(e.target.value) }))
            }
            style={{ width: '5rem' }}
          />
        </label>
        <label>
          n_examples:&nbsp;
          <input
            type="number"
            value={input.n_examples}
            onChange={(e) =>
              setInput((s) => ({ ...s, n_examples: Number(e.target.value) }))
            }
            style={{ width: '5rem' }}
          />
        </label>
      </div>

      <h3>weight_row (before)</h3>
      <VectorBars data={input.weight_row} />

      <h3>weight_row (after)</h3>
      {after.length > 0 && <VectorBars data={after} />}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
