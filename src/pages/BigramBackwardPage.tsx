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
import { bigramBackwardPresets } from '../presets';
import type { BigramBackwardRequest, Vector } from '../types';
import { VectorBars } from '../viz/VectorBars';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('gradLogitsZ', 'softmaxProbsP', 'oneHotTarget');

export function BigramBackwardPage() {
  const [input, setInput] = useState<BigramBackwardRequest>(
    bigramBackwardPresets[0].value
  );

  const fetcher = useCallback(
    (req: BigramBackwardRequest) => api.bigramBackward(req),
    []
  );
  const { data, error } = useDebouncedApi(input, fetcher);

  const grad = (data?.result as Vector | undefined) ?? [];

  return (
    <div>
      <h2>Bigram Backward (gradient)</h2>
      <PresetPicker presets={bigramBackwardPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.bigramBackward01
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

      <h3>probs</h3>
      <VectorBars data={input.probs} />

      <h3>grad = probs − onehot(target)</h3>
      {grad.length > 0 && <VectorBars data={grad} />}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
