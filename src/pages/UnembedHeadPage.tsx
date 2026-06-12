import { useCallback, useState } from 'react';

import { api } from '../api';
import { CodePanel } from '../components/CodePanel';
import { ErrorBanner } from '../components/ErrorBanner';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { buildLegend } from '../legendEntries';
import { MatrixEditor } from '../components/MatrixEditor';
import { PresetPicker } from '../components/PresetPicker';
import { VectorEditor } from '../components/VectorEditor';
import { useDebouncedApi } from '../hooks/useDebouncedApi';
import { unembedHeadPresets } from '../presets';
import type { UnembedHeadRequest, Vector } from '../types';
import { MatrixGrid } from '../viz/MatrixGrid';
import { VectorBars } from '../viz/VectorBars';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('logitsVec', 'hiddenStateH', 'unembedHeadW', 'outputBiasB');

export function UnembedHeadPage() {
  const [input, setInput] = useState<UnembedHeadRequest>(
    unembedHeadPresets[0].value
  );

  const fetcher = useCallback(
    (req: UnembedHeadRequest) => api.unembedHead(req),
    []
  );
  const { data, error } = useDebouncedApi(input, fetcher);

  return (
    <div>
      <h2>Unembedding Head</h2>
      <PresetPicker presets={unembedHeadPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.unembedHead01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <VectorEditor
        label="h (n_embd)"
        value={input.h}
        onChange={(h) => setInput((s) => ({ ...s, h }))}
      />
      <MatrixEditor
        label="W_head (n_embd × V)"
        value={input.W_head}
        onChange={(W_head) => setInput((s) => ({ ...s, W_head }))}
      />
      <VectorEditor
        label="b (V)"
        value={input.b}
        onChange={(b) => setInput((s) => ({ ...s, b }))}
      />

      <h3>Visualization of W_head</h3>
      <MatrixGrid data={input.W_head} />

      <h3>Result: logits</h3>
      {data && Array.isArray(data.result) && !Array.isArray(data.result[0]) && (
        <VectorBars data={data.result as Vector} />
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
