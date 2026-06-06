import { useCallback, useState } from 'react';

import { api } from '../api';
import { CodePanel } from '../components/CodePanel';
import { ErrorBanner } from '../components/ErrorBanner';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { buildLegend } from '../legendEntries';
import { MatrixEditor } from '../components/MatrixEditor';
import { PresetPicker } from '../components/PresetPicker';
import { useDebouncedApi } from '../hooks/useDebouncedApi';
import { positionEmbedPresets } from '../presets';
import type { PositionEmbedRequest, Vector } from '../types';
import { MatrixGrid } from '../viz/MatrixGrid';
import { VectorBars } from '../viz/VectorBars';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('positionVectorP', 'positionMatrixP', 'timeStepT');

export function PositionEmbedPage() {
  const [input, setInput] = useState<PositionEmbedRequest>(
    positionEmbedPresets[0].value
  );

  const fetcher = useCallback((req: PositionEmbedRequest) => api.positionEmbed(req), []);
  const { data, error } = useDebouncedApi(input, fetcher);

  return (
    <div>
      <h2>Position Embedding</h2>
      <PresetPicker presets={positionEmbedPresets} onPick={setInput} />
      <FormulaDisplay latex={data?.formula_latex ?? FORMULAS.positionEmbed01} />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <MatrixEditor
        label="P (T × n_embd)"
        value={input.P}
        onChange={(P) => setInput((s) => ({ ...s, P }))}
      />
      <label>
        t:&nbsp;
        <input
          type="number"
          value={input.t}
          onChange={(e) =>
            setInput((s) => ({ ...s, t: Number(e.target.value) }))
          }
          style={{ width: '4rem' }}
        />
      </label>

      <h3>Visualization</h3>
      <MatrixGrid data={input.P} highlightRow={input.t} />

      <h3>Result</h3>
      {data && Array.isArray(data.result) && !Array.isArray(data.result[0]) && (
        <VectorBars data={data.result as Vector} />
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
