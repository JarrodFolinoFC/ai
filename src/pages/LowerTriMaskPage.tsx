import { useCallback, useState } from 'react';

import { api } from '../api';
import { CodePanel } from '../components/CodePanel';
import { ErrorBanner } from '../components/ErrorBanner';
import { FormulaDisplay } from '../components/FormulaDisplay';
import { Legend } from '../components/Legend';
import { buildLegend } from '../legendEntries';
import { PresetPicker } from '../components/PresetPicker';
import { useDebouncedApi } from '../hooks/useDebouncedApi';
import { lowerTriMaskPresets } from '../presets';
import type { LowerTriMaskRequest, Matrix } from '../types';
import { MatrixGrid } from '../viz/MatrixGrid';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('maskEntryMij', 'rowIdxI', 'colIdxJ');

export function LowerTriMaskPage() {
  const [input, setInput] = useState<LowerTriMaskRequest>(
    lowerTriMaskPresets[0].value
  );

  const fetcher = useCallback(
    (req: LowerTriMaskRequest) => api.lowerTriMask(req),
    []
  );
  const { data, error } = useDebouncedApi(input, fetcher);

  return (
    <div>
      <h2>Lower-Triangular Mask</h2>
      <PresetPicker presets={lowerTriMaskPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.lowerTriMask01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <label>
        T:&nbsp;
        <input
          type="number"
          min={1}
          max={32}
          value={input.T}
          onChange={(e) => setInput({ T: Number(e.target.value) })}
          style={{ width: '4rem' }}
        />
      </label>

      <h3>Visualization</h3>
      {data && Array.isArray(data.result) && Array.isArray(data.result[0]) && (
        <MatrixGrid data={data.result as Matrix} />
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
