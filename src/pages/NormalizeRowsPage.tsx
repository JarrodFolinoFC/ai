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
import { normalizeRowsPresets } from '../presets';
import type { Matrix, NormalizeRowsRequest } from '../types';
import { MatrixGrid } from '../viz/MatrixGrid';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('normWeightWij', 'maskEntryMij', 'rowSumNormalizer');

export function NormalizeRowsPage() {
  const [input, setInput] = useState<NormalizeRowsRequest>(
    normalizeRowsPresets[0].value
  );

  const fetcher = useCallback(
    (req: NormalizeRowsRequest) => api.normalizeRows(req),
    []
  );
  const { data, error } = useDebouncedApi(input, fetcher);

  return (
    <div>
      <h2>Row-Normalize a Mask</h2>
      <PresetPicker presets={normalizeRowsPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.normalizeRows01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Input M</h3>
      <MatrixEditor
        label="M"
        value={input.M}
        onChange={(M) => setInput({ M })}
      />
      <h3>Input grid</h3>
      <MatrixGrid data={input.M} />

      <h3>Output W (row-stochastic)</h3>
      {data && Array.isArray(data.result) && Array.isArray(data.result[0]) && (
        <MatrixGrid data={data.result as Matrix} />
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
