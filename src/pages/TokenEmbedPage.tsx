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
import { tokenEmbedPresets } from '../presets';
import type { TokenEmbedRequest, Vector } from '../types';
import { MatrixGrid } from '../viz/MatrixGrid';
import { VectorBars } from '../viz/VectorBars';
import { FORMULAS } from '../formulas';

export function TokenEmbedPage() {
  const [input, setInput] = useState<TokenEmbedRequest>(
    tokenEmbedPresets[0].value
  );

  const fetcher = useCallback((req: TokenEmbedRequest) => api.tokenEmbed(req), []);
  const { data, error } = useDebouncedApi(input, fetcher);

  return (
    <div>
      <h2>Token Embedding</h2>
      <PresetPicker presets={tokenEmbedPresets} onPick={setInput} />
      <FormulaDisplay latex={data?.formula_latex ?? r_default} />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <MatrixEditor
        label="E (vocab_size × n_embd)"
        value={input.E}
        onChange={(E) => setInput((s) => ({ ...s, E }))}
      />
      <label>
        idx:&nbsp;
        <input
          type="number"
          value={input.idx}
          onChange={(e) =>
            setInput((s) => ({ ...s, idx: Number(e.target.value) }))
          }
          style={{ width: '4rem' }}
        />
      </label>

      <h3>Visualization</h3>
      <MatrixGrid data={input.E} highlightRow={input.idx} />

      <h3>Result</h3>
      {data && Array.isArray(data.result) && !Array.isArray(data.result[0]) && (
        <VectorBars data={data.result as Vector} />
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}

const r_default = FORMULAS.tokenEmbed01;

const LEGEND = buildLegend('tokenEmbeddingX', 'embeddingMatrixE', 'tokenIdx');
