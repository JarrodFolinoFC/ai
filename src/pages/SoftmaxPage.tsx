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
import { softmaxPresets } from '../presets';
import type { SoftmaxRequest, Vector } from '../types';
import { VectorBars } from '../viz/VectorBars';

import { color, font } from '../theme';
import { FORMULAS } from '../formulas';

const LEGEND = buildLegend('logitsZ', 'logitZi', 'maxLogit', 'softmaxProbPi');

export function SoftmaxPage() {
  const [input, setInput] = useState<SoftmaxRequest>(softmaxPresets[0].value);

  const fetcher = useCallback((req: SoftmaxRequest) => api.softmax(req), []);
  const { data, error } = useDebouncedApi(input, fetcher);

  const probs = (data?.result as Vector | undefined) ?? [];
  const sumProbs = probs.reduce((a, b) => a + b, 0);

  return (
    <div>
      <h2>Softmax</h2>
      <PresetPicker presets={softmaxPresets} onPick={setInput} />
      <FormulaDisplay
        latex={
          data?.formula_latex ??
          FORMULAS.softmax01
        }
      />
      <Legend entries={LEGEND} />
      <ErrorBanner message={error} />

      <h3>Inputs</h3>
      <VectorEditor
        label="logits"
        value={input.logits}
        onChange={(logits) => setInput({ logits })}
      />

      <h3>Probabilities</h3>
      {probs.length > 0 && <VectorBars data={probs} />}
      {probs.length > 0 && (
        <div style={{ fontSize: font.size.sm, color: color.text.secondary }}>
          Σ = {sumProbs.toFixed(6)}
        </div>
      )}

      <CodePanel source={data?.source_code} />
    </div>
  );
}
