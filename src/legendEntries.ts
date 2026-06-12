import type { LegendEntry } from './components/Legend';

// Canonical registry of every legend entry used across the visualizer pages.
// Keyed by a semantic name (NOT the LaTeX symbol) because the same symbol means
// different things on different pages — e.g. `t` is a query position in attention
// but a time-step for position embeddings, and `h` is a combined input embedding
// in one place and a hidden state in another. Near-duplicate wordings of the same
// concept are collapsed to a single canonical entry.
export const LEGEND_ENTRIES = {
  // — token / position embeddings —
  tokenEmbeddingX: { symbol: 'x', meaning: 'token embedding for the chosen index' },
  embeddingMatrixE: { symbol: 'E', meaning: 'token embedding matrix [V, d_model]' },
  tokenIdx: { symbol: '\\text{idx}', meaning: 'token id (row to look up in E)' },
  positionVectorP: { symbol: 'p', meaning: 'position embedding for time-step t' },
  positionMatrixP: {
    symbol: 'P',
    meaning: 'position embedding matrix [T, d_model] (sinusoidal)',
  },
  timeStepT: { symbol: 't', meaning: 'time-step within the context window' },

  // — combined input / hidden states —
  combinedInputH: { symbol: 'h', meaning: 'combined input embedding (token + position)' },
  hiddenStateH: { symbol: 'h', meaning: 'hidden state for this position' },
  hiddenHj: { symbol: 'h_j', meaning: 'hidden vector at key position j (E + P)' },
  mixedHiddenHPrime: { symbol: "h'", meaning: 'mixed hidden vector after attention' },
  mixedHiddenHtPrime: {
    symbol: "h'_t",
    meaning: 'mixed hidden vector at the query position',
  },

  // — attention —
  queryPosT: { symbol: 't', meaning: 'query position (the row being mixed)' },
  keyPosJ: { symbol: 'j', meaning: 'key position (token being attended to)' },
  embedDimD: { symbol: 'd', meaning: 'embedding dimension index' },
  rawScoreSj: { symbol: 's_j', meaning: 'raw attention score for key position j' },
  maskedScoreSjPrime: { symbol: "s'_j", meaning: 'score after causal mask' },
  attnWeightAj: {
    symbol: 'a_j',
    meaning: 'attention weight for key j (softmax of masked scores)',
  },
  negInfMasked: { symbol: '-\\infty', meaning: 'masked-out value (softmax → 0)' },

  // — masks / matrix indices —
  maskEntryMij: { symbol: 'M_{i,j}', meaning: 'mask entry at row i, column j' },
  rowIdxI: { symbol: 'i', meaning: 'row index (query position)' },
  colIdxJ: { symbol: 'j', meaning: 'column index (key position)' },
  normWeightWij: {
    symbol: 'W_{i,j}',
    meaning: 'row-normalized weight at row i, column j',
  },
  rowSumNormalizer: {
    symbol: '\\sum_j M_{i,j}',
    meaning: 'sum of row i (the normalizer)',
  },

  // — unembedding / logits —
  logitsVec: { symbol: '\\text{logits}', meaning: 'raw scores over the vocabulary' },
  unembedHeadW: {
    symbol: 'W_{\\text{head}}',
    meaning: 'unembedding (output projection) matrix',
  },
  outputBiasB: { symbol: 'b', meaning: 'output bias vector' },

  // — softmax —
  logitsZ: { symbol: 'z', meaning: 'logits vector (raw scores)' },
  logitZi: { symbol: 'z_i', meaning: 'i-th logit' },
  maxLogit: {
    symbol: '\\max(z)',
    meaning: 'largest logit (subtracted for numerical stability)',
  },
  softmaxProbPi: { symbol: 'p_i', meaning: 'softmax probability for class i' },
  softmaxProbsP: { symbol: 'p', meaning: 'softmax probabilities (forward pass output)' },

  // — loss / one-hot truth —
  ceLoss: { symbol: '\\mathcal{L}', meaning: 'cross-entropy loss for one example' },
  probTarget: {
    symbol: 'p_{\\text{target}}',
    meaning: 'softmax probability assigned to the true class',
  },
  epsilon: { symbol: '10^{-12}', meaning: 'tiny epsilon to avoid log(0)' },
  oneHotTruthY: {
    symbol: 'y',
    meaning: 'one-hot truth vector (1 at the target index, 0 elsewhere)',
  },
  oneHotYi: {
    symbol: 'y_i',
    meaning: 'value at position i (1 if i is the target, else 0)',
  },
  vocabSizeV: { symbol: 'V', meaning: 'vocabulary size (length of y)' },
  targetIndex: { symbol: '\\text{target}', meaning: 'index of the correct class' },
  oneHotTarget: {
    symbol: '\\text{onehot}(\\text{target})',
    meaning: 'one-hot vector with 1 at the true class index',
  },
  gradLogitsZ: {
    symbol: '\\nabla_z \\mathcal{L}',
    meaning: 'gradient of loss w.r.t. logits z',
  },

  // — SGD step —
  updatedWeightsWPrime: { symbol: "w'", meaning: 'updated weights after the step' },
  currentWeightsW: { symbol: 'w', meaning: 'current weights (before the step)' },
  learningRateEta: { symbol: '\\eta', meaning: 'learning rate' },
  batchSizeN: {
    symbol: 'N',
    meaning: 'batch size (number of examples averaged over)',
  },
  summedGradG: { symbol: 'g', meaning: 'summed gradient over the batch' },
} satisfies Record<string, LegendEntry>;

export type LegendKey = keyof typeof LEGEND_ENTRIES;

// Build an ordered legend from the keys a page needs:
//   buildLegend('embeddingMatrixE', 'tokenIdx')
export function buildLegend(...keys: LegendKey[]): LegendEntry[] {
  return keys.map((k) => LEGEND_ENTRIES[k]);
}
