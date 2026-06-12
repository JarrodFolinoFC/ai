import type {
  BigramBackwardRequest,
  CombinedInputRequest,
  CrossEntropyLossRequest,
  LowerTriMaskRequest,
  NormalizeRowsRequest,
  PositionEmbedRequest,
  SgdStepRequest,
  SoftmaxRequest,
  TokenEmbedRequest,
  UnembedHeadRequest,
} from './types';
import type { Preset } from './components/PresetPicker';

const tinyE = [
  [0.1, 0.2, 0.3, 0.4],
  [1.0, 0.0, -1.0, 0.5],
  [0.5, 0.5, 0.5, 0.5],
  [-0.1, 0.9, 0.3, -0.7],
  [0.0, 0.0, 1.0, 0.0],
  [0.7, -0.3, 0.2, 0.1],
];

const tinyP = [
  [0.0, 0.0, 0.0, 0.0],
  [0.5, -0.5, 0.0, 0.0],
  [0.0, 0.0, 0.5, -0.5],
  [-0.5, 0.5, 0.0, 0.0],
  [0.0, 0.0, -0.5, 0.5],
];

export const tokenEmbedPresets: Preset<TokenEmbedRequest>[] = [
  { label: 'V=6, n_embd=4, idx=2', value: { E: tinyE, idx: 2 } },
  { label: 'V=6, n_embd=4, idx=0', value: { E: tinyE, idx: 0 } },
];

export const positionEmbedPresets: Preset<PositionEmbedRequest>[] = [
  { label: 'T=5, n_embd=4, t=2', value: { P: tinyP, t: 2 } },
  { label: 'T=5, n_embd=4, t=0', value: { P: tinyP, t: 0 } },
];

export const combinedInputPresets: Preset<CombinedInputRequest>[] = [
  {
    label: 'idx=2 at t=2',
    value: { E: tinyE, P: tinyP, idx: 2, t: 2 },
  },
  {
    label: 'idx=0 at t=0',
    value: { E: tinyE, P: tinyP, idx: 0, t: 0 },
  },
];

export const lowerTriMaskPresets: Preset<LowerTriMaskRequest>[] = [
  { label: 'T=4', value: { T: 4 } },
  { label: 'T=8', value: { T: 8 } },
];

export const normalizeRowsPresets: Preset<NormalizeRowsRequest>[] = [
  {
    label: 'lower-tri T=4',
    value: {
      M: [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
      ],
    },
  },
  {
    label: 'random integers',
    value: {
      M: [
        [2, 1, 1],
        [0, 4, 0],
        [3, 3, 3],
      ],
    },
  },
];

export const unembedHeadPresets: Preset<UnembedHeadRequest>[] = [
  {
    label: 'n_embd=2, V=3',
    value: {
      h: [1.0, 2.0],
      W_head: [
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 3.0],
      ],
      b: [0.0, 0.0, 0.0],
    },
  },
];

export const softmaxPresets: Preset<SoftmaxRequest>[] = [
  { label: 'flat (uniform out)', value: { logits: [0, 0, 0, 0] } },
  { label: 'increasing', value: { logits: [1, 2, 3, 4] } },
  { label: 'one big logit', value: { logits: [0, 0, 5, 0] } },
  { label: 'huge (stability test)', value: { logits: [1000, 1001, 1002] } },
];

export const crossEntropyLossPresets: Preset<CrossEntropyLossRequest>[] = [
  { label: 'good prediction', value: { probs: [0.05, 0.9, 0.05], target_id: 1 } },
  { label: 'bad prediction', value: { probs: [0.9, 0.05, 0.05], target_id: 1 } },
  { label: 'uniform', value: { probs: [0.25, 0.25, 0.25, 0.25], target_id: 0 } },
];

export const bigramBackwardPresets: Preset<BigramBackwardRequest>[] = [
  { label: 'confident wrong', value: { probs: [0.8, 0.1, 0.1], target_id: 2 } },
  { label: 'confident right', value: { probs: [0.1, 0.8, 0.1], target_id: 1 } },
  { label: 'uniform', value: { probs: [0.25, 0.25, 0.25, 0.25], target_id: 3 } },
];

export const sgdStepPresets: Preset<SgdStepRequest>[] = [
  {
    label: 'small step',
    value: { weight_row: [1.0, 2.0, 3.0], grad_row: [0.5, -1.0, 0.0], lr: 0.1, n_examples: 2 },
  },
  {
    label: 'big lr',
    value: { weight_row: [0.0, 0.0, 0.0], grad_row: [1.0, -1.0, 0.5], lr: 1.0, n_examples: 1 },
  },
];
