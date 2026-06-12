// Shared toy training corpora used across the visualization pages.
// All pages draw from this single canonical corpus so the demos stay consistent.

export const CORPUS_25 =
  'the cat on the sofa . the dog in the hat . the cat in the hat . the dog on the sofa . the cat on the sofa';

export const CORPUS_50 =
  'the cat on the sofa the dog in the hat the cat in the hat the dog on the sofa ' +
  'the cat on the hat the dog in the sofa the cat the dog the hat the sofa the cat ' +
  'on the sofa the dog in the hat the cat';

// 50 + 25 = 75 tokens, composed from the canonical corpora above.
export const CORPUS_75 = `${CORPUS_50} ${CORPUS_25}`;

export const CORPUS_OPTIONS = {
  '25': CORPUS_25,
  '50': CORPUS_50,
  '75': CORPUS_75,
} as const;

export type CorpusKey = keyof typeof CORPUS_OPTIONS;
