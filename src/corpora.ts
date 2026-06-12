export const CORPUS_29 =
  'the cat on the sofa . the dog in the hat . the cat in the hat . the dog on the sofa . the cat on the sofa';

export const CORPUS_50 =
  'the cat on the sofa . the dog in the hat . the cat in the hat . the dog on the sofa . ' +
  'the cat on the hat . the dog on the sofa . the cat . the dog . the hat . the sofa . the cat ' +
  'on the sofa . the dog in the hat . the cat';

export const CORPUS_75 = `${CORPUS_50} ${CORPUS_29}`;

export const CORPUS_OPTIONS = [
  CORPUS_29,
  CORPUS_50,
  CORPUS_75,
] as const;
