import { useState } from 'react';
import { BigramFlow } from '../components/BigramFlow';
import { TrainingCorpusOptions } from '../components/TrainingCorpusOptions';
import { VOCAB } from '../consts';
import { CORPUS_OPTIONS } from '../corpora';

export function Stage1FlowPage() {
  const [corpusIndex, setCorpusIndex] = useState<number>(0);
  return (
    <>
      <TrainingCorpusOptions
        onChange={setCorpusIndex}
      />
      <BigramFlow vocab={VOCAB} corpus={CORPUS_OPTIONS[corpusIndex]} />
    </>
  );
}
