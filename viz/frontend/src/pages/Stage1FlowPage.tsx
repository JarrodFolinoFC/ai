import { useState } from 'react';
import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { BigramFlow } from '../components/BigramFlow';
import { TrainingCorpusOptions } from '../components/TrainingCorpusOptions';
import { VOCAB } from '../consts';

export function Stage1FlowPage() {
  const [corpusKey, setCorpusKey] = useState<CorpusKey>('25');
  return (
    <>
      <TrainingCorpusOptions
        corpusKey={corpusKey}
        onChange={setCorpusKey}
        note="(changing this resets training)"
      />
      <BigramFlow vocab={VOCAB} corpus={CORPUS_OPTIONS[corpusKey]} />
    </>
  );
}
