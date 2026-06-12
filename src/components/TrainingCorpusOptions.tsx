import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { color, space, font } from '../theme';

interface Props {
  corpusKey: CorpusKey;
  onChange: (key: CorpusKey) => void;
  // Optional caption after the size options, e.g. "(changing this resets training)".
  note?: string;
  // Radio-group name; override when more than one corpus selector exists per page.
  name?: string;
}

// Corpus-size selector (driven by CORPUS_OPTIONS). Pages own the selected key so
// they can derive the corpus string and pass it into TrainingCorpus / their logic.
export function TrainingCorpusOptions({
  corpusKey,
  onChange,
  note,
  name = 'corpus-size',
}: Props) {
  return (
    <div style={{ margin: `${space.sm} 0` }}>
      <span style={{ marginRight: space.sm }}>size:</span>
      {(Object.keys(CORPUS_OPTIONS) as CorpusKey[]).map((k) => (
        <label key={k} style={{ marginRight: space.md }}>
          <input
            type="radio"
            name={name}
            value={k}
            checked={corpusKey === k}
            onChange={() => onChange(k)}
          />
          &nbsp;{k}
        </label>
      ))}
      {note && (
        <span style={{ color: color.text.secondary, fontSize: font.size.sm }}>
          {note}
        </span>
      )}
    </div>
  );
}
