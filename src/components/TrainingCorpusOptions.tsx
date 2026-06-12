import { CORPUS_OPTIONS  } from '../corpora';
import { space } from '../theme';

interface Props {
  onChange: (key: number) => void;
}

export function TrainingCorpusOptions({
  onChange
}: Props) {
  return (
    <div style={{ margin: `${space.sm} 0` }}>
      <span style={{ marginRight: space.sm }}>Corpus size:</span>
      {CORPUS_OPTIONS.map((c, i) => (
        <label key={c.length} style={{ marginRight: space.md }}>
          <input
            type="radio"
            name={'corpus-size'}
            value={i}
            onChange={() => onChange(i)}
          />
          &nbsp;{c.trim().split(/\s+/).length}
        </label>
      ))}
    </div>
  );
}
