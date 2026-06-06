import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { tokenize } from '../funcs';
import { color, space, radius, font } from '../theme';

interface Props {
  corpusKey: CorpusKey;
  onChange: (key: CorpusKey) => void;
  // Optional caption after the size options, e.g. "(changing this resets training)".
  note?: string;
  // Radio-group name; override when more than one corpus selector exists per page.
  name?: string;
}

// Standardized "Training corpus" block: a token-count heading, the size selector
// (driven by CORPUS_OPTIONS), and the corpus text. Pages own the selected key so
// they can derive the corpus for their own logic.
export function TrainingCorpus({
  corpusKey,
  onChange,
  note,
  name = 'corpus-size',
}: Props) {
  const corpus = CORPUS_OPTIONS[corpusKey];
  const total = tokenize(corpus).length;

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>Training corpus ({total} tokens)</h3>
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
      <pre
        style={{
          maxWidth: font.prose,
          padding: space.md,
          background: color.bg.subtle,
          border: `1px solid ${color.border.default}`,
          borderRadius: radius.sm,
          whiteSpace: 'pre-wrap',
          fontFamily: 'monospace',
        }}
      >
        {corpus}
      </pre>
    </div>
  );
}
