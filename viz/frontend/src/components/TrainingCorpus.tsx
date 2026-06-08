import { CORPUS_OPTIONS, type CorpusKey } from '../corpora';
import { tokenize } from '../funcs';
import { color, space, radius, font } from '../theme';
import { Panel } from './Panel';
import { CurrentPairSummary } from './CurrentPairSummary';
import { TrainingCorpusOptions } from './TrainingCorpusOptions';

interface Props {
  // Corpus text to display. When omitted, falls back to deriving it from
  // `corpusKey` and rendering the built-in size selector (legacy mode). When
  // provided, the caller owns selection (e.g. via TrainingCorpusOptions) and no
  // selector is rendered.
  corpus?: string;
  corpusKey?: CorpusKey;
  onChange?: (key: CorpusKey) => void;
  // Optional caption after the size options, e.g. "(changing this resets training)".
  note?: string;
  // Radio-group name; override when more than one corpus selector exists per page.
  name?: string;
  // When provided, the corpus renders with the current training pair highlighted
  // (prev = "a"/input, next = "b"/target) instead of as plain text, and the whole
  // block becomes a card. null = not started yet. Requires pairsLength for the
  // "pair X of N" caption.
  pairIdx?: number | null;
  pairsLength?: number;
  // Current training pair; when provided its prev/target summary is shown.
  currentPair?: { prev: number; target: number };
  vocab?: readonly string[];
  started?: boolean;
}

// Standardized "Training corpus" block: a token-count heading, the size selector
// (driven by CORPUS_OPTIONS), and the corpus text. Pages own the selected key so
// they can derive the corpus for their own logic. When pairIdx is supplied the
// block becomes a card that also tracks the current training pair: the prev/next
// summary plus the corpus with that pair highlighted.
export function TrainingCorpus({
  corpus,
  corpusKey,
  onChange,
  note,
  name,
  pairIdx,
  pairsLength,
  currentPair,
  vocab,
  started = false,
}: Props) {
  const resolvedCorpus = corpus ?? (corpusKey !== undefined ? CORPUS_OPTIONS[corpusKey] : '');
  const tokens = tokenize(resolvedCorpus);
  const total = tokens.length;
  const showPosition = pairIdx !== undefined;
  const markIdx = pairIdx ?? -1;
  const heading = `Training corpus (${total} tokens)`;

  // Legacy mode: no `corpus` passed, so this component owns the size selector.
  const sizeSelector =
    corpus === undefined && corpusKey !== undefined && onChange !== undefined ? (
      <TrainingCorpusOptions corpusKey={corpusKey} onChange={onChange} note={note} name={name} />
    ) : null;

  if (!showPosition) {
    return (
      <div>
        <h3 style={{ marginTop: 0 }}>{heading}</h3>
        {sizeSelector}
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
          {resolvedCorpus}
        </pre>
      </div>
    );
  }

  return (
    <Panel title={heading}>
      {sizeSelector}
      {currentPair && vocab && (
        <CurrentPairSummary
          vocab={vocab}
          prev={currentPair.prev}
          target={currentPair.target}
          started={started}
        />
      )}
      <div style={{ color: color.text.secondary, margin: `${space.sm} 0` }}>
        corpus position (
        {pairIdx !== null ? `pair ${pairIdx} of ${pairsLength}` : 'not started yet'}
        ):{' '}
        <span style={{ color: color.highlight, fontWeight: 'bold' }}>
          a = prev (input)
        </span>
        ,{' '}
        <span style={{ color: color.text.emphasis, fontWeight: 'bold' }}>
          b = next (target)
        </span>
      </div>
      <div
        style={{
          maxWidth: font.prose,
          padding: space.md,
          paddingTop: '1.2rem',
          background: color.bg.subtle,
          border: `1px solid ${color.border.default}`,
          borderRadius: radius.sm,
          fontFamily: 'monospace',
          lineHeight: 2.4,
        }}
      >
        {tokens.map((t, i) => {
          const isPrev = i === markIdx;
          const isTarget = i === markIdx + 1;
          const marked = isPrev || isTarget;
          return (
            <span
              key={i}
              style={{
                position: 'relative',
                display: 'inline-block',
                padding: '0.05rem 0.25rem',
                borderRadius: '3px',
                background: isPrev
                  ? color.highlightBg
                  : isTarget
                    ? color.info.border
                    : 'transparent',
                color: marked ? color.text.emphasis : color.text.muted,
                fontWeight: marked ? 'bold' : 'normal',
                marginRight: space.xs,
              }}
            >
              {marked && (
                <span
                  style={{
                    position: 'absolute',
                    top: '-1rem',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: '0.75rem',
                    fontStyle: 'italic',
                    color: isPrev ? color.highlight : color.text.emphasis,
                  }}
                >
                  {isPrev ? 'a' : 'b'}
                </span>
              )}
              {t}
            </span>
          );
        })}
      </div>
    </Panel>
  );
}
