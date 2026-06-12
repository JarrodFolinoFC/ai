import { tokenize } from '../funcs';
import { color, space, radius, font } from '../theme';
import { Panel } from './Panel';

interface Props {
  corpus?: string;
  onChange?: (key: number) => void;
  name?: string;
  pairIdx?: number | null;
  pairsLength?: number;
  currentPair?: { prev: number; target: number };
  vocab?: readonly string[];
  started?: boolean;
}

export function TrainingCorpus({
  corpus,
  pairIdx,
  pairsLength
}: Props) {
  const resolvedCorpus = corpus;
  const tokens = tokenize(resolvedCorpus || '');
  const total = tokens.length;
  const showPosition = pairIdx !== undefined;
  const markIdx = pairIdx ?? -1;
  const heading = `Training corpus (${total} tokens)`;


  if (!showPosition) {
    return (
      <div>
        <h3 style={{ marginTop: 0 }}>{heading}</h3>
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
    <Panel title={heading} live>
      <div style={{ color: color.text.secondary, margin: `${space.sm} 0` }}>
        {pairIdx !== null ? `pair ${pairIdx} of ${pairsLength}` : 'not started yet'}
        {' '}<span style={{ color: color.highlight, fontWeight: 'bold' }}>
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
