import { color, space } from '../theme';

interface CorpusPositionTrackerProps {
  // Index of the flashed training pair (prev token), or null when not started.
  pairIdx: number | null;
  pairsLength: number;
  corpusTokens: string[];
}

// Shows the corpus with the current training pair highlighted: the prev token
// ("a", input) and the next token ("b", target).
export function CorpusPositionTracker({
  pairIdx,
  pairsLength,
  corpusTokens,
}: CorpusPositionTrackerProps) {
  const markIdx = pairIdx ?? -1;
  return (
    <div style={{ flex: '1 1 28ch', minWidth: '20ch' }}>
      <div style={{ color: color.text.secondary, marginBottom: space.sm }}>
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
      <div style={{ lineHeight: 2.4 }}>
        {corpusTokens.map((t, i) => {
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
    </div>
  );
}
