import { color } from '../theme';

interface ChangeBadgeProps {
  prev: number;
  delta: number;
  decimals?: number;
  signedPrev?: boolean;
}

// Shown above a cell while a single training step is annotated: the previous
// value (muted) and the change (green up / red down / black no-change).
export function ChangeBadge({ prev, delta, decimals = 2, signedPrev = false }: ChangeBadgeProps) {
  const deltaColor =
    delta > 0 ? color.success.fg : delta < 0 ? color.error.fg : color.text.primary;
  return (
    <div
      style={{
        fontSize: '0.6rem',
        lineHeight: 1.1,
        marginBottom: '0.1rem',
        whiteSpace: 'nowrap',
      }}
    >
      <span style={{ color: color.text.muted }}>
        {signedPrev && prev >= 0 ? '+' : ''}
        {prev.toFixed(decimals)}
      </span>{' '}
      <span style={{ color: deltaColor, fontWeight: 'bold' }}>
        {delta > 0 ? '+' : ''}
        {delta.toFixed(decimals)}
      </span>
    </div>
  );
}
