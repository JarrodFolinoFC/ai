import { color } from '../theme';

interface RecentLossProps {
  lossHistory: number[];
}

// Mean of the most recent training losses (last 50 steps); hidden until
// training has produced at least one loss.
export function RecentLoss({ lossHistory }: RecentLossProps) {
  if (lossHistory.length === 0) return null;
  const recent = lossHistory.slice(-50);
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  return (
    <p style={{ fontFamily: 'monospace', color: color.text.secondary }}>
      recent mean loss (last {recent.length}): {mean.toFixed(3)}
    </p>
  );
}
