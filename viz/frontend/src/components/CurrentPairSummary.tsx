import { color } from '../theme';

interface CurrentPairSummaryProps {
  vocab: readonly string[];
  // Vocab indices of the current training pair.
  prev: number;
  target: number;
  // Whether training has started (false shows the upcoming pair instead).
  started: boolean;
}

// Compact readout of the training pair in play: its prev (input) and target
// (next) tokens, with vocab ids.
export function CurrentPairSummary({ vocab, prev, target, started }: CurrentPairSummaryProps) {
  return (
    <div style={{ whiteSpace: 'nowrap' }}>
      {started ? 'current pair:' : 'next pair:'}
      <br />
      prev=<strong style={{ color: color.highlight }}>{vocab[prev]}</strong> ({prev})
      <br />
      target=<strong style={{ color: color.text.emphasis }}>{vocab[target]}</strong>{' '}
      ({target})
    </div>
  );
}
