import { color, radius } from '../theme';

interface ForwardPassCaptionProps {
  // The prev (input) token whose W row is read.
  prevToken: string;
}

// Caption above the forward-pass tables explaining p = softmax(W[prev]).
export function ForwardPassCaption({ prevToken }: ForwardPassCaptionProps) {
  return (
    <div style={{ marginTop: 0, color: color.text.secondary }}>
      forward pass: <code>p = softmax(W[prev])</code> — read row{' '}
      <span
        style={{
          background: color.highlightBg,
          color: color.text.emphasis,
          fontWeight: 'bold',
          padding: '0 0.2rem',
          borderRadius: radius.sm,
        }}
      >
        {prevToken}
      </span>{' '}
      of W, exponentiate each logit, then divide by the total:
    </div>
  );
}
