import { color } from '../theme';
import { useStep } from '../stepContext';

interface LiveBadgeProps {
  // Override the displayed step (e.g. a card showing the previous step's state).
  // Defaults to the current step from context.
  step?: number;
}

// Title indicator for live cards. Before any training step it shows a hollow
// "untrained" icon with no number; once stepping has started it becomes a
// pulsing dot plus the current step.
export function LiveBadge({ step: override }: LiveBadgeProps = {}) {
  const ctxStep = useStep();
  if (ctxStep === null) return null;

  // The step this badge represents (override for previous-step cards).
  const step = override ?? ctxStep;

  // Untrained when the represented step is the initial state (step 0).
  if (step === 0) {
    return (
      <span
        title="Not trained yet — take a step"
        style={{ display: 'inline-flex', alignItems: 'center', marginRight: '0.45rem' }}
      >
        <span
          style={{
            width: '0.5rem',
            height: '0.5rem',
            borderRadius: '50%',
            border: `1.5px solid ${color.text.muted}`,
            background: 'transparent',
          }}
        />
      </span>
    );
  }

  return (
    <span
      title="Updates live each training step"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.3rem',
        marginRight: '0.45rem',
        fontSize: '0.7rem',
        fontWeight: 'normal',
        color: color.positive,
      }}
    >
      <span
        style={{
          width: '0.5rem',
          height: '0.5rem',
          borderRadius: '50%',
          background: color.positive,
          animation: 'live-pulse 1.6s ease-in-out infinite',
        }}
      />
      {step}
    </span>
  );
}
