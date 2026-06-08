import type { CSSProperties, ReactNode } from 'react';
import { Card } from 'antd';

import { useStep } from '../stepContext';
import { LiveBadge } from './LiveBadge';

interface PanelProps {
  // Card header title; omit for a header-less bordered container.
  title?: ReactNode;
  // Greyed out (e.g. before training has started).
  dimmed?: boolean;
  // When true, prefix the title with a "live" badge (pulsing dot + current step)
  // to mark a card whose contents update on every training step.
  live?: boolean;
  // Override the step shown in the live badge (e.g. a previous-step card). The
  // grey-until-trained logic still uses the real current step.
  badgeStep?: number;
  children: ReactNode;
}

const dimStyle: CSSProperties = { opacity: 0.4, filter: 'grayscale(1)' };

// Standard titled container for the visualization panels — a compact antd Card
// so every panel shares the same header, border, radius, and padding.
export function Panel({ title, dimmed = false, live = false, badgeStep, children }: PanelProps) {
  const step = useStep();
  // The step this card represents (badgeStep for cards showing another step,
  // e.g. the previous-step card). Live cards stay greyed out while the step
  // they show is the untrained initial state (step 0).
  const shownStep = badgeStep ?? step;
  const greyed = dimmed || (live && shownStep === 0);

  const header =
    live && title !== undefined ? (
      <span style={{ display: 'inline-flex', alignItems: 'center' }}>
        <LiveBadge step={badgeStep} />
        {title}
      </span>
    ) : (
      title
    );

  return (
    <Card size="small" title={header} style={greyed ? dimStyle : undefined}>
      {children}
    </Card>
  );
}
