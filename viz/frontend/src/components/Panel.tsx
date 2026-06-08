import type { CSSProperties, ReactNode } from 'react';
import { Card } from 'antd';

interface PanelProps {
  // Card header title; omit for a header-less bordered container.
  title?: ReactNode;
  // Greyed out (e.g. before training has started).
  dimmed?: boolean;
  children: ReactNode;
}

const dimStyle: CSSProperties = { opacity: 0.4, filter: 'grayscale(1)' };

// Standard titled container for the visualization panels — a compact antd Card
// so every panel shares the same header, border, radius, and padding.
export function Panel({ title, dimmed = false, children }: PanelProps) {
  return (
    <Card size="small" title={title} style={dimmed ? dimStyle : undefined}>
      {children}
    </Card>
  );
}
