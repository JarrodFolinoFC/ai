import { type ReactNode } from 'react';
import { color, radius, font } from '../theme';

export function ExplainMore({
  label = 'Explain it more',
  children,
}: {
  label?: string;
  children: ReactNode;
}) {
  return (
    <details
      style={{
        maxWidth: font.prose,
        margin: '0.5rem 0',
        padding: '0.5rem 0.75rem',
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        background: color.bg.surface,
      }}
    >
      <summary
        style={{
          cursor: 'pointer',
          fontWeight: 'bold',
          color: color.info.fg,
          fontSize: font.size.md,
        }}
      >
        {label}
      </summary>
      <p
        style={{
          color: color.text.secondary,
          fontSize: font.size.sm,
          lineHeight: 1.5,
          marginBottom: 0,
        }}
      >
        {children}
      </p>
    </details>
  );
}
