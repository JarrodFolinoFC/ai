import { color } from '../theme';

export function TokenChip({ text, isPrev = false }: { text: string; isPrev?: boolean }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.05rem 0.25rem',
        borderRadius: '3px',
        background: isPrev ? color.highlightBg : color.info.border,
        color: color.text.emphasis,
        fontWeight: 'bold',
      }}
    >
      {text}
    </span>
  );
}
