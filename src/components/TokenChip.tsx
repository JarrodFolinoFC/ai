import { color } from '../theme';

// Token chip styled like the highlighted tokens in TrainingCorpus: the prev
// ("a"/input) token gets the highlight background, the target ("b"/next) token
// gets the info background, both with bold emphasis text.
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
