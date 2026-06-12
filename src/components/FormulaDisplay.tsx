import katex from 'katex';
import { useEffect, useRef } from 'react';

interface Props {
  latex: string;
  // Render inline (within a line of text) rather than as a centered block.
  inline?: boolean;
}

export function FormulaDisplay({ latex, inline = false }: Props) {
  const divRef = useRef<HTMLDivElement>(null);
  const spanRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = inline ? spanRef.current : divRef.current;
    if (!el) return;
    katex.render(latex, el, {
      displayMode: !inline,
      throwOnError: false,
    });
  }, [latex, inline]);

  return inline ? (
    <span ref={spanRef} data-testid="formula-display" />
  ) : (
    <div ref={divRef} data-testid="formula-display" />
  );
}
