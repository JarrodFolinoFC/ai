import katex from 'katex';
import { useEffect, useRef } from 'react';

interface Props {
  latex: string;
}

export function FormulaDisplay({ latex }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    katex.render(latex, ref.current, {
      displayMode: true,
      throwOnError: false,
    });
  }, [latex]);

  return <div ref={ref} data-testid="formula-display" />;
}
