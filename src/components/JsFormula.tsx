import { useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-javascript';
import 'prismjs/themes/prism.css';

import { color, space, radius } from '../theme';

interface JsFormulaProps {
  // The JavaScript source to render, syntax-highlighted.
  code: string;
}

// Renders a formula as a syntax-highlighted, always-visible JavaScript snippet.
// Unlike CodePanel this has no toggle — it sits inline as a worked code example.
export function JsFormula({ code }: JsFormulaProps) {
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current) Prism.highlightElement(codeRef.current);
  }, [code]);

  return (
    <pre
      style={{
        margin: 0,
        padding: space.sm,
        background: color.bg.subtle,
        border: `1px solid ${color.border.default}`,
        borderRadius: radius.sm,
        overflow: 'auto',
        fontSize: '0.8rem',
      }}
    >
      <code ref={codeRef} className="language-javascript">
        {code}
      </code>
    </pre>
  );
}
