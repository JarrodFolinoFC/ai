import { useEffect, useRef, useState } from 'react';
import { Button } from 'antd';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css';

import { color, space, radius } from '../theme';

interface Props {
  source: string | undefined;
}

export function CodePanel({ source }: Props) {
  const [open, setOpen] = useState(false);
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (open && source && codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [open, source]);

  const disabled = !source;

  return (
    <div style={{ marginTop: space.lg }}>
      <Button
        size="small"
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        aria-expanded={open}
      >
        {open ? 'Hide code' : 'Show code'}
      </Button>
      {open && source && (
        <pre
          style={{
            marginTop: space.sm,
            padding: space.md,
            background: color.bg.subtle,
            border: `1px solid ${color.border.default}`,
            borderRadius: radius.sm,
            overflow: 'auto',
            fontSize: '0.8rem',
          }}
        >
          <code ref={codeRef} className="language-python">
            {source}
          </code>
        </pre>
      )}
    </div>
  );
}
