import { Tooltip } from 'antd';
import katex from 'katex';
import { useEffect, useRef } from 'react';

// Renders a KaTeX formula. Inherits the surrounding text color, so it shows
// correctly against the dark tooltip background.
function Formula({ latex }: { latex: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    katex.render(latex, ref.current, {
      displayMode: true,
      throwOnError: false,
    });
  }, [latex]);

  return <div ref={ref} style={{ marginTop: '0.4rem' }} />;
}

interface TermProps {
  // The word shown inline (with a dotted underline / help cursor).
  label: string;
  // Plain-language explanation shown at the top of the hover card.
  explain: string;
  // Optional KaTeX formula rendered beneath the explanation.
  formula?: string;
}

// An inline glossary term: a dotted-underlined word that reveals a hover card
// with a prose explanation plus an optional rendered math formula.
export function Term({ label, explain, formula }: TermProps) {
  const card = (
    <div style={{ maxWidth: '22rem' }}>
      <div>{explain}</div>
      {formula && <Formula latex={formula} />}
    </div>
  );

  return (
    <Tooltip title={card}>
      <span style={{ textDecoration: 'underline dotted', cursor: 'help' }}>
        {label}
      </span>
    </Tooltip>
  );
}
