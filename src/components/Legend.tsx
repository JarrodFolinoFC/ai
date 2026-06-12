import katex from 'katex';
import { useEffect, useRef } from 'react';

import { color, space, font } from '../theme';

export interface LegendEntry {
  symbol: string;
  meaning: string;
}

interface LegendProps {
  entries: LegendEntry[];
}

interface LegendRowProps {
  entry: LegendEntry;
}

function LegendRow({ entry }: LegendRowProps) {
  const symbolRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!symbolRef.current) return;
    katex.render(entry.symbol, symbolRef.current, {
      displayMode: false,
      throwOnError: false,
    });
  }, [entry.symbol]);

  return (
    <div
      data-testid="legend-row"
      style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr',
        columnGap: space.md,
        alignItems: 'baseline',
        padding: '0.15rem 0',
      }}
    >
      <span ref={symbolRef} />
      <span>{entry.meaning}</span>
    </div>
  );
}

export function Legend({ entries }: LegendProps) {
  if (entries.length === 0) return null;
  return (
    <div
      data-testid="legend"
      style={{
        marginTop: space.sm,
        marginBottom: space.md,
        fontSize: font.size.sm,
        color: color.text.secondary,
      }}
    >
      {entries.map((entry) => (
        <LegendRow key={entry.symbol} entry={entry} />
      ))}
    </div>
  );
}
