import { InputNumber } from 'antd';
import type { Vector } from '../types';

interface Props {
  value: Vector;
  onChange: (next: Vector) => void;
  label?: string;
}

export function VectorEditor({ value, onChange, label }: Props) {
  const setCell = (i: number, n: number | null) => {
    if (n === null) return;
    const next = value.slice();
    next[i] = n;
    onChange(next);
  };

  return (
    <div>
      {label && <div style={{ fontWeight: 600 }}>{label}</div>}
      <div style={{ display: 'flex', gap: '4px' }}>
        {value.map((cell, i) => (
          <InputNumber
            key={i}
            value={cell}
            step="any"
            onChange={(n) => setCell(i, n as number | null)}
            style={{ width: '4.5rem' }}
            aria-label={`${label ?? 'vector'}[${i}]`}
          />
        ))}
      </div>
    </div>
  );
}
