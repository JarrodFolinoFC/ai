import { InputNumber } from 'antd';
import type { Matrix } from '../types';

interface Props {
  value: Matrix;
  onChange: (next: Matrix) => void;
  label?: string;
}

export function MatrixEditor({ value, onChange, label }: Props) {
  const setCell = (i: number, j: number, n: number | null) => {
    if (n === null) return;
    const next = value.map((row) => row.slice());
    next[i][j] = n;
    onChange(next);
  };

  return (
    <div>
      {label && <div style={{ fontWeight: 600 }}>{label}</div>}
      <table>
        <tbody>
          {value.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} style={{ padding: '2px' }}>
                  <InputNumber
                    value={cell}
                    step="any"
                    onChange={(n) => setCell(i, j, n as number | null)}
                    style={{ width: '4.5rem' }}
                    aria-label={`${label ?? 'matrix'}[${i}][${j}]`}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
