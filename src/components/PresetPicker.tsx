import { Select } from 'antd';

export interface Preset<T> {
  label: string;
  value: T;
}

interface Props<T> {
  presets: Preset<T>[];
  onPick: (value: T) => void;
}

export function PresetPicker<T>({ presets, onPick }: Props<T>) {
  return (
    <Select
      placeholder="Preset: choose…"
      style={{ minWidth: 160 }}
      onChange={(idx: number) => {
        if (presets[idx]) onPick(presets[idx].value);
      }}
      options={presets.map((p, i) => ({ label: p.label, value: i }))}
    />
  );
}
