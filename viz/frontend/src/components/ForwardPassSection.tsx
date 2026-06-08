import { ForwardPassTable } from './ForwardPassTable';
import { Panel } from './Panel';

interface Pass {
  logits: number[];
  exps: number[];
  expSum: number;
  probs: number[];
}

interface ForwardPassSectionProps {
  vocab: readonly string[];
  targetIdx: number;
  prevToken: string;
  flashing: boolean;
  before: Pass;
  current: Pass;
}

export function ForwardPassSection({
  vocab,
  targetIdx,
  prevToken,
  before,
  current,
}: ForwardPassSectionProps) {
  return (
    <>
      <Panel title="before this step — drives the gradient">
        <ForwardPassTable vocab={vocab} {...before} targetIdx={targetIdx} prevToken={prevToken} />
      </Panel>
      <Panel title="after this step — result of the update">
        <ForwardPassTable vocab={vocab} {...current} targetIdx={targetIdx} prevToken={prevToken} />
      </Panel>
    </>
  );
}
