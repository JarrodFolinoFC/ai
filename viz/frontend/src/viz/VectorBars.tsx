import * as d3 from 'd3';
import { useEffect, useRef } from 'react';

import type { Vector } from '../types';
import { color } from '../theme';

interface Props {
  data: Vector;
  width?: number;
  height?: number;
}

export function VectorBars({ data, width = 320, height = 120 }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);
    if (data.length === 0) return;

    const x = d3
      .scaleBand<number>()
      .domain(d3.range(data.length))
      .range([0, width])
      .padding(0.1);

    const maxAbs = Math.max(1e-9, d3.max(data.map((v) => Math.abs(v))) ?? 1);
    const y = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([height, 0]);

    const zero = y(0);

    svg
      .selectAll('rect.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', (_, i) => x(i)!)
      .attr('width', x.bandwidth())
      .attr('y', (v) => Math.min(zero, y(v)))
      .attr('height', (v) => Math.abs(y(v) - zero))
      .attr('fill', (v) => (v >= 0 ? color.positive : color.negative));

    svg
      .selectAll('text.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', (_, i) => (x(i) ?? 0) + x.bandwidth() / 2)
      .attr('y', (v) => (v >= 0 ? y(v) - 2 : y(v) + 12))
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .text((v) => v.toFixed(2));
  }, [data, width, height]);

  return <svg ref={ref} />;
}
