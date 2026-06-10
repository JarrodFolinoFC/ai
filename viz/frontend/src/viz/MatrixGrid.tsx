import * as d3 from 'd3';
import { PRECISION } from '../consts';
import { useEffect, useRef } from 'react';

import type { Matrix } from '../types';
import { viz } from '../theme';

interface Props {
  data: Matrix;
  highlightRow?: number;
  cellSize?: number;
}

export function MatrixGrid({ data, highlightRow, cellSize = 32 }: Props) {
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    const rows = data.length;
    const cols = rows > 0 ? data[0].length : 0;

    svg.attr('width', cols * cellSize).attr('height', rows * cellSize);

    const flat = data.flat();
    const extent = d3.extent(flat) as [number, number];
    const safeExtent: [number, number] =
      extent[0] === extent[1] ? [extent[0] - 1, extent[1] + 1] : extent;
    const schemes: Record<string, (t: number) => string> = {
      Blues: d3.interpolateBlues,
    };
    const cellColor = d3
      .scaleSequential(schemes[viz.matrixScheme])
      .domain(safeExtent);

    const cells = svg
      .selectAll('g.row')
      .data(data)
      .enter()
      .append('g')
      .attr('class', 'row')
      .attr('transform', (_, i) => `translate(0, ${i * cellSize})`);

    cells
      .selectAll('rect')
      .data((row, i) => row.map((v) => ({ v, i })))
      .enter()
      .append('rect')
      .attr('class', (d) =>
        d.i === highlightRow ? 'cell highlighted' : 'cell'
      )
      .attr('x', (_, j) => j * cellSize)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', (d) => cellColor(d.v))
      .attr('stroke', (d) =>
        d.i === highlightRow ? viz.highlightStroke : viz.cellStroke
      )
      .attr('stroke-width', (d) => (d.i === highlightRow ? 2 : 1));

    cells
      .selectAll('text')
      .data((row) => row)
      .enter()
      .append('text')
      .attr('x', (_, j) => j * cellSize + cellSize / 2)
      .attr('y', cellSize / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 10)
      .attr('fill', viz.matrixText)
      .text((d) => Number(d).toFixed(PRECISION));
  }, [data, highlightRow, cellSize]);

  return <svg ref={ref} />;
}
