/**
 * A small topology of the pipeline, drawn once near the hero.
 *
 * Pure SVG and CSS: no canvas, no animation library, nothing per-frame.
 * The pulse travels the path slowly and stops entirely under
 * prefers-reduced-motion, where it becomes a plain static diagram.
 */
export function HeroDiagram() {
  const nodes = [
    { x: 18, label: "Question" },
    { x: 96, label: "Plan" },
    { x: 174, label: "Search" },
    { x: 252, label: "Evidence" },
    { x: 330, label: "Verify" },
    { x: 408, label: "Report" },
  ];

  return (
    <svg
      className="topology"
      viewBox="0 0 426 74"
      role="img"
      aria-label="Pipeline: question, plan, search, evidence, verify, report"
    >
      <line className="topology__spine" x1="18" y1="30" x2="408" y2="30" />
      <line className="topology__pulse" x1="18" y1="30" x2="408" y2="30" />
      {nodes.map((node, i) => (
        <g key={node.label}>
          <circle
            className="topology__node"
            cx={node.x}
            cy={30}
            r={i === 0 || i === nodes.length - 1 ? 6 : 4.5}
            style={{ animationDelay: `${i * 0.45}s` }}
          />
          <text className="topology__label" x={node.x} y={58} textAnchor="middle">
            {node.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
