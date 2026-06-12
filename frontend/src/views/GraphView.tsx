import { motion } from "framer-motion";
import { useCallback, useMemo } from "react";
import ReactFlow, { Background, Controls, MiniMap, type Edge, type Node } from "reactflow";
import "reactflow/dist/style.css";
import { api, type GraphPayload } from "../api";
import { EntityNode, MemoryNode } from "../components/GraphNodes";
import { Empty, ErrorState, Loading } from "../components/States";
import { useFetch } from "../hooks";
import { useProject } from "../project";

const NODE_TYPES = { memory: MemoryNode, entity: EntityNode };

function toFlow(payload: GraphPayload): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = payload.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: n.position,
    data: n.data,
  }));
  const edges: Edge[] = payload.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    animated: e.animated,
    style:
      e.kind === "supersedes"
        ? { stroke: "#fb7185", strokeWidth: 1.5 }
        : { stroke: "#3f3f50", strokeDasharray: "4 3" },
    label: e.kind === "supersedes" ? "supersedes" : undefined,
    labelStyle: { fill: "#fb7185", fontSize: 10, fontFamily: "monospace" },
    labelBgStyle: { fill: "#101018" },
  }));
  return { nodes, edges };
}

export default function GraphView() {
  const { project } = useProject();
  const fetchGraph = useCallback(() => api.graph(project), [project]);
  const { data, loading, error } = useFetch(fetchGraph);

  const flow = useMemo(() => (data ? toFlow(data) : { nodes: [], edges: [] }), [data]);

  if (loading) return <Loading label="Loading graph…" />;
  if (error) return <ErrorState message={error} />;
  if (!flow.nodes.length) {
    return (
      <Empty
        title="No graph yet"
        hint="python scripts/seed_dashboard.py populates the demo project"
      />
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="panel h-[calc(100vh-8.5rem)] overflow-hidden"
    >
      <ReactFlow
        nodes={flow.nodes}
        edges={flow.edges}
        nodeTypes={NODE_TYPES}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        nodesDraggable
        nodesConnectable={false}
      >
        <Background color="#262635" gap={24} size={1} />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable nodeColor={() => "#262635"} maskColor="rgba(10,10,15,0.7)" />
      </ReactFlow>
    </motion.div>
  );
}
