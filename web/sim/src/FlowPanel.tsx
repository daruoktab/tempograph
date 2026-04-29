import { useCallback, useEffect } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PillNode } from "./PillNode";

const nodeTypes = { pill: PillNode };

function FlowInner({
  initialNodes,
  initialEdges,
}: {
  initialNodes: Node[];
  initialEdges: Edge[];
}) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const { fitView } = useReactFlow();

  // Sinkronisasi node & edge dari props ke state internal React Flow
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
    
    // Beri sedikit waktu agar node ter-render sebelum melakukan fitView
    const timeout = setTimeout(() => {
      fitView({ padding: 0.25, duration: 800 });
    }, 50);
    
    return () => clearTimeout(timeout);
  }, [initialNodes, initialEdges, setNodes, setEdges, fitView]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      minZoom={0.1}
      maxZoom={1.5}
      proOptions={{ hideAttribution: false }}
    >
      <Background variant={BackgroundVariant.Dots} color="var(--border-dark)" gap={24} size={2} />
      <Controls showInteractive={false} position="bottom-left" />
      <MiniMap
        style={{ 
          background: "var(--bg-panel)", 
          border: "1px solid var(--border-base)", 
          borderRadius: "var(--radius-md)", 
          overflow: "hidden", 
          boxShadow: "var(--shadow-sm)",
          width: 120, // Diperkecil agar tidak menutupi
          height: 80,
          opacity: 0.85,
        }}
        nodeColor={() => "var(--border-dark)"}
        maskColor="rgba(248, 250, 252, 0.6)"
        position="bottom-right"
        zoomable
        pannable
      />
    </ReactFlow>
  );
}

type Props = {
  nodes: Node[];
  edges: Edge[];
  title: string;
};

export function FlowPanel({ nodes, edges, title }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%", position: "relative" }}>
      <div 
        style={{ 
          position: "absolute", 
          top: 16, 
          left: 16, 
          zIndex: 4, 
          background: "rgba(255,255,255,0.85)", 
          backdropFilter: "blur(8px)", 
          padding: "6px 14px", 
          borderRadius: "var(--radius-full)", 
          fontSize: 12, 
          fontWeight: 600, 
          color: "var(--text-muted)", 
          border: "1px solid var(--border-base)", 
          boxShadow: "var(--shadow-sm)" 
        }}
      >
        {title}
      </div>
      {/* Container ini WAJIB memiliki height agar ReactFlow bisa merender graf */}
      <div style={{ flex: 1, width: "100%", height: "100%", position: "relative" }}>
        <ReactFlowProvider>
          <FlowInner initialNodes={nodes} initialEdges={edges} />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
