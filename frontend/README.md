# TMSP Frontend

React-based frontend for the **Test for Medical Stepwise Predictions (TMSP)** application — a tool for visualizing, traversing, and benchmarking ICD-10-CM code predictions.

## Tech Stack

- **React 19** with TypeScript
- **Vite** for development and bundling
- **D3.js** for graph visualization
- **SSE (Server-Sent Events)** for real-time streaming

## Architecture Overview

```
src/
├── components/          # Reusable UI components
│   ├── GraphViewer.tsx      # D3-based graph visualization (shared across tabs)
│   ├── TrajectoryViewer.tsx # Decision history timeline
│   ├── VisualizeReportViewer.tsx
│   ├── BenchmarkReportViewer.tsx
│   └── shared/              # Modals, panels, common UI elements
├── features/            # Feature-specific logic
│   ├── visualize/           # Visualize tab state & sidebar
│   ├── traverse/            # Traverse tab state & sidebar
│   └── benchmark/           # Benchmark tab state & sidebar
├── lib/                 # Utilities and helpers
│   ├── api.ts               # Backend API calls
│   ├── sse.ts               # SSE streaming client
│   ├── types.ts             # TypeScript type definitions
│   ├── benchmark.ts         # Benchmark comparison logic
│   ├── graphPositioning.ts  # Layout algorithm
│   └── nodeStyles.ts        # Node appearance helpers
└── App.tsx              # Main app with tab routing
```

## Feature Tabs

### Visualize
Build and explore ICD-10-CM code hierarchies. Enter codes to see their ancestry tree with lateral relationships (codeFirst, codeAlso, useAdditionalCode, sevenChrDef).

### Traverse
Run LLM-powered traversals from clinical notes. Supports two modes:
- **Scaffolded**: Step-by-step tree traversal with batch decisions
- **Zero-shot**: Single-pass code prediction

Features real-time streaming visualization and spot rewind for debugging.

### Benchmark
Compare expected codes against traversal results. Metrics include:
- **Traversal Recall**: Coverage of expected trajectory
- **Final Codes Recall**: Exact match rate for endpoints
- **Outcome categories**: Matched, Undershot, Overshot, Missed

## Key Components

| Component | Purpose |
|-----------|---------|
| `GraphViewer` | D3-based SVG graph with zoom/pan, node overlays, benchmark status coloring |
| `TrajectoryViewer` | Timeline of batch decisions with expand/collapse |
| `BenchmarkReportViewer` | Three-column comparison (Missed / Correct / Extra) with metrics |
| `*Sidebar` | Tab-specific controls (code input, clinical note, LLM settings) |

## Graph Visualization

- **Node types**: Finalized, Ancestor, Placeholder, Activator (sevenChrDef)
- **Edge types**: Hierarchy (solid), Lateral (dashed orange)
- **Benchmark overlays**: Expected (dashed), Traversed (green), Matched (green fill)
- **Layout**: Bounded region allocation algorithm preventing collisions

## Development

```bash
npm install      # Install dependencies
npm run dev      # Start dev server (http://localhost:5173)
npm run build    # Production build
npm run lint     # Run ESLint
```

## Configuration

The frontend expects a backend server at the URL configured in `src/lib/constants.ts`. Default: `http://localhost:8000`.
