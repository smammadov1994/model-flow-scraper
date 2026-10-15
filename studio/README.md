# Model Studio (frontend)

Chat on the left, workflow canvas on the right. Describe what you want to build and the backend
returns a pipeline of AI models that fits your device. Contract: `../docs/STUDIO.md`.

Stack: Vite 5 + React 18 + TypeScript + `@xyflow/react` 12 (React Flow). Hand-written CSS, light and dark.

## Develop

```sh
cd studio
npm install
npm run dev          # http://localhost:5173/studio/  (proxies /api to http://localhost:8765)
```

Start the Model Finder web server (port 8765) for the real backend. No backend? Open
`http://localhost:5173/studio/?demo=1` — fakes chat replies: the first message (with "Run on: Ask me") gets the where-should-this-run buttons, then a plan from `src/demo/pipeline.json` (phone) or a local/cloud variant (`src/demo/plan.ts`). Add `&pipeline=1` to start with the phone pipeline already loaded.

## Build

```sh
npm run typecheck
npm run build        # writes ../src/model_finder/static/studio  (served at /studio/)
```

The build output is committed so the Python package serves it without Node.

## Notes

- `?s=<session_id>` in the URL restores a session (`GET /api/studio/sessions/{id}`).
- Dragging nodes saves positions (debounced 500 ms) via `POST /api/studio/move`.
- All API text is rendered through React (no `dangerouslySetInnerHTML`).
- The "Run on:" switch (Ask me / This computer / Phone / Cloud) is saved in localStorage (`studio.target`) and sent as `deployment` with each message.
