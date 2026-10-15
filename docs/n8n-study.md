# n8n study (for Model Studio)

Purpose: learn (a) what makes n8n's canvas feel good, so we can design our own, and (b) the exact workflow
file format, so "Export to n8n" produces files that import cleanly. This is a study for interoperability and
UX learning. **Nothing from n8n is copied into the project**: no source, CSS, icons or branding. This document
describes behaviour in our own words; the sample workflow is our own.

Target: `n8nio/n8n:2.6.4` (the container `model-finder-n8n`, read only). Date of study: 2026-10-15.

Files produced:
- `docs/n8n-study.md` (this file)
- `research/n8n-study/sample-workflow.json` (our validated sample, Manual Trigger -> Code -> HTTP Request + Sticky Note)
- `research/n8n-study/rea-summary.json` (numbers and chunk list from the REA run)
- `research/n8n-study/rea-canvas-symbols.txt` (symbol names REA extracted from the canvas chunk, no code)

## 1. What REA showed

### How it was run

1. Copied the built editor out of the container, read only, into a scratch directory outside the repo:
   `docker cp model-finder-n8n:/usr/local/lib/node_modules/n8n/node_modules/n8n-editor-ui/dist <scratch>/n8n-editor/dist`
   (42 MB, 428 files in `assets/`, one `index.html`, no source maps). Nothing in it was executed.
2. Static analysis with REA 6.5.0 (`rea-agents` on npm; the CLI is `rea`):
   ```
   npx -y rea-agents@latest analyze-javascript-application <dist> --format json --full-output \
       --analysis-timeout-ms 600000 --max-heap-mb 4096 > rea-n8n.json
   ```
   Took 118 s. The raw JSON is 224 MB, so it stays in the scratch directory; only a summary is in the repo.
3. A focused run on one chunk (the canvas bundle):
   ```
   npx -y rea-agents@latest analyze-javascript-application <dist> \
       --source-path assets/useCanvasMapping-BDRTplLi.js --format json --full-output --max-heap-mb 4096
   ```

### What worked

- Whole-app run: 270 files selected, 269 JavaScript files parsed, 0 parse failures, about 34 MB of text, 4.75 million
  AST nodes. Output: an application graph (1,043 nodes, 4,017 edges: 2,823 `imports`, 754 `contains`, 344 `calls`)
  and a semantic graph (about 80,000 nodes, 126,000 relations).
- It confirms the editor is a plain browser SPA: no Electron windows, no preload, no IPC. Entry is `index.html` ->
  one `index-*.js` script. It uses `sessionStorage` (get and set) and has 160 "network endpoint candidates".
- The chunk list (file names are intact even though the code is bundled) maps the product's modules clearly. The
  relevant ones: `useCanvasMapping`, `useCanvasOperations`, `canvas.utils`, `CanvasRunWorkflowButton`, `NodeView`,
  `NodeCreator`, `nodeCreator.store`, `NodeCreation`, `NodeDetailsView`, `NodeSettings`, `ParameterInputList`,
  `NodeIcon`, `nodeTransforms`, `LogsPanel`, `RunData`, `useRunWorkflow`, `TriggerPanel`, `WorkflowPreview`.
- Using the REA symbol and import data on `useCanvasMapping`, we learned that this one chunk contains the whole
  canvas: the Vue Flow core, d3-zoom/d3-selection (pan and zoom), the dagre layout library (auto "tidy up"), and
  about 30 n8n `Canvas*` components. It imports `canvas.utils`, `useCanvasOperations`, `useRunWorkflow`,
  `nodeCreator.store`, `NodeSettings`, `ParameterInputList`, `NodeIcon`, `KeyboardShortcutTooltip`, `useKeybindings`,
  `usePinnedData`. See `rea-canvas-symbols.txt`.
- Component names inside it (this is the canvas anatomy): `Canvas`, `CanvasNode`, `CanvasNodeRenderer`,
  `CanvasNodeDefault`, `CanvasNodeTrigger`, `CanvasNodeStickyNote`, `CanvasNodeAddNodes`, `CanvasNodeToolbar`,
  `CanvasNodeStatusIcons`, `CanvasNodeSettingsIcons`, `CanvasNodeStickyColorSelector`, `CanvasNodeTooltip`,
  `CanvasNodeDisabledStrikeThrough`, `CanvasHandleRenderer`, `CanvasHandleMainInput`, `CanvasHandleMainOutput`,
  `CanvasHandleNonMainInput`, `CanvasHandleNonMainOutput`, `CanvasHandleDot`, `CanvasHandleDiamond`,
  `CanvasHandlePlus`, `CanvasEdge`, `CanvasEdgeToolbar`, `CanvasEdgeTooltip`, `CanvasConnectionLine`,
  `CanvasArrowHeadMarker`, `CanvasBackground`, `CanvasBackgroundStripedPattern`, `CanvasControlButtons`,
  `ContextMenu`.

### What did not work, or only partly

- REA reports coverage `partial` for the whole-app run: 435 limitations, 427 graph nodes of kind `unknown` ("resource
  constraint ... expanded application relationships"), and the semantic graph is `truncated` at its result budget.
  So the import graph for most big chunks (e.g. `NodeView`, `NodeCreator`) is empty and the semantic graph only
  covers a subset of files. Raising `--max-semantic-nodes` above 1,000,000 is rejected (validation error).
- Passing `--source-path` several times did not narrow the run (it still selected all 270 files); a single
  `--source-path` works and is fast (a few seconds, 1 file parsed, 47 MB JSON).
- REA analyses structure (modules, functions, imports, call graph, literals). It is not a UI renderer; the numeric
  UX facts below came from reading the (readable, not minified) chunk text locally, guided by the REA module map.
  Nothing was run in a browser: the instance needs an owner account, and we do not create accounts.

## 2. Canvas UX patterns worth emulating (our own design)

All values below are observations of behaviour. We design and style our own.

**Graph engine.** Vue Flow (a node/edge library for Vue, same idea as React Flow / `@xyflow/react` that we use). The
whole canvas is a single custom node type and a single custom edge type; the node's "render type" decides how it
looks (normal node, sticky note, "add first step" placeholder).

**Grid and viewport.**
- Dotted background, 16 px grid, snap-to-grid on, nodes snap to multiples of 16.
- Zoom range 0 (no lower limit) to 4. Scroll wheel pans (not zooms); Ctrl/Cmd+scroll (pinch) zooms; hold Space
  or Ctrl, or use middle mouse, to pan by dragging; plain left-drag on empty space draws a selection box.
- Connection snap radius of 60 px when dragging a new link.
- Bottom-left control cluster: zoom to fit, zoom in, zoom out, (reset zoom when not 100%), tidy up (auto layout).
  Keyboard: `1` fit, `+`/`-` zoom. Tooltips show the shortcut.
- Tidy-up runs a dagre layered layout (REA shows `createDagreGraph` and a separate vertical variant used for
  AI sub-nodes; direction not confirmed by us). Zoom-to-fit is a button and also runs after loading a workflow.
- A mini-map exists as a component (styles `minimap`, `minimap-mask`); it is off the default screen.

**Node card anatomy (default node).**
- A 96 x 96 px rounded tile (squarish). The node's icon sits in the centre at 40 px. The name and a smaller
  subtitle (e.g. the HTTP method and URL) are drawn **below** the tile, not inside it. That keeps tiles compact
  and lets long names wrap.
- Triggers are the same tile with a lightning bolt on the left (`CanvasNodeTrigger`); they have no input handle.
- "Configuration" (AI sub-nodes) nodes are smaller circles (80 px); "configurable" nodes are wide (256 x 96) with
  extra diamond-shaped sockets along the bottom for sub-nodes (model, memory, tools). We do not need this in v1.
- Height grows by 32 px per extra main handle beyond two so handles never overlap.
- Selected: highlighted border. Disabled: dimmed with a strike-through line between input and output.
- A floating toolbar appears above the node on hover/selection: run this node, enable/disable, delete, "more"
  (context menu). Sticky notes get a colour button instead of run.

**Handles.**
- Input handle on the left (a small dot or ring), output handle on the right (a dot). Outputs with several
  branches (like an IF node) stack vertically, each labelled (`true`, `false`).
- An unconnected output shows a short line ending in a **plus button**; clicking it opens the node creator
  already linked to that handle. This "add the next step from here" affordance is the core of n8n's flow.
  After a run, a connected output shows the item count ("12 items") above the line.
- Dragging from a handle shows a ghost line with an arrow head; valid targets highlight; releasing on empty space
  opens the node creator (insert-and-connect).
- An output may link to many inputs; an input of type "main" accepts many links; sub-node sockets have
  max-connection limits that make over-limit handles non-connectable.

**Edges.**
- Normal "main" edges are smooth bezier curves; when the target is to the **left** of the source (a loop back),
  the edge becomes a two-segment rounded elbow that dips 130 px below, with 16 px corners. Non-main (AI) edges
  are dashed (5,6) and bezier.
- Closed arrow head at the end. Edge colour is neutral grey; turns green after a successful run and purple/accent
  when the data is pinned. Colour lightness adjusts with zoom so lines stay visible when zoomed out.
- Hover an edge: a small toolbar appears at its midpoint with **add node** (insert in between) and **delete**.
  The toolbar stays for a short delay after the mouse leaves so it is easy to hit.

**Add-node flow (node creator).** A right-side panel (slides in) with a search bar at the top and a list of
categories; views are: Triggers ("What triggers this workflow?" with the "Manual" option), Regular nodes
(Transform data, Flow, Helpers, Core, AI, then app integrations), and AI sub-views. Choosing a node that has
several actions shows a second list of "Actions". Items show an icon, name, one-line description; a "no results"
state offers suggestions. For an empty workflow the canvas shows a large dashed "Add first step" tile.

**Node details view (NDV, the side panel).** Double-click a node. It opens as a large three-column overlay:
**Input data | node parameters | Output data**. The middle panel has tabs **Parameters / Settings** (plus Docs
link; Credential and Action tabs for app nodes), a title that can be renamed in place, and the parameters form
(auto-generated from the node's schema: text, number, toggle, dropdown, JSON/code editor, key-value list, with
`{{ expression }}` support). The outer panels are draggable to resize and show the previous node's output and
this node's output as Table / JSON / Schema. The main panel is 390 px wide by default.

**Run and status badges.**
- Big primary "Execute workflow" button at the bottom centre (a split button when several triggers exist).
- While running: a node pulses/spins; edges animate. After: a green check badge on the node tile, a green edge,
  the item count on the output. Errors: a red warning badge with a tooltip listing the messages (grouped,
  "(x3)"). Warning for stale/"dirty" results. A "Logs" panel docks at the bottom with per-node run results.
- Small status icons at the corner of the tile for: pinned data, retry on fail, execute once, continue on error,
  always output data, disabled.

**Sticky notes.** A resizable rectangle on the canvas (default 240 x 160) with Markdown content (double click to
edit), behind the nodes, and 7 colour presets chosen from a popover. A note can be dropped over nodes to group
them visually. It never connects to anything and is skipped on execution.

**What to emulate in Model Studio (suggested).**
1. Squarish tile (about 96 px) with icon and the title below; subtitle = chosen model + size. Our node types
   (input, model, logic, output) map to colour/icon, not different shapes.
2. Left input / right output dot handles, plus-button on the open output that opens "add step" ("swap model" /
   "add step after this").
3. Bezier edges with an arrow head and labels at the midpoint (our edges have `label`, e.g. `audio`); hover
   toolbar with delete/insert.
4. 16 px snapping grid, dotted background, fit-to-view button, zoom controls bottom left, tidy-up (dagre or
   a simple layered layout).
5. Right-hand details panel (not modal): model card, "why this fit", alternatives with a swap action, memory
   against the device budget.
6. Status badges: `ok` green check, `over_budget` amber, `no_match` red, `pending` spinner.
7. Sticky notes for the planner's explanations ("Why this pipeline").
8. Minimap optional; useful only past roughly 15 nodes.

## 3. The workflow file format (n8n 2.6.4)

Sources: `n8n-workflow` type definitions shipped in the image (`dist/esm/interfaces.d.ts`: `INode`, `IConnections`,
`IWorkflowBase`, `IWorkflowSettings`), the node implementations in `n8n-nodes-base`, docs.n8n.io
(workflows > export and import), and live import tests (section 4). The public source is in GitHub
`n8n-io/n8n`, `packages/workflow/src/interfaces.ts`.

### Top level

```jsonc
{
  "name": "My workflow",            // string, required by the editor; shown as the title
  "nodes": [ /* INode[] */ ],       // required
  "connections": { /* IConnections */ },   // required, {} if none
  "settings": { "executionOrder": "v1" },  // optional but always include executionOrder v1
  "pinData": {},                     // optional: { "<node name>": [ { "json": {...} } ] }
  "active": false,                   // optional; imports are never auto-activated
  "meta": { "instanceId": "..." },  // optional, free-form (templateId, instanceId, onboardingId)
  "id": "...",                       // optional; ignored/regenerated on import
  "tags": [], "versionId": "...", "staticData": null   // optional, safe to omit
}
```

`IWorkflowSettings` known keys: `executionOrder` (`"v0"|"v1"`, use `"v1"`), `timezone`, `errorWorkflow`,
`callerPolicy`, `callerIds`, `saveDataErrorExecution`, `saveDataSuccessExecution`, `saveManualExecutions`,
`saveExecutionProgress`, `executionTimeout`, `binaryMode`, `timeSavedPerExecution`, `availableInMCP`.

### Node (`INode`)

| field | type | notes |
|---|---|---|
| `id` | string | UUID. Unique per workflow. The editor generates one if missing, but always supply one |
| `name` | string | **Unique within the workflow**; it is the key used by `connections` |
| `type` | string | `<package>.<nodeName>`, e.g. `n8n-nodes-base.manualTrigger` |
| `typeVersion` | number | must be a version the node supports (table below). Decimals allowed (4.4) |
| `position` | `[x, y]` | top-left of the tile in canvas pixels; use multiples of 16 |
| `parameters` | object | node-specific; `{}` is valid (defaults apply) |
| optional | | `disabled`, `notes`, `notesInFlow`, `retryOnFail`, `maxTries`, `waitBetweenTries`, `alwaysOutputData`, `executeOnce`, `onError`, `continueOnFail`, `credentials`, `webhookId` |

Only parameters that differ from the defaults need to be present.

### Connections (`IConnections`)

Keyed by **source node name** (not id) -> connection type (`main`) -> array indexed by **output index** -> array of
targets, each `{ "node": "<target node NAME>", "type": "main", "index": <target input index> }`.

```json
{
  "A": { "main": [ [ { "node": "B", "type": "main", "index": 0 } ] ] },
  "IF": { "main": [ [ { "node": "ThenNode", "type": "main", "index": 0 } ],
                    [ { "node": "ElseNode", "type": "main", "index": 0 } ] ] }
}
```

Rules: output 0 is the first inner array; an output with no links is `[]` (or `null`); a source with several
targets on the same output lists them in the same inner array (fan-out). Names, not ids. Sticky notes and nodes
without links simply do not appear. Non-main types (`ai_languageModel`, etc.) exist but we do not use them.

### Node types and `typeVersion` valid in 2.6.4 (verified in the image)

| node | `type` | `typeVersion` |
|---|---|---|
| Manual Trigger | `n8n-nodes-base.manualTrigger` | `1` |
| Code | `n8n-nodes-base.code` | `1` or `2` (use `2`) |
| HTTP Request | `n8n-nodes-base.httpRequest` | `1, 2, 3, 4, 4.1, 4.2, 4.3, 4.4` (default `4.4`) |
| Sticky Note | `n8n-nodes-base.stickyNote` | `1` |
| (also handy) Set / Edit Fields | `n8n-nodes-base.set` | `3.4` |
| (also handy) Webhook | `n8n-nodes-base.webhook` | `2` (needs a `webhookId`) |
| (also handy) No Operation | `n8n-nodes-base.noOp` | `1` |

The last three rows are common knowledge from n8n's docs and were not re-verified in this study; verify before
using them.

### Parameter shapes

**Manual Trigger** - no parameters.
```json
{ "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1, "parameters": {} }
```
Default display name is "When clicking 'Execute workflow'"; any name works. Only one allowed per workflow.

**Code (JavaScript)**
```json
{ "type": "n8n-nodes-base.code", "typeVersion": 2,
  "parameters": { "mode": "runOnceForAllItems", "language": "javaScript", "jsCode": "return [{ json: { a: 1 } }];" } }
```
`mode`: `runOnceForAllItems` (default) or `runOnceForEachItem`. `language`: `javaScript` (default; `python` also
exists in v2). The code must return an array of `{ json: {...} }` objects (all-items mode). In v1 the language
field is absent and the code lives in `jsCode` as well.

**HTTP Request (v4.4)**
```json
{ "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4,
  "parameters": { "method": "POST", "url": "http://localhost:8765/api/example",
                  "sendBody": true, "contentType": "json", "specifyBody": "json",
                  "jsonBody": "={{ JSON.stringify($json) }}", "options": {} } }
```
`method` defaults to `GET`. For a JSON body: `sendBody: true`, `contentType: "json"`, `specifyBody: "json"` (the
alternative `keypair` uses `bodyParameters`), body in `jsonBody`. A string starting with `=` is an expression;
`{{ ... }}` is evaluated inside it. Other keys: `sendQuery` + `queryParameters`, `sendHeaders` + `headerParameters`
(`specifyHeaders`: `keypair|json`, `jsonHeaders`), `authentication` (`none` default,
`genericCredentialType`, `predefinedCredentialType`), `options` (timeout, redirect, response format...).

**Sticky Note**
```json
{ "type": "n8n-nodes-base.stickyNote", "typeVersion": 1,
  "parameters": { "content": "## Title\nMarkdown text", "width": 640, "height": 140, "color": 4 } }
```
`color` is a **number 1 to 7** (not a string). Defaults: width 240, height 160, color 1. `position` is the
note's top-left. Place a note behind nodes by giving it a lower y and a size that covers them.

### Layout hints for generated files

- Default tile is 96 x 96; a comfortable horizontal gap is 160 to 260 px between node positions (the sample
  uses 260). Positions of consecutive nodes in the same row share the same y.
- Position is the top-left of the tile; our Pipeline positions can be passed through, rounded to multiples of 16.
- Our node `type`s could map as: input -> Manual Trigger (first) / Webhook; model -> HTTP Request pointing at
  an inference endpoint, or a Code placeholder holding the chosen model id in a comment; logic -> Code or Set;
  output -> Set or Respond to Webhook. Put the model id, size, RAM and the "why" in the node's `notes` field
  (`"notesInFlow": true` shows it under the node) and in a Sticky Note.

## 4. Validated sample

`research/n8n-study/sample-workflow.json` (also reproduced here). Validated against a **throwaway** container
(`n8nio/n8n:2.6.4`, its own empty user folder in the scratch directory; the real `model-finder-n8n` container and
volume were not touched):

```
docker run --rm --entrypoint n8n -v <scratch>/n8n-validate:/data -e N8N_USER_FOLDER=/data \
    n8nio/n8n:2.6.4 import:workflow --input=/data/sample.json
```

Results:
- `import:workflow`: **"Successfully imported 1 workflow."**
- `export:workflow --all` from the same folder: the workflow is stored with our nodes, connections,
  `settings.executionOrder = v1`, `meta.instanceId`, and `active = false`; n8n adds `id`, `versionId`, `createdAt`,
  etc.
- `n8n execute --id=<id>` (headless run): Manual Trigger and Code ran, the HTTP Request node was reached and failed
  with "The service refused the connection" because nothing listens on `localhost:8765/api/example`. That is the
  expected result and shows the parameters were read correctly (method, url, body).

```json
{
  "name": "Model Studio sample: voice assistant (design)",
  "nodes": [
    { "id": "3f1c7a52-9b0e-4d6a-8e11-0a1b2c3d4e01", "name": "Manual Trigger",
      "type": "n8n-nodes-base.manualTrigger", "typeVersion": 1, "position": [0, 0], "parameters": {} },
    { "id": "3f1c7a52-9b0e-4d6a-8e11-0a1b2c3d4e02", "name": "Prepare request",
      "type": "n8n-nodes-base.code", "typeVersion": 2, "position": [260, 0],
      "parameters": { "mode": "runOnceForAllItems", "language": "javaScript",
        "jsCode": "return [{ json: { model: 'openai/whisper-large-v3-turbo', task: 'automatic-speech-recognition' } }];" } },
    { "id": "3f1c7a52-9b0e-4d6a-8e11-0a1b2c3d4e03", "name": "Call model server",
      "type": "n8n-nodes-base.httpRequest", "typeVersion": 4.4, "position": [520, 0],
      "parameters": { "method": "POST", "url": "http://localhost:8765/api/example", "sendBody": true,
        "contentType": "json", "specifyBody": "json", "jsonBody": "={{ JSON.stringify($json) }}", "options": {} } },
    { "id": "3f1c7a52-9b0e-4d6a-8e11-0a1b2c3d4e04", "name": "Sticky Note",
      "type": "n8n-nodes-base.stickyNote", "typeVersion": 1, "position": [-40, -180],
      "parameters": { "content": "## Exported from Model Studio\nThree steps of the design, wired in order.",
        "width": 640, "height": 140, "color": 4 } }
  ],
  "connections": {
    "Manual Trigger": { "main": [[{ "node": "Prepare request", "type": "main", "index": 0 }]] },
    "Prepare request": { "main": [[{ "node": "Call model server", "type": "main", "index": 0 }]] }
  },
  "settings": { "executionOrder": "v1" },
  "pinData": {},
  "active": false,
  "meta": { "instanceId": "model-studio-export" }
}
```

## 5. Gotchas for generating importable files

1. **The CLI importer is permissive; it is not a validator.** Probes that all still printed "Successfully imported":
   nodes without `id`; a connection pointing at a node name that does not exist; an unknown node `type`; a file
   with no `settings`, `pinData`, `active` or `meta`; a top-level array `[ {workflow} ]`. So "imports cleanly"
   proves little. Validate our own output before writing it: unique `name`s, every connection endpoint exists,
   every `type`/`typeVersion` pair is in the table above, and no duplicate ids. In the editor an unknown type
   shows as a broken "?" node; a dangling connection is dropped or errors on load.
2. **Connections are keyed by node name, not id.** Renaming a node means rewriting `connections`. Names must be
   unique; when exporting our pipeline titles, de-duplicate ("Speech to text", "Speech to text 2").
3. **Names with special characters are fine** as JSON keys, but keep them readable; the editor shows them.
4. `typeVersion` must be one the node supports (Code: 1 or 2; HTTP Request: up to 4.4). Using a version that
   does not exist makes the node unusable in the editor.
5. **Sticky Note `color` is a number (1-7), `width`/`height` numbers**; `content` is Markdown. Do not give it
   `connections`.
6. **Do not include credentials.** Exported files carry credential names and ids (docs warn about this). Our
   export has none; HTTP nodes should use no authentication or a placeholder the user fills in.
7. Include `settings.executionOrder: "v1"` (the current execution model); `active: false`; imports are inactive
   and need the user to activate triggers explicitly. A Manual Trigger needs no activation.
8. **Expressions**: a parameter string starting with `=` is an expression (`"={{ $json.x }}"`). A literal URL or
   text must not start with `=`. In the Code node the JavaScript is a plain string: escape quotes and newlines
   for JSON.
9. `position` is `[x, y]` numbers (not an object). Our Pipeline uses `{x, y}`; convert.
10. On the editor's "Import from file" / paste, n8n accepts either a full workflow or just `{nodes, connections}`
    (that is what copy-paste uses). Download from our app as `<title>.json` with
    `Content-Disposition: attachment`, content type `application/json`.
11. Encoding: UTF-8 JSON, no comments, no trailing commas (our docs show `jsonc` only for readability).
12. Bigger exports: keep one trigger, a linear or branching DAG, no cycles unless intended; n8n does allow loops.

## 6. Provenance and licensing notes

- REA: `morluto/rea` (MIT), npm `rea-agents@latest` resolved to 6.5.0. Used only for static analysis of a local copy.
- n8n is under the Sustainable Use License (source-available). We studied the shipped editor bundle and type
  definitions for interoperability and UX learning only. We reproduce no n8n code, styles, icons, or logos; the
  JSON format is a data format and our sample is our own.
- The scratch copy of the editor and the raw REA outputs live outside the repo and can be deleted:
  `/private/tmp/claude-501/-Users-Apple-Desktop/21c1473e-d589-45ba-8f10-0acba9a93279/scratchpad/`
  (`n8n-editor/`, `rea-n8n.json` 224 MB, `n8n-validate/`).
