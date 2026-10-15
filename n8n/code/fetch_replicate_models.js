// n8n Code node: page through Replicate's public models (newest first) and slim each one.
const token = $env.REPLICATE_API_TOKEN;
if (!token) return [{ json: { error: 'REPLICATE_API_TOKEN is not set' } }];
const headers = { Authorization: `Bearer ${token}` };
const MAX_PAGES = 40;
function slimReplicate(r) {
  const lv = r.latest_version || null;
  const input = lv && lv.openapi_schema && lv.openapi_schema.components && lv.openapi_schema.components.schemas
    && lv.openapi_schema.components.schemas.Input;
  return {
    owner: r.owner, name: r.name, url: r.url, description: r.description, visibility: r.visibility,
    run_count: r.run_count, github_url: r.github_url, paper_url: r.paper_url, license_url: r.license_url,
    weights_url: r.weights_url, cover_image_url: r.cover_image_url, is_official: r.is_official,
    latest_version: lv ? { id: lv.id, created_at: lv.created_at,
      input_names: input && input.properties ? Object.keys(input.properties).slice(0, 25) : [] } : null,
  };
}
const out = [];
let url = 'https://api.replicate.com/v1/models?sort_by=model_created_at&sort_direction=desc';
try {
  for (let page = 0; url && page < MAX_PAGES; page++) {
    let body;
    for (let attempt = 1; ; attempt++) {  // Replicate pages are sometimes slow; retry before giving up
      try { body = await this.helpers.httpRequest({ url, headers, json: true, timeout: 120000 }); break; }
      catch (e) { if (attempt >= 3) throw e; await new Promise(r => setTimeout(r, 3000 * attempt)); }
    }
    for (const r of body.results || []) out.push({ json: slimReplicate(r) });
    url = body.next;
    await new Promise(r => setTimeout(r, 250));
  }
} catch (e) {
  out.push({ json: { error: `Replicate models stopped after ${out.length} models: ${e.message}` } });
}
return out;
