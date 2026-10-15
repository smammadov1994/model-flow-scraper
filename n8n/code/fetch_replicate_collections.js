// n8n Code node: fetch every Replicate collection and the models in it (slimmed).
const token = $env.REPLICATE_API_TOKEN;
if (!token) return [{ json: { error: 'REPLICATE_API_TOKEN is not set' } }];
const headers = { Authorization: `Bearer ${token}` };
const out = [];
try {
  const slugs = [];
  let url = 'https://api.replicate.com/v1/collections';
  while (url) {
    const body = await this.helpers.httpRequest({ url, headers, json: true, timeout: 60000 });
    for (const c of body.results || []) if (c.slug) slugs.push(c.slug);
    url = body.next;
  }
  for (const slug of slugs) {
    const c = await this.helpers.httpRequest({ url: `https://api.replicate.com/v1/collections/${slug}`, headers, json: true, timeout: 60000 });
    out.push({ json: { slug: c.slug, name: c.name, models: (c.models || []).map(r => ({
      owner: r.owner, name: r.name, url: r.url, description: r.description, run_count: r.run_count,
      github_url: r.github_url, paper_url: r.paper_url, license_url: r.license_url, cover_image_url: r.cover_image_url,
      latest_version: r.latest_version ? { id: r.latest_version.id, created_at: r.latest_version.created_at } : null,
    })) } });
    await new Promise(r => setTimeout(r, 150));
  }
} catch (e) {
  out.push({ json: { error: `Replicate collections failed: ${e.message}` } });
}
return out;
