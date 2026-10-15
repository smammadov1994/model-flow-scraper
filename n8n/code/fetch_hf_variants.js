// n8n Code node: for each picked base model, fetch its most-downloaded quantized copies.
//__SLIM_HF__
const ids = $input.all().map(i => i.json.id).filter(Boolean);
const out = [];
const errors = [];
for (let i = 0; i < ids.length; i += 5) {
  const batch = ids.slice(i, i + 5);
  const results = await Promise.all(batch.map(id => this.helpers.httpRequest({
    url: `https://huggingface.co/api/models?filter=base_model:quantized:${id}&sort=downloads&direction=-1&limit=6&__HF_QS__`,
    json: true, timeout: 60000,
  }).catch(e => { errors.push(`${id}: ${e.message}`); return []; })));
  for (const list of results) for (const m of list) out.push({ json: slimHf(m) });
  await new Promise(r => setTimeout(r, 500));
}
if (errors.length) out.push({ json: { error: `variant lookups failed for ${errors.length} models`, details: errors.slice(0, 10) } });
return out;
