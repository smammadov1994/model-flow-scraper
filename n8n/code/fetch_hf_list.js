// n8n Code node: fetch one Hugging Face model list and slim it.
// __URL__ is filled in by build_workflow.py.
//__SLIM_HF__
try {
  const res = await this.helpers.httpRequest({ url: '__URL__', json: true, timeout: 120000 });
  return res.map(m => ({ json: slimHf(m) }));
} catch (e) {
  return [{ json: { error: `HF list failed: ${e.message}` } }];
}
