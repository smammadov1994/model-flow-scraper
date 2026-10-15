// Keep only the Hugging Face fields the join uses, so node outputs stay small enough
// for the n8n editor to show live.
const WEIGHT_RE = /\.(gguf|safetensors|onnx|onnx_data|pte|tflite|task|litertlm|mlmodel|bin|pt|pth|ckpt|nemo)$|\.mlpackage\/|\.mlmodelc\/|model_index\.json$/i;
function slimHf(m) {
  const card = m.cardData || {};
  const files = (m.siblings || []).map(s => s.rfilename);
  return {
    id: m.id, author: m.author, pipeline_tag: m.pipeline_tag, library_name: m.library_name, tags: m.tags || [],
    downloads: m.downloads, likes: m.likes, trendingScore: m.trendingScore,
    createdAt: m.createdAt, lastModified: m.lastModified, gated: m.gated, baseModels: m.baseModels,
    inferenceProviderMapping: (m.inferenceProviderMapping || []).map(p => ({ provider: p.provider })),
    safetensors: m.safetensors ? { parameters: m.safetensors.parameters, total: m.safetensors.total } : null,
    gguf: m.gguf ? { total: m.gguf.total, architecture: m.gguf.architecture, context_length: m.gguf.context_length } : null,
    cardData: {
      license: card.license, license_name: card.license_name, license_link: card.license_link,
      language: card.language, base_model: card.base_model, base_model_relation: card.base_model_relation,
      pipeline_tag: card.pipeline_tag, library_name: card.library_name, tags: card.tags,
    },
    siblings: files.filter(f => WEIGHT_RE.test(f)).slice(0, 200).map(rfilename => ({ rfilename })),
    file_count: files.length,
  };
}
