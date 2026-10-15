// n8n Code node: "Pick bases for variants"
// From the trending list, pick original models (not already quantized copies) whose
// smaller community copies (GGUF / MLX / FP8 ...) are worth looking up.
const LOOKUP_TASKS = new Set(['text-generation', 'image-text-to-text', 'any-to-any', 'audio-text-to-text',
  'video-text-to-text', 'automatic-speech-recognition', 'text-to-speech', 'text-to-image', 'image-to-image',
  'text-to-video', 'image-to-video', 'feature-extraction', 'sentence-similarity']);
const MAX = 100;
const picked = $input.all()
  .map(i => i.json)
  .filter(m => m && m.id && !m.error)
  .filter(m => LOOKUP_TASKS.has(m.pipeline_tag))
  .filter(m => !(m.baseModels && m.baseModels.relation === 'quantized'))
  .filter(m => !(m.tags || []).some(t => ['gguf', 'mlx'].includes(t)))
  .sort((a, b) => (b.trendingScore || 0) - (a.trendingScore || 0))
  .slice(0, MAX);
return picked.map(m => ({ json: { id: m.id } }));
