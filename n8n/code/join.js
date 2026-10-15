// n8n Code node: "Clean & join"
// Pulls the raw output of every source node, cleans each record, joins records that
// describe the same model (Hugging Face repo <-> Replicate model <-> X posts), and
// emits ONE item per model in the schema documented in ../SCHEMA.md.
// Size/RAM/device-tier rules mirror src/model_finder/enrich.py so both pipelines agree.

const FETCHED_AT = new Date().toISOString();
const SCHEMA_VERSION = '1.0';

// ---------------------------------------------------------------- read source nodes
const sourceErrors = [];
function nodeItems(name) {
  let items;
  try { items = $(name).all().map(i => i.json); } catch (e) { return []; }
  for (const j of items) if (j && j.error) sourceErrors.push(`${name}: ${j.error}`);
  return items.filter(j => j && !j.error);
}

const hfNewestRaw = nodeItems('HF newest models');
const hfTrendingRaw = nodeItems('HF trending models');
const hfVariantsRaw = nodeItems('HF quantized variants');
const repPages = nodeItems('Replicate models');
const repCollections = nodeItems('Replicate collections');
const xText = nodeItems('Parse X file').map(j => j.data || '').join('\n');

// ---------------------------------------------------------------- reference tables
const BPW = {
  IQ1_S: 1.56, IQ1_M: 1.75, IQ2_XXS: 2.06, IQ2_XS: 2.31, IQ2_S: 2.5, IQ2_M: 2.7, Q2_K: 2.63,
  IQ3_XXS: 3.06, IQ3_XS: 3.3, IQ3_S: 3.44, IQ3_M: 3.66, Q3_K_S: 3.5, Q3_K_M: 3.91, Q3_K_L: 4.27, Q3_K_XL: 4.3,
  IQ4_XS: 4.25, IQ4_NL: 4.5, Q4_0: 4.55, Q4_1: 5.0, Q4_K_S: 4.58, Q4_K_M: 4.85, Q4_K_XL: 5.0, MXFP4: 4.25,
  Q5_0: 5.54, Q5_1: 6.0, Q5_K_S: 5.54, Q5_K_M: 5.69, Q5_K_XL: 5.8, Q6_K: 6.59, Q6_K_XL: 6.8, Q8_0: 8.5, Q8_K_XL: 9.0,
  TQ1_0: 1.69, TQ2_0: 2.06, F16: 16, BF16: 16, F32: 32,
  int4: 4.5, int8: 8.5, fp8: 8.5, fp16: 16, bf16: 16, fp32: 32,
};
const TIER_BUDGET_GB = { phone: 4.0, laptop: 12.0, gpu: 24.0 };
const OVERHEAD = 1.2;
const GEN_EXTRA_GB = 1.0;
const PHONE_FORMATS = new Set(['gguf', 'onnx', 'coreml', 'executorch', 'tflite', 'mlx', 'mlc']);

// task -> [group, inputs, outputs, plain-English label, search keywords]
const TASKS = {
  'text-generation': ['text', ['text'], ['text'], 'chat / text generation', ['llm', 'chatbot', 'assistant', 'writing']],
  'text2text-generation': ['text', ['text'], ['text'], 'text-to-text', ['rewrite', 'summarize']],
  'summarization': ['text', ['text'], ['text'], 'summarization', ['summarize']],
  'translation': ['text', ['text'], ['text'], 'translation', ['translate', 'multilingual']],
  'question-answering': ['text', ['text'], ['text'], 'question answering', ['qa']],
  'fill-mask': ['text', ['text'], ['text'], 'fill-mask (encoder)', ['bert', 'encoder']],
  'text-classification': ['text', ['text'], ['label'], 'text classification', ['sentiment', 'classifier', 'moderation']],
  'token-classification': ['text', ['text'], ['label'], 'entity tagging', ['ner', 'pii']],
  'zero-shot-classification': ['text', ['text'], ['label'], 'zero-shot classification', ['classifier']],
  'table-question-answering': ['text', ['table', 'text'], ['text'], 'table QA', ['spreadsheet']],
  'feature-extraction': ['embedding', ['text'], ['embedding'], 'embeddings', ['embedding', 'vector search', 'rag', 'semantic search']],
  'sentence-similarity': ['embedding', ['text'], ['embedding'], 'sentence embeddings', ['embedding', 'similarity', 'rag']],
  'text-ranking': ['embedding', ['text'], ['score'], 'reranker', ['rerank', 'rag', 'search']],
  'image-classification': ['vision', ['image'], ['label'], 'image classification', ['classify images']],
  'object-detection': ['vision', ['image'], ['boxes'], 'object detection', ['detect objects', 'yolo']],
  'zero-shot-object-detection': ['vision', ['image', 'text'], ['boxes'], 'open-vocabulary detection', ['detect objects']],
  'image-segmentation': ['vision', ['image'], ['mask'], 'image segmentation', ['segment', 'cutout']],
  'mask-generation': ['vision', ['image'], ['mask'], 'mask generation', ['segment anything']],
  'depth-estimation': ['vision', ['image'], ['depth'], 'depth estimation', ['3d', 'depth map']],
  'keypoint-detection': ['vision', ['image'], ['keypoints'], 'pose / keypoints', ['pose']],
  'image-feature-extraction': ['vision', ['image'], ['embedding'], 'image embeddings', ['image search']],
  'zero-shot-image-classification': ['vision', ['image', 'text'], ['label'], 'zero-shot image classification', ['clip']],
  'video-classification': ['vision', ['video'], ['label'], 'video classification', []],
  'text-to-image': ['image-generation', ['text'], ['image'], 'image generation', ['generate images', 'art', 'diffusion']],
  'image-to-image': ['image-generation', ['image', 'text'], ['image'], 'image editing', ['edit photos', 'upscale', 'inpaint']],
  'unconditional-image-generation': ['image-generation', [], ['image'], 'image generation', []],
  'text-to-3d': ['image-generation', ['text'], ['3d'], 'text to 3D', ['3d model']],
  'image-to-3d': ['image-generation', ['image'], ['3d'], 'image to 3D', ['3d model']],
  'text-to-video': ['video', ['text'], ['video'], 'video generation', ['generate video']],
  'image-to-video': ['video', ['image', 'text'], ['video'], 'image to video', ['animate image']],
  'video-to-video': ['video', ['video'], ['video'], 'video editing', []],
  'automatic-speech-recognition': ['speech-in', ['audio'], ['text'], 'speech to text', ['transcription', 'dictation', 'voice input', 'asr', 'stt']],
  'audio-classification': ['speech-in', ['audio'], ['label'], 'audio classification', ['sound detection']],
  'voice-activity-detection': ['speech-in', ['audio'], ['label'], 'voice activity detection', ['vad']],
  'text-to-speech': ['speech-out', ['text'], ['audio'], 'text to speech', ['voice', 'tts', 'read aloud', 'speech synthesis', 'voice generation']],
  'text-to-audio': ['speech-out', ['text'], ['audio'], 'audio / music generation', ['music', 'sound effects']],
  'audio-to-audio': ['speech-out', ['audio'], ['audio'], 'audio to audio', ['voice conversion', 'denoise', 'speech enhancement']],
  'image-text-to-text': ['multimodal', ['image', 'text'], ['text'], 'vision-language chat', ['vlm', 'describe images', 'ocr']],
  'image-to-text': ['multimodal', ['image'], ['text'], 'image captioning / OCR', ['caption', 'ocr']],
  'visual-question-answering': ['multimodal', ['image', 'text'], ['text'], 'visual QA', ['vlm']],
  'document-question-answering': ['multimodal', ['document', 'text'], ['text'], 'document QA', ['pdf', 'ocr']],
  'video-text-to-text': ['multimodal', ['video', 'text'], ['text'], 'video understanding', ['video chat']],
  'audio-text-to-text': ['multimodal', ['audio', 'text'], ['text'], 'audio understanding', ['voice assistant', 'speech chat']],
  'any-to-any': ['multimodal', ['text', 'image', 'audio'], ['text', 'image', 'audio'], 'omni / any-to-any', ['omni', 'voice assistant']],
  'reinforcement-learning': ['other', [], [], 'reinforcement learning', []],
  'robotics': ['other', ['image', 'text'], ['actions'], 'robotics', ['vla', 'robot']],
  'time-series-forecasting': ['other', ['series'], ['series'], 'time-series forecasting', ['forecast']],
  'tabular-classification': ['other', ['table'], ['label'], 'tabular classification', []],
  'tabular-regression': ['other', ['table'], ['number'], 'tabular regression', []],
};
const LLM_TASKS = new Set(['text-generation', 'text2text-generation', 'image-text-to-text', 'any-to-any',
  'video-text-to-text', 'audio-text-to-text', 'visual-question-answering']);
const GEN_TASKS = new Set(['text-to-image', 'image-to-image', 'text-to-video', 'image-to-video',
  'video-to-video', 'unconditional-image-generation']);
const LIBRARY_TASK = {
  'sentence-transformers': 'sentence-similarity', 'diffusers': 'text-to-image', 'whisperkit': 'automatic-speech-recognition',
  'nemo': 'automatic-speech-recognition', 'pyannote-audio': 'voice-activity-detection', 'open_clip': 'zero-shot-image-classification',
  'timm': 'image-classification', 'ultralytics': 'object-detection', 'mlx-lm': 'text-generation', 'lerobot': 'robotics',
};
// Keyword rules applied to names, descriptions and Replicate collection slugs.
const KEYWORD_TASKS = [
  [/speech[-_ ]?to[-_ ]?text|transcri|\basr\b|\bstt\b|whisper|parakeet|canary|moonshine|dictation/i, 'automatic-speech-recognition'],
  [/text[-_ ]?to[-_ ]?speech|\btts\b|voice[-_ ]?clon|kokoro|xtts|piper|chatterbox|cosyvoice|orpheus|speech synthesis|\bvoices?\b/i, 'text-to-speech'],
  [/music|song|sound[-_ ]?effect|audio[-_ ]?gen|\bsfx\b/i, 'text-to-audio'],
  [/image[-_ ]?to[-_ ]?video|img2vid|animate/i, 'image-to-video'],
  [/text[-_ ]?to[-_ ]?video|video[-_ ]?gen|\bvideo\b/i, 'text-to-video'],
  [/upscal|super[-_ ]?res|restor|remove[-_ ]?background|inpaint|image[-_ ]?edit|image[-_ ]?to[-_ ]?image|face[-_ ]?swap|relight/i, 'image-to-image'],
  [/text[-_ ]?to[-_ ]?image|image[-_ ]?gen|flux|stable[-_ ]?diffusion|sdxl|\bdiffusion\b|imagen/i, 'text-to-image'],
  [/\b3d\b|mesh|gaussian/i, 'image-to-3d'],
  [/caption|\bocr\b|vision[-_ ]?language|\bvlm\b|image[-_ ]?to[-_ ]?text/i, 'image-text-to-text'],
  [/embed|retriev|\bbge\b|\be5\b|\bgte\b/i, 'feature-extraction'],
  [/rerank/i, 'text-ranking'],
  [/segment|\bsam\b|matting/i, 'image-segmentation'],
  [/detect|\byolo\b/i, 'object-detection'],
  [/\bllm\b|language[-_ ]?model|\bchat\b|instruct|reasoning|\bcoder?\b|\bgpt\b/i, 'text-generation'],
];

const PERMISSIVE = new Set(['apache-2.0', 'mit', 'cc-by-4.0', 'cc0-1.0', 'bigscience-openrail-m', 'creativeml-openrail-m',
  'mpl-2.0', 'unlicense', 'cc-by-sa-4.0', 'agpl-3.0', 'afl-3.0', 'artistic-2.0', 'isc', 'wtfpl', 'ecl-2.0', 'zlib',
  'openrail', 'openrail++', 'bsd', 'bsd-2-clause', 'bsd-3-clause', 'gpl-3.0', 'lgpl-3.0']);
const NONCOMMERCIAL = /(^|-)nc(-|$)|non[-_]?commercial|research/i;
const CUSTOM = /^(llama|gemma|qwen|deepseek|other|openmdw|health-ai|nvidia|fair|flux)/i;

// Accounts/orgs whose brand-new uploads are worth keeping even before they get downloads.
const NOTABLE_AUTHORS = new Set(['meta-llama', 'google', 'microsoft', 'qwen', 'deepseek-ai', 'mistralai', 'nvidia',
  'openai', 'apple', 'liquidai', 'huggingfacetb', 'stabilityai', 'black-forest-labs', 'tencent', 'baidu', 'zai-org',
  'moonshotai', 'minimaxai', 'ibm-granite', 'allenai', 'facebook', 'hexgrad', 'resembleai', 'kyutai', 'sesame',
  'unsloth', 'bartowski', 'lmstudio-community', 'mlx-community', 'ggml-org', 'onnx-community', 'openbmb',
  'stepfun-ai', 'xiaomimimo', 'inclusionai', 'bytedance-seed', 'bytedance', 'lightricks', 'wan-ai', 'nari-labs',
  'canopylabs', 'fishaudio', 'boson-ai', 'coqui', 'argmaxinc', 'pytorch', 'executorch-community', 'ai21labs', 'cohere',
  'coherelabs', 'swiss-ai', 'servicenow-ai', 'arcee-ai', 'nousresearch', 'prime-intellect', 'thudm']);

// ---------------------------------------------------------------- helpers
const str = x => (typeof x === 'string' && x.trim() ? x.trim() : null);
const arr = x => (Array.isArray(x) ? x : x == null ? [] : [x]);
const uniq = xs => [...new Set(xs.filter(v => v !== null && v !== undefined && v !== ''))];
const num = x => (typeof x === 'number' && isFinite(x) ? x : null);
const round = (x, d = 2) => (x == null ? null : Math.round(x * 10 ** d) / 10 ** d);
const norm = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const log10 = x => Math.log10(1 + Math.max(0, x || 0));

function paramsLabel(n) {
  if (!n) return null;
  if (n >= 1e9) return `${round(n / 1e9, n >= 1e10 ? 0 : 1)}B`;
  return `${Math.round(n / 1e6)}M`;
}

function paramsFromName(name) {
  const base = (name || '').split('/').pop();
  const re = /(?<![A-Za-z0-9])(?:(\d+)[xX])?(\d+(?:\.\d+)?)([BbMm])(?![A-Za-z0-9])/g;
  let m;
  while ((m = re.exec(base))) {
    const n = Math.round((m[1] ? +m[1] : 1) * +m[2] * (/b/i.test(m[3]) ? 1e9 : 1e6));
    if (n >= 1e6 && n <= 1e13) return n;
  }
  return null;
}
function activeParamsFromName(name) {
  const m = (name || '').match(/(?<![A-Za-z0-9])A(\d+(?:\.\d+)?)B(?![A-Za-z0-9])/i);
  return m ? Math.round(+m[1] * 1e9) : null;
}

const QUANT_RES = [
  /(?<![A-Z0-9])(IQ[1-4]_(?:XXS|XS|S|M|NL))(?![A-Z0-9])/, /(?<![A-Z0-9])(Q[2-8]_K(?:_(?:XL|[SML]))?)(?![A-Z0-9])/,
  /(?<![A-Z0-9])(Q[4-8]_[01])(?![A-Z0-9])/, /(?<![A-Z0-9])(TQ[12]_0)(?![A-Z0-9])/, /(?<![A-Z0-9])(MXFP4)(?![A-Z0-9])/,
  /(?<![A-Z0-9])(BF16)(?![A-Z0-9])/, /(?<![A-Z0-9])(F16)(?![A-Z0-9])/, /(?<![A-Z0-9])(F32)(?![A-Z0-9])/,
];
function ggufQuants(files) {
  const out = [];
  for (const f of files.filter(f => /\.gguf$/i.test(f) && !/mmproj/i.test(f))) {
    const base = f.split('/').pop().toUpperCase();
    for (const re of QUANT_RES) { const m = base.match(re); if (m) { out.push(m[1]); break; } }
  }
  return uniq(out).sort((a, b) => (BPW[a] || 99) - (BPW[b] || 99));
}

function detectFormats(files, tags, library, id) {
  const f = new Set();
  const has = re => files.some(x => re.test(x));
  const tag = t => tags.includes(t);
  if (has(/\.safetensors$/i) || tag('safetensors')) f.add('safetensors');
  if (has(/\.gguf$/i) || tag('gguf') || library === 'gguf') f.add('gguf');
  if (has(/\.onnx(_data)?$/i) || tag('onnx') || library === 'transformers.js') f.add('onnx');
  if (has(/\.mlpackage\//i) || has(/\.mlmodelc\//i) || has(/\.mlmodel$/i) || tag('coreml') || library === 'whisperkit') f.add('coreml');
  if (has(/\.pte$/i) || tag('executorch')) f.add('executorch');
  if (has(/\.tflite$/i) || tag('tflite') || tag('litert') || has(/\.task$/i) || has(/\.litertlm$/i)) f.add('tflite');
  if (tag('mlx') || library === 'mlx' || /^mlx-community\//i.test(id)) f.add('mlx');
  if (has(/(^|\/)pytorch_model.*\.bin$/i) || has(/\.pt$|\.pth$|\.ckpt$/i)) f.add('pytorch');
  if (library === 'diffusers' || has(/model_index\.json$/)) f.add('diffusers');
  if (tag('awq') || /awq/i.test(id)) f.add('awq');
  if (tag('gptq') || /gptq/i.test(id)) f.add('gptq');
  if (tag('exl2') || /exl2/i.test(id)) f.add('exl2');
  if (tag('bitsandbytes') || /bnb/i.test(id)) f.add('bnb');
  if (has(/\.nemo$/i) || library === 'nemo') f.add('nemo');
  return [...f].sort();
}

function quantBitsFromName(id, tags) {
  const s = `${id} ${tags.join(' ')}`.toLowerCase();
  let m = s.match(/(?<![a-z0-9])(\d+(?:\.\d+)?)-?bit(?![a-z0-9])/);
  if (m) return +m[1];
  m = s.match(/(?<![a-z0-9])(?:nv)?(?:mx)?(fp|int)([48])(?![0-9])/);
  if (m) return +m[2];
  if (/awq|gptq/.test(s)) return 4;
  return null;
}

function inferTask(raw) {
  const card = raw.cardData || {};
  const tags = uniq([...arr(raw.tags), ...arr(card.tags)]);
  if (str(raw.pipeline_tag)) return [raw.pipeline_tag, 'pipeline_tag'];
  if (str(card.pipeline_tag)) return [card.pipeline_tag, 'card'];
  const t = tags.find(t => TASKS[t]);
  if (t) return [t, 'tags'];
  const lib = (str(raw.library_name) || str(card.library_name) || '').toLowerCase();
  if (LIBRARY_TASK[lib]) return [LIBRARY_TASK[lib], 'library'];
  const byName = keywordTask(`${raw.id} ${tags.join(' ')}`);
  if (byName) return [byName, 'name'];
  return [null, 'unknown'];
}
function keywordTask(text) {
  for (const [re, task] of KEYWORD_TASKS) if (re.test(text || '')) return task;
  return null;
}

function estimateRamGb(params, bpw, task) {
  if (!params || !bpw) return null;
  let ram = (params * bpw / 8 / 1e9) * OVERHEAD;
  if (LLM_TASKS.has(task)) ram += 0.17 * Math.sqrt(params / 1e9);
  if (GEN_TASKS.has(task)) ram += GEN_EXTRA_GB;
  return round(ram, 2);
}
function practical(v) {
  const u = (v || '').toUpperCase();
  return !(/^(IQ1|IQ2|Q2|TQ)/.test(u) || /^[12](\.\d+)?BIT$/.test(u));
}

function licenseInfo(id, url, name) {
  const lid = str(id) ? id.toLowerCase() : null;
  let commercial = 'unknown';
  if (lid && PERMISSIVE.has(lid)) commercial = 'yes';
  else if ((lid && NONCOMMERCIAL.test(lid)) || (name && NONCOMMERCIAL.test(name))) commercial = 'no';
  else if (lid && CUSTOM.test(lid)) commercial = 'check_terms';
  return { id: lid, name: str(name), url: str(url), commercial_use: commercial };
}

function hfRepoFromText(text) {
  const out = [];
  const re = /huggingface\.co\/(?!datasets\/|spaces\/|papers\/|docs\/|blog\/)([A-Za-z0-9][\w.-]*\/[A-Za-z0-9][\w.-]*)/g;
  let m;
  while ((m = re.exec(text || ''))) out.push(m[1].replace(/[.…]+$/, ''));
  return uniq(out);
}

// ---------------------------------------------------------------- 1. Hugging Face
const report = { fetched_at: FETCHED_AT, sources: {}, dropped: {}, joins: {}, warnings: sourceErrors };
const hfById = new Map();
for (const [list, label] of [[hfTrendingRaw, 'trending'], [hfVariantsRaw, 'variant_lookup'], [hfNewestRaw, 'newest']]) {
  for (const m of list) {
    if (!str(m.id)) continue;
    const prev = hfById.get(m.id);
    if (prev) { prev._lists.add(label); continue; }
    hfById.set(m.id, { ...m, _lists: new Set([label]) });
  }
}
report.sources.huggingface = { trending: hfTrendingRaw.length, newest: hfNewestRaw.length,
  quantized_variants: hfVariantsRaw.length, unique: hfById.size };

const entities = new Map(); // key -> entity
const nameIndex = new Map(); // normalized name -> key

function indexName(name, key) {
  const n = norm(name);
  if (n.length >= 4 && !nameIndex.has(n)) nameIndex.set(n, key);
}

function newEntity(key, displayName) {
  return {
    id: key, name: displayName, aliases: [], status: 'released',
    task: null, size: null, formats: [], quantizations: [], hardware: null, license: null,
    languages: [], lineage: null,
    availability: { huggingface: null, replicate: null },
    popularity: {}, social: null,
    links: { huggingface: null, replicate: null, github: null, paper: null, x_posts: [] },
    dates: {}, description: null,
    _tags: [], _taskSource: null, _sources: new Set(),
  };
}

for (const m of hfById.values()) {
  const card = m.cardData || {};
  const tags = uniq([...arr(m.tags), ...arr(card.tags)]);
  const files = arr(m.siblings).map(s => s.rfilename).filter(Boolean);
  const library = (str(m.library_name) || str(card.library_name) || '').toLowerCase() || null;
  const [task, taskSource] = inferTask(m);
  const key = `hf:${m.id}`;
  const e = newEntity(key, m.id.split('/').pop());
  e._sources.add('huggingface');
  e._tags = tags;
  e._taskSource = taskSource;
  e._task = task;

  // size
  const stTotal = num(m.safetensors && m.safetensors.total);
  const ggTotal = num(m.gguf && m.gguf.total);
  const nameParams = paramsFromName(m.id);
  // Single-file / ComfyUI-style repos often only state the size in weight filenames ("...-22b-dev-bf16.safetensors").
  const fileParams = nameParams ? null : Math.max(0, ...files
    .filter(f => /\.(safetensors|gguf|bin|pt|pth|ckpt)$/i.test(f) && !/lora|vae|upscal|text_encoder|mmproj/i.test(f))
    .map(f => paramsFromName(f.replace(/\//g, '-')) || 0)) || null;
  let params = stTotal || ggTotal || nameParams || fileParams;
  const paramsSource = stTotal ? 'safetensors' : ggTotal ? 'gguf_header' : nameParams ? 'name' : fileParams ? 'filename' : null;
  const stDtypes = m.safetensors && m.safetensors.parameters ? Object.keys(m.safetensors.parameters) : [];
  // packed int dtypes (U8/I32) mean the safetensors count is of packed storage, not real params
  if (stTotal && stDtypes.some(d => ['U32', 'I32', 'U8'].includes(d)) && nameParams) params = nameParams;
  e._params = params;
  e.size = {
    params, params_label: paramsLabel(params), params_active: activeParamsFromName(m.id),
    params_source: paramsSource, native_dtypes: stDtypes,
    context_length: num(m.gguf && m.gguf.context_length),
    architecture: str(m.gguf && m.gguf.architecture) || null,
  };

  e.formats = detectFormats(files, tags, library, m.id);
  e.quantizations = ggufQuants(files);
  e._quantBits = quantBitsFromName(m.id, tags);

  const lineage = m.baseModels && arr(m.baseModels.models).length
    ? { relation: str(m.baseModels.relation), base_models: arr(m.baseModels.models).map(b => b.id) }
    : (card.base_model ? { relation: str(card.base_model_relation), base_models: arr(card.base_model) } : null);
  e.lineage = lineage;

  const licTag = (tags.find(t => t.startsWith('license:')) || '').slice(8) || null;
  e.license = licenseInfo(str(card.license) || licTag, str(card.license_link), str(card.license_name));
  e.languages = uniq(arr(card.language).map(l => String(l).toLowerCase()));

  e.availability.huggingface = {
    repo: m.id, author: m.author || m.id.split('/')[0], library, gated: m.gated || false,
    lists: [...m._lists], file_count: m.file_count ?? files.length,
    inference_providers: arr(m.inferenceProviderMapping).map(p => p.provider).filter(Boolean),
  };
  e.links.huggingface = `https://huggingface.co/${m.id}`;
  e.popularity = {
    hf_downloads_30d: num(m.downloads) || 0, hf_likes: num(m.likes) || 0, hf_trending_score: num(m.trendingScore) || 0,
  };
  e.dates = { created_at: str(m.createdAt), last_modified: str(m.lastModified) };
  entities.set(key, e);
  indexName(m.id.split('/').pop(), key);
  indexName(m.id, key);
}

// ---------------------------------------------------------------- 2. Replicate
const repModels = new Map();
for (const r of repPages) if (r && r.owner && r.name) repModels.set(`${r.owner}/${r.name}`, r);
const repCollectionsBySlug = {};
for (const c of repCollections) {
  if (!c || !c.slug) continue;
  for (const r of arr(c.models)) {
    if (!r || !r.owner || !r.name) continue;
    const k = `${r.owner}/${r.name}`;
    if (!repModels.has(k)) repModels.set(k, r);
    (repCollectionsBySlug[k] = repCollectionsBySlug[k] || []).push(c.slug);
  }
}
report.sources.replicate = { models: repModels.size, collections: repCollections.length,
  note: repPages.length || repCollections.length ? undefined : 'no Replicate data (missing REPLICATE_API_TOKEN?)' };

let repJoined = 0;
for (const [k, r] of repModels) {
  const text = [r.description, r.github_url, r.paper_url, r.weights_url, r.license_url].filter(Boolean).join(' ');
  const hfRepos = hfRepoFromText(text);
  let key = hfRepos.map(h => `hf:${h}`).find(h => entities.has(h));
  if (!key) key = nameIndex.get(norm(r.name));
  let e = key ? entities.get(key) : null;
  if (e) repJoined++;
  else {
    key = `replicate:${k}`;
    e = newEntity(key, r.name);
    e.status = 'api_only';
    entities.set(key, e);
    indexName(r.name, key);
  }
  e._sources.add('replicate');
  const collections = uniq(repCollectionsBySlug[k] || []);
  e.availability.replicate = {
    model: k, url: r.url || `https://replicate.com/${k}`, run_count: num(r.run_count) || 0,
    visibility: r.visibility || null, is_official: r.is_official ?? collections.includes('official'), collections,
    inputs: (r.latest_version && r.latest_version.input_names) || [],
    latest_version: r.latest_version ? r.latest_version.id : null,
    cover_image_url: r.cover_image_url || null, hf_repos_mentioned: hfRepos,
  };
  e.links.replicate = e.availability.replicate.url;
  e.links.github = e.links.github || str(r.github_url);
  e.links.paper = e.links.paper || str(r.paper_url);
  e.popularity.replicate_runs = num(r.run_count) || 0;
  if (!e.description && str(r.description)) e.description = r.description.trim().slice(0, 600);
  if (!e._task) {
    const t = keywordTask(collections.join(' ')) || keywordTask(`${r.name} ${r.description || ''}`);
    if (t) { e._task = t; e._taskSource = collections.length ? 'replicate_collection' : 'replicate_text'; }
  }
  if (!e.license || !e.license.id) e.license = licenseInfo(null, r.license_url, null);
  if (!e.dates.created_at && r.latest_version) e.dates.created_at = str(r.latest_version.created_at);
}
report.joins.replicate_matched_to_hf = repJoined;

// ---------------------------------------------------------------- 3. X signals
const xLatest = new Map();
let xBad = 0;
for (const line of xText.split('\n')) {
  if (!line.trim()) continue;
  let rec;
  try { rec = JSON.parse(line); } catch (e) { xBad++; continue; }
  if (!rec || !rec.url || rec.status) continue;
  const prev = xLatest.get(rec.url);
  if (!prev || String(rec.seen_at) > String(prev.seen_at)) xLatest.set(rec.url, rec);
}
report.sources.x = { posts: xLatest.size, unparseable_lines: xBad };

const X_WINDOW_DAYS = 14;
let xMatched = 0, xAnnounced = 0;
for (const p of xLatest.values()) {
  const ageDays = p.posted_at ? (Date.now() - Date.parse(p.posted_at)) / 864e5 : 0;
  if (ageDays > X_WINDOW_DAYS) continue;
  const keys = new Set();
  for (const h of uniq([...arr(p.hf_repos), ...hfRepoFromText(arr(p.other_links).join(' '))])) {
    const clean = h.replace(/[.…]+$/, '');
    if (entities.has(`hf:${clean}`)) keys.add(`hf:${clean}`);
    else {
      const k = nameIndex.get(norm(clean)) || nameIndex.get(norm(clean.split('/').pop()));
      if (k) keys.add(k);
      else if (/^[\w.-]+\/[\w.-]+$/.test(clean) && !clean.endsWith('-')) {
        const key = `hf:${clean}`;
        const e = newEntity(key, clean.split('/').pop());
        e.status = 'seen_on_x';
        e.links.huggingface = `https://huggingface.co/${clean}`;
        e._needsHf = true;
        entities.set(key, e);
        indexName(clean.split('/').pop(), key);
        keys.add(key);
      }
    }
  }
  for (const name of arr(p.models)) {
    const k = nameIndex.get(norm(name));
    if (k) { keys.add(k); continue; }
    if (keys.size) continue; // already attached via a link
    const key = `x:${norm(name)}`;
    if (norm(name).length < 3) continue;
    if (!entities.has(key)) {
      const e = newEntity(key, String(name).trim());
      e.status = p.kind === 'weights_soon' ? 'announced' : 'seen_on_x';
      entities.set(key, e);
      indexName(name, key);
      xAnnounced++;
    }
    keys.add(key);
  }
  if (keys.size) xMatched++;
  const likes = num(p.likes) || 0, reposts = num(p.reposts) || 0;
  const engagement = likes + 3 * reposts;
  const decayed = engagement * Math.exp(-ageDays / 7);
  for (const key of keys) {
    const e = entities.get(key);
    e._sources.add('x');
    const s = e.social || (e.social = { x_mentions: 0, x_engagement: 0, x_buzz: 0, takes: [], top_posts: [],
      kinds: [], size_hints: [], license_hints: [], on_device_mentions: 0, modality_hints: [], first_seen: null });
    s.x_mentions++;
    s.x_engagement += engagement;
    s.x_buzz += decayed;
    if (str(p.take)) s.takes.push(p.take.trim());
    s.top_posts.push({ url: p.url, author: p.author_handle || null, posted_at: p.posted_at || null, likes, reposts,
      views: num(p.views), take: str(p.take), excerpt: (p.text || '').replace(/\s+/g, ' ').slice(0, 280) });
    s.kinds.push(p.kind);
    if (str(p.size_hints)) s.size_hints.push(p.size_hints);
    if (str(p.license_hints)) s.license_hints.push(p.license_hints);
    if (p.on_device) s.on_device_mentions++;
    s.modality_hints.push(...arr(p.modality));
    if (!s.first_seen || String(p.posted_at) < s.first_seen) s.first_seen = p.posted_at || null;
  }
}
report.joins.x_posts_attached = xMatched;
report.joins.x_new_announced_models = xAnnounced;

// ---------------------------------------------------------------- 4. finish each entity
const MODALITY_TASK = { speech_to_text: 'automatic-speech-recognition', text_to_speech: 'text-to-speech',
  voice_clone: 'text-to-speech', image_gen: 'text-to-image', video_gen: 'text-to-video', code: 'text-generation',
  text: 'text-generation', vision: 'image-text-to-text', multimodal: 'any-to-any', embedding: 'feature-extraction',
  audio: 'text-to-audio' };

let droppedLowSignal = 0;
const out = [];
for (const e of entities.values()) {
  const hf = e.availability.huggingface;
  const s = e.social;
  if (s) {
    s.top_posts.sort((a, b) => (b.likes + 3 * b.reposts) - (a.likes + 3 * a.reposts));
    s.top_posts = s.top_posts.slice(0, 5);
    s.takes = uniq(s.takes).slice(0, 5);
    s.kinds = uniq(s.kinds);
    s.size_hints = uniq(s.size_hints).slice(0, 3);
    s.license_hints = uniq(s.license_hints).slice(0, 3);
    s.modality_hints = uniq(s.modality_hints);
    s.x_buzz = round(s.x_buzz, 1);
    e.links.x_posts = s.top_posts.map(p => p.url);
  }

  // Quality gate: skip brand-new HF uploads nobody uses or talks about.
  if (hf && !e._sources.has('x') && !e._sources.has('replicate')) {
    const p = e.popularity;
    const keep = hf.lists.includes('trending') || p.hf_downloads_30d >= 25 || p.hf_likes >= 2 ||
      NOTABLE_AUTHORS.has((hf.author || '').toLowerCase());
    if (!keep) { droppedLowSignal++; continue; }
  }

  // task
  let task = e._task;
  if (!task && s) {
    const t = s.modality_hints.map(m => MODALITY_TASK[m]).find(Boolean);
    if (t) { task = t; e._taskSource = 'x_post'; }
  }
  if (!task) { task = keywordTask(`${e.id} ${e.description || ''} ${s ? s.takes.join(' ') : ''}`); if (task) e._taskSource = 'keywords'; }
  const t = TASKS[task] || null;
  const tagTasks = uniq(e._tags.filter(x => TASKS[x]));
  const voiceClone = /voice[-_ ]?clon|zero[-_ ]?shot[-_ ]?tts|speaker[-_ ]?adapt/i.test(`${e.id} ${e._tags.join(' ')} ${e.description || ''}`) ||
    (s && s.modality_hints.includes('voice_clone'));
  e.task = {
    primary: task || null, label: t ? t[3] : null, group: t ? t[0] : null,
    all: uniq([task, ...tagTasks]),
    inputs: t ? t[1] : [], outputs: t ? t[2] : [],
    capabilities: uniq([...(t ? t[4] : []), voiceClone ? 'voice cloning' : null,
      e._tags.includes('code') || /coder|code/i.test(e.id) ? 'coding' : null,
      e._tags.some(x => /reasoning|thinking/.test(x)) ? 'reasoning' : null,
      e._tags.includes('tool-use') || e._tags.includes('function-calling') ? 'tool calling' : null]),
    source: task ? e._taskSource : null,
  };

  if (!e._params && s && s.size_hints.length) {
    const p = paramsFromName(s.size_hints.join(' ').replace(/\s+/g, ''));
    if (p) { e._params = p; e.size = { params: p, params_label: paramsLabel(p), params_active: null, params_source: 'x_post' }; }
  }

  // popularity score 0-100 (filled after percentile pass)
  e.popularity = {
    hf_downloads_30d: e.popularity.hf_downloads_30d || 0, hf_likes: e.popularity.hf_likes || 0,
    hf_trending_score: e.popularity.hf_trending_score || 0, replicate_runs: e.popularity.replicate_runs || 0,
    x_mentions: s ? s.x_mentions : 0, x_engagement: s ? s.x_engagement : 0, x_buzz: s ? s.x_buzz : 0,
    _raw: log10(e.popularity.hf_downloads_30d) * 1.0 + log10(e.popularity.hf_likes) * 1.5 +
      log10(e.popularity.hf_trending_score) * 2.0 + log10(e.popularity.replicate_runs) * 0.6 + log10(s ? s.x_buzz : 0) * 1.5,
  };

  if (!e.license) e.license = licenseInfo(null, null, null);
  if (s && !e.license.id && s.license_hints.length) e.license.hint_from_x = s.license_hints[0];
  out.push(e);
}

// Quantized / converted copies (unsloth GGUF, mlx-community 4bit, FP8...) inherit size, task and license from their base.
const outByKey = new Map(out.map(e => [e.id, e]));
const QUANT_REL = new Set(['quantized']);
for (const e of out) {
  if (!e.lineage || !QUANT_REL.has(e.lineage.relation)) continue;
  const base = e.lineage.base_models.map(b => outByKey.get(`hf:${b}`)).find(Boolean);
  if (!base) continue;
  // Repos sometimes mislabel distills / different sizes as "quantized". A real quantized copy has
  // about the same parameter count (packed 4-bit storage can under-count, so compare loosely).
  const exact = e.size && ['safetensors', 'gguf_header'].includes(e.size.params_source);
  const nameP = paramsFromName(e.id);
  const childP = nameP || (exact ? null : e._params);
  if (base._params && childP && (childP / base._params < 0.8 || childP / base._params > 1.25)) { // same band as enrich.py family linking
    e._notQuantOf = base.id;
    continue;
  }
  e._base = base;
  e.lineage.variant_of = base.id; // lets retrieval collapse copies under their original
  if (base._params) {
    e._params = base._params;
    e.size = { ...(e.size || {}), params: base._params, params_label: paramsLabel(base._params), params_source: 'base_model' };
  }
  if (e.size && !e.size.params_active && base.size) e.size.params_active = base.size.params_active;
  if (!e.task.primary && base.task.primary) e.task = { ...base.task, source: 'base_model' };
  if (!e.license.id && base.license.id) e.license = { ...base.license, inherited_from: base.id };
  if (!e.languages.length) e.languages = base.languages;
}

function hardwareFor(e) {
  const params = e._params;
  const task = e.task.primary;
  const variants = {};
  for (const q of e.quantizations) if (BPW[q]) variants[q] = BPW[q];
  if (e._quantBits) variants[`${e._quantBits}bit`] = e._quantBits >= 16 ? e._quantBits : e._quantBits + 0.5;
  const nativeBits = e.size && e.size.native_dtypes && e.size.native_dtypes.includes('F32') && e.size.native_dtypes.length === 1 ? 32
    : e.formats.includes('pytorch') && !e.formats.includes('safetensors') ? 32 : 16;
  if (!Object.keys(variants).length || (!e._quantBits && e.formats.some(f => ['safetensors', 'pytorch', 'diffusers', 'coreml', 'executorch', 'tflite', 'mlx'].includes(f))))
    variants.native = nativeBits;
  const estimates = Object.entries(variants)
    .map(([v, bpw]) => ({ variant: v, bits_per_weight: bpw, est_ram_gb: estimateRamGb(params, bpw, task) }))
    .filter(x => x.est_ram_gb != null)
    .sort((a, b) => a.est_ram_gb - b.est_ram_gb);
  const usable = estimates.filter(x => practical(x.variant));
  if (!estimates.length && e.social && e.social.size_hints.length) {
    // X-only models: "under 300 MB" / "4.5 GB" -> treat the stated download size as the weights.
    const m = e.social.size_hints.join(' ').match(/(\d+(?:\.\d+)?)\s*(MB|GB)/i);
    if (m) {
      const gb = +m[1] / (/mb/i.test(m[2]) ? 1000 : 1);
      estimates.push({ variant: 'as_announced', bits_per_weight: null, est_ram_gb: round(gb * OVERHEAD, 2) });
    }
  }
  return { estimates, smallest: usable[0] || estimates[0] || null };
}
function tiersFor(ram, formats, saysOnDevice) {
  if (ram == null) return [];
  const t = [];
  if (ram <= TIER_BUDGET_GB.phone && (formats.some(f => PHONE_FORMATS.has(f)) || saysOnDevice)) t.push('phone');
  if (ram <= TIER_BUDGET_GB.laptop) t.push('laptop');
  if (ram <= TIER_BUDGET_GB.gpu) t.push('gpu');
  t.push('server');
  return t;
}

for (const e of out) {
  const { estimates, smallest } = hardwareFor(e);
  e._smallest = smallest;
  e.hardware = {
    est_min_ram_gb: smallest ? smallest.est_ram_gb : null,
    smallest_variant: smallest ? smallest.variant : null,
    variants: estimates.slice(0, 8),
    // An X-only model counts as phone-ready only when posts say it runs on-device.
    runs_on: tiersFor(smallest && smallest.est_ram_gb, e.formats,
      !e.availability.huggingface && !!(e.social && e.social.on_device_mentions)),
    on_device_ready: e.formats.filter(f => PHONE_FORMATS.has(f)),
    best_option: null,
    community_variants: [],
    notes: [],
  };
}

// Roll each base model's community variants up into it, so "Qwen 27B" shows it fits a laptop via someone's GGUF.
let rolledUp = 0;
for (const e of out) {
  const base = e._base;
  if (!base || !e._smallest) continue;
  rolledUp++;
  base.hardware.community_variants.push({
    repo: e.availability.huggingface ? e.availability.huggingface.repo : e.id, formats: e.formats,
    smallest_variant: e._smallest.variant, est_ram_gb: e._smallest.est_ram_gb,
    runs_on: e.hardware.runs_on, downloads_30d: e.popularity.hf_downloads_30d,
  });
}
report.joins.quantized_variants_rolled_up = rolledUp;

for (const e of out) {
  const s = e.social;
  const hf = e.availability.huggingface;
  const h = e.hardware;
  h.community_variants.sort((a, b) => b.downloads_30d - a.downloads_30d);
  h.community_variants = h.community_variants.slice(0, 8);
  const own = e._smallest ? { repo: hf ? hf.repo : e.id, variant: e._smallest.variant, est_ram_gb: e._smallest.est_ram_gb, runs_on: h.runs_on } : null;
  const options = [own, ...h.community_variants.map(v => ({ repo: v.repo, variant: v.smallest_variant, est_ram_gb: v.est_ram_gb, runs_on: v.runs_on }))]
    .filter(Boolean);
  // best = smallest RAM among options that are practical; prefer phone-capable ones
  options.sort((a, b) => (b.runs_on.includes('phone') - a.runs_on.includes('phone')) || a.est_ram_gb - b.est_ram_gb);
  h.best_option = options[0] || null;
  h.runs_on_any_variant = uniq(options.flatMap(o => o.runs_on));
  const phoneMaybe = h.est_min_ram_gb != null && h.est_min_ram_gb <= TIER_BUDGET_GB.phone && !h.runs_on_any_variant.includes('phone');
  h.notes = uniq([
    phoneMaybe ? 'small enough for a phone, but no phone-ready format (GGUF/MLX/CoreML/ONNX/ExecuTorch/TFLite) published yet' : null,
    !e._params ? 'size unknown' : null,
    e.status === 'api_only' ? 'hosted API only (Replicate); weights may not be downloadable' : null,
    e.size && e.size.params_active ? `mixture-of-experts: ${paramsLabel(e.size.params_active)} active per token, but all weights must fit in memory` : null,
    h.best_option && own && h.best_option.repo !== own.repo ? `smallest practical copy: ${h.best_option.repo} (${h.best_option.variant}, ~${h.best_option.est_ram_gb} GB)` : null,
    s && s.on_device_mentions ? 'people on X mention running it on-device' : null,
  ]);

  // plain-English summary + the text that will later be embedded
  const sizeTxt = e.size && e.size.params_label ? `${e.size.params_label} parameters` : 'unknown size';
  const tiers = h.runs_on_any_variant;
  const where = tiers.length ? `runs on: ${tiers.join(', ')}` : (e.status === 'api_only' ? 'hosted API' : 'hardware unknown');
  const best = h.best_option;
  e.summary = `${e.name}: ${e.task.label || 'unknown task'}, ${sizeTxt}, ${where}` +
    (best ? ` (~${best.est_ram_gb} GB RAM with ${best.variant}${own && best.repo !== own.repo ? ` from ${best.repo}` : ''})` : '') + '.';
  e.search_text = [
    e.summary,
    e.task.capabilities.length ? `Good for: ${e.task.capabilities.join(', ')}.` : '',
    e.task.inputs.length ? `Input: ${e.task.inputs.join(' + ')}. Output: ${e.task.outputs.join(' + ')}.` : '',
    e.formats.length ? `Formats: ${e.formats.join(', ')}${e.quantizations.length ? ` (${e.quantizations.slice(0, 6).join(', ')})` : ''}.` : '',
    h.community_variants.length ? `Community copies: ${uniq(h.community_variants.flatMap(v => v.formats)).join(', ')}.` : '',
    e.license.id ? `License: ${e.license.id} (commercial use: ${e.license.commercial_use}).` : '',
    e.languages.length ? `Languages: ${e.languages.slice(0, 12).join(', ')}.` : '',
    e.lineage && e.lineage.base_models.length ? `Based on ${e.lineage.base_models.join(', ')} (${e.lineage.relation || 'derived'}).` : '',
    e.description ? `Description: ${e.description}` : '',
    s && s.takes.length ? `What people say: ${s.takes.slice(0, 3).join(' ')}` : '',
    hf ? `Tags: ${e._tags.filter(t => !/^(region|endpoints|license|base_model|arxiv|dataset):/.test(t)).slice(0, 20).join(', ')}.` : '',
  ].filter(Boolean).join(' ');

  // data quality
  const missing = [];
  if (!e.task.primary) missing.push('task');
  if (!e._params) missing.push('params');
  if (!e.license.id) missing.push('license');
  if (!e.formats.length && e.status !== 'api_only') missing.push('formats');
  if (!e.languages.length && ['text', 'speech-in', 'speech-out', 'multimodal'].includes(e.task.group)) missing.push('languages');
  if (!e.description && !hf) missing.push('description');
  const inferred = [];
  if (e.task.source && !['pipeline_tag', 'card', 'tags'].includes(e.task.source)) inferred.push(`task:${e.task.source}`);
  if (e.size && e.size.params_source && !['safetensors', 'gguf_header'].includes(e.size.params_source)) inferred.push(`params:${e.size.params_source}`);
  if (e.license.inherited_from) inferred.push('license:base_model');
  e.quality = {
    completeness: round(1 - missing.length / 6, 2), missing, inferred,
    needs_enrichment: missing.length > 0 || !!e._needsHf,
    needs_hf_fetch: !!e._needsHf,
  };
  e.provenance = { sources: [...e._sources].sort(), fetched_at: FETCHED_AT, schema_version: SCHEMA_VERSION };
}
report.dropped.low_signal_new_hf_uploads = droppedLowSignal;

// popularity percentile -> 0-100
const sorted = out.map(e => e.popularity._raw).sort((a, b) => a - b);
for (const e of out) {
  const r = e.popularity._raw;
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < r) lo = mid + 1; else hi = mid; }
  e.popularity.score = sorted.length > 1 ? Math.round((lo / (sorted.length - 1)) * 100) : 100;
  delete e.popularity._raw;
}

// strip working fields, order keys
const ORDER = ['id', 'name', 'status', 'summary', 'task', 'size', 'hardware', 'formats', 'quantizations', 'license',
  'languages', 'lineage', 'availability', 'popularity', 'social', 'links', 'dates', 'description', 'search_text',
  'quality', 'provenance'];
const clean = out
  .sort((a, b) => b.popularity.score - a.popularity.score)
  .map(e => Object.fromEntries(ORDER.map(k => [k, e[k] === undefined ? null : e[k]])));

report.output = {
  models: clean.length,
  by_source_mix: clean.reduce((acc, e) => { const k = e.provenance.sources.join('+'); acc[k] = (acc[k] || 0) + 1; return acc; }, {}),
  by_task_group: clean.reduce((acc, e) => { const k = e.task.group || 'unknown'; acc[k] = (acc[k] || 0) + 1; return acc; }, {}),
  runs_on_phone: clean.filter(e => e.hardware.runs_on_any_variant.includes('phone')).length,
  avg_completeness: round(clean.reduce((a, e) => a + e.quality.completeness, 0) / Math.max(1, clean.length), 2),
};

return [{ json: { models: clean, report } }];
