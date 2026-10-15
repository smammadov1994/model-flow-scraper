const LABELS: Record<string, string> = {
  'automatic-speech-recognition': 'Speech to text',
  'text-generation': 'Chat / text',
  'text2text-generation': 'Text rewriting',
  'text-to-speech': 'Text to speech',
  'text-to-audio': 'Audio generation',
  'text-to-image': 'Image generation',
  'text-to-video': 'Video generation',
  'image-to-image': 'Image editing',
  'feature-extraction': 'Embeddings',
  'sentence-similarity': 'Embeddings',
  'image-text-to-text': 'Vision + chat',
  'image-to-text': 'Image captioning',
  'image-classification': 'Image recognition',
  'object-detection': 'Object detection',
  'image-segmentation': 'Image segmentation',
  'depth-estimation': 'Depth estimation',
  'audio-classification': 'Sound recognition',
  'voice-activity-detection': 'Voice detection',
  'text-classification': 'Text classification',
  'token-classification': 'Entity extraction',
  'zero-shot-classification': 'Zero-shot classification',
  'question-answering': 'Question answering',
  'summarization': 'Summarization',
  'translation': 'Translation',
  'fill-mask': 'Fill in the blank',
  'document-question-answering': 'Document Q&A',
  'visual-question-answering': 'Visual Q&A',
  'reranking': 'Re-ranking',
}

export function taskLabel(task: string | null | undefined): string {
  if (!task) return 'Model'
  const known = LABELS[task]
  if (known) return known
  return task.replace(/-/g, ' ')
}
