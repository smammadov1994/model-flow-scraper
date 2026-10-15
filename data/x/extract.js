// Paste into the X tab (built-in browser javascript_tool) to pull visible posts as JSON.
// Read-only: it only inspects the DOM, never clicks.
[...document.querySelectorAll('article[data-testid="tweet"]')].map(a => {
  const href = [...a.querySelectorAll('a[href*="/status/"]')]
    .map(x => x.getAttribute('href'))
    .find(h => /^\/[^/]+\/status\/\d+$/.test(h));
  const name = a.querySelector('[data-testid="User-Name"]')?.innerText.split('\n') || [];
  const textEl = a.querySelector('[data-testid="tweetText"]');
  const quote = a.querySelectorAll('[data-testid="tweetText"]')[1];
  return {
    url: href ? 'https://x.com' + href : null,
    posted_at: a.querySelector('time')?.getAttribute('datetime') || null,
    author_name: name[0] || null,
    author_handle: name.find(s => s.startsWith('@')) || null,
    text: textEl?.innerText || '',
    quoted_text: quote?.innerText || null,
    link_texts: [...a.querySelectorAll('[data-testid="tweetText"] a')]
      .map(x => x.innerText.replace(/\s+/g, ''))
      .filter(t => t.includes('.') && !t.startsWith('@') && !t.startsWith('#')),
    stats: a.querySelector('[role="group"]')?.getAttribute('aria-label') || '',
    is_ad: /\bAd\b/.test(a.innerText.split('\n').slice(0, 6).join(' ')),
  };
}).filter(p => p.url && !p.is_ad)
