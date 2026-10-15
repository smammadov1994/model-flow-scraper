import { useEffect, useRef, useState } from 'react'
import type { AskOption, ChatMessage, Deployment } from '../types'

const EXAMPLES = [
  'A voice assistant that runs on my iPhone',
  'Transcribe meetings and summarize them on my MacBook',
  'Generate product photos on an 8GB GPU',
  'Private document Q&A on my laptop',
]

const PROGRESS = ['Planning steps…', 'Finding models…', 'Checking memory…']

interface Props {
  messages: ChatMessage[]
  pending: boolean
  onSend: (text: string, deployment?: Deployment) => void
  onNew: () => void
  canReset: boolean
  target: Deployment | null
  onTarget: (t: Deployment | null) => void
  hardwareLabel: string | null
}

const TARGETS: { value: Deployment | null; label: string }[] = [
  { value: null, label: 'Ask me' },
  { value: 'local', label: 'This computer' },
  { value: 'phone', label: 'Phone' },
  { value: 'cloud', label: 'Cloud' },
]

function ProgressRow() {
  const [i, setI] = useState(0)
  useEffect(() => {
    const t = window.setInterval(() => setI((x) => (x + 1) % PROGRESS.length), 2400)
    return () => window.clearInterval(t)
  }, [])
  return (
    <div className="msg msg-assistant progress" role="status" aria-live="polite">
      <span className="dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      <span>{PROGRESS[i]}</span>
    </div>
  )
}

export default function ChatPanel({ messages, pending, onSend, onNew, canReset, target, onTarget, hardwareLabel }: Props) {
  const [text, setText] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, pending])

  const submit = (value: string) => {
    const v = value.trim()
    if (!v || pending) return
    onSend(v)
    setText('')
  }

  return (
    <section className="chat" aria-label="Chat">
      <header className="chat-head">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="5.5" cy="12" r="2.2" />
              <circle cx="18.5" cy="6" r="2.2" />
              <circle cx="18.5" cy="18" r="2.2" />
              <path d="M7.7 12h3.3c2.5 0 3-4 5.3-5.2M11 12c2.5 0 3 4 5.3 5.2" />
            </svg>
          </span>
          <span>Model Studio</span>
        </div>
        {canReset ? (
          <button type="button" className="btn btn-ghost btn-small" onClick={onNew} disabled={pending}>
            New pipeline
          </button>
        ) : null}
      </header>

      <div className="chat-list" ref={listRef}>
        {messages.length === 0 && !pending ? (
          <div className="chat-empty">
            <p className="chat-hello">What do you want to build?</p>
            <p className="muted">Say it in plain words. Mention the device if you have one in mind.</p>
            <div className="example-chips">
              {EXAMPLES.map((ex) => (
                <button key={ex} type="button" className="example" onClick={() => submit(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {messages.map((m, i) => {
          const answered = messages.slice(i + 1).some((x) => x.role === 'user')
          return (
            <div key={i} className={`msg msg-${m.role}${m.ask ? ' msg-ask' : ''}`}>
              {m.role === 'error' ? <span className="sr-only">Error: </span> : null}
              {m.text}
              {m.ask ? (
                <div className="ask-options" role="group" aria-label={m.ask.question}>
                  {m.ask.options.map((o: AskOption) => (
                    <button
                      key={o.value}
                      type="button"
                      className="ask-btn"
                      disabled={answered || pending}
                      onClick={() => onSend(o.label, o.value)}
                    >
                      <span className="ask-label">{o.label}</span>
                      {o.detail ? <span className="ask-detail">{o.detail}</span> : null}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          )
        })}
        {pending ? <ProgressRow /> : null}
      </div>

      <div className="target-row">
        <span className="target-title" id="target-title">
          Run on:
        </span>
        <div className="seg" role="radiogroup" aria-labelledby="target-title">
          {TARGETS.map((t) => {
            const on = target === t.value
            const hw = t.value === 'local' && hardwareLabel ? hardwareLabel : undefined
            return (
              <button
                key={t.label}
                type="button"
                role="radio"
                aria-checked={on}
                className={`seg-btn${on ? ' is-on' : ''}`}
                title={hw}
                onClick={() => onTarget(t.value)}
              >
                {t.label}
              </button>
            )
          })}
        </div>
        {hardwareLabel && target === 'local' ? <span className="target-hw">{hardwareLabel}</span> : null}
      </div>

      <form
        className="chat-form"
        onSubmit={(e) => {
          e.preventDefault()
          submit(text)
        }}
      >
        <label htmlFor="chat-input" className="sr-only">
          Describe what you want to build
        </label>
        <textarea
          id="chat-input"
          rows={2}
          value={text}
          placeholder="Describe what you want to build…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit(text)
            }
          }}
        />
        <button type="submit" className="btn btn-primary" disabled={pending || !text.trim()}>
          Send
        </button>
      </form>
    </section>
  )
}
