import { useEffect, useMemo, useRef, useState } from 'react';

const SYSTEM_PROMPT =
  'You are Vyasa — the sage who authored the Mahabharata and is said to have inspired the Ramayana. You hold the wisdom of both epics. Answer questions with depth, compassion, and clarity, drawing from the stories, characters, philosophy, and verses of both epics. Speak in a calm, timeless voice. Keep responses to 3-5 sentences unless the depth of the question demands more. Never break character. /no_think';

const MODEL = 'qwen3:latest';
const STORAGE_KEY = 'ekasutra.ollamaUrl';
const ENV_URL = (import.meta.env.VITE_OLLAMA_URL || '').trim();

function stripThink(text) {
  if (!text) return '';
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*$/gi, '')
    .trim();
}

function normalizeUrl(url) {
  if (!url) return '';
  return url.replace(/\/+$/, '');
}

export default function AskSage() {
  const [open, setOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [ollamaUrl, setOllamaUrl] = useState(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      return normalizeUrl(saved || ENV_URL);
    } catch {
      return normalizeUrl(ENV_URL);
    }
  });
  const [urlDraft, setUrlDraft] = useState(ollamaUrl);

  const listRef = useRef(null);
  const inputRef = useRef(null);

  const effectiveUrl = useMemo(() => normalizeUrl(ollamaUrl), [ollamaUrl]);

  useEffect(() => {
    if (open && inputRef.current && !settingsOpen) {
      inputRef.current.focus();
    }
  }, [open, settingsOpen]);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, sending, open]);

  function saveUrl(next) {
    const cleaned = normalizeUrl(next);
    setOllamaUrl(cleaned);
    try {
      if (cleaned) {
        window.localStorage.setItem(STORAGE_KEY, cleaned);
      } else {
        window.localStorage.removeItem(STORAGE_KEY);
      }
    } catch {
      /* localStorage may be unavailable */
    }
  }

  async function sendMessage(e) {
    e?.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;

    if (!effectiveUrl) {
      setError(
        'No Ollama URL is configured. Open settings to set one.',
      );
      return;
    }

    const nextMessages = [...messages, { role: 'user', content: text }];
    setMessages(nextMessages);
    setDraft('');
    setSending(true);
    setError(null);

    try {
      const res = await fetch(`${effectiveUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: MODEL,
          stream: false,
          think: false,
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            ...nextMessages,
          ],
        }),
      });

      if (!res.ok) {
        throw new Error(`Ollama returned ${res.status}`);
      }

      const data = await res.json();
      const raw = data?.message?.content || '';
      const clean = stripThink(raw);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: clean || '…' },
      ]);
    } catch (err) {
      setError(
        'The sage is unreachable. Please check your Ollama connection.',
      );
    } finally {
      setSending(false);
    }
  }

  function clearConversation() {
    setMessages([]);
    setError(null);
  }

  return (
    <>
      <button
        type="button"
        className={`sage-fab ${open ? 'is-open' : ''}`}
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? 'Close Ask a Sage' : 'Ask a Sage'}
        aria-expanded={open}
      >
        <span className="sage-fab-glow" aria-hidden="true" />
        <span className="sage-fab-icon" aria-hidden="true">
          ✦
        </span>
      </button>

      <aside
        className={`sage-drawer ${open ? 'is-open' : ''}`}
        role="dialog"
        aria-label="Ask a Sage"
        aria-hidden={!open}
      >
        <header className="sage-header">
          <div className="sage-header-titles">
            <span className="eyebrow">Ask</span>
            <h2 className="sage-title">Vyasa</h2>
            <p className="sage-subtitle">Sage of both epics</p>
          </div>
          <div className="sage-header-actions">
            <button
              type="button"
              className="icon-button"
              onClick={() => {
                setUrlDraft(ollamaUrl);
                setSettingsOpen((v) => !v);
              }}
              aria-label="Settings"
              title="Settings"
            >
              <span aria-hidden="true">⚙</span>
            </button>
            <button
              type="button"
              className="icon-button"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>
        </header>

        {settingsOpen && (
          <div className="sage-settings">
            <label className="sage-settings-label" htmlFor="ollama-url">
              Ollama base URL
            </label>
            <div className="sage-settings-row">
              <input
                id="ollama-url"
                className="sage-input"
                type="url"
                value={urlDraft}
                placeholder="http://192.168.1.x:11434"
                onChange={(e) => setUrlDraft(e.target.value)}
                spellCheck={false}
              />
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  saveUrl(urlDraft);
                  setSettingsOpen(false);
                }}
              >
                Save
              </button>
            </div>
            <p className="sage-hint">
              Model: <code>{MODEL}</code>. Saved to this browser only.
            </p>
          </div>
        )}

        <div className="sage-messages" ref={listRef}>
          {messages.length === 0 && !sending && !error && (
            <p className="sage-placeholder">
              Ask about a character, a moment, a shloka, or the wisdom
              beneath the story. Vyasa will answer.
            </p>
          )}

          {messages.map((m, i) => (
            <div
              key={i}
              className={`sage-msg sage-msg--${m.role}`}
            >
              <span className="sage-msg-role">
                {m.role === 'user' ? 'You' : 'Vyasa'}
              </span>
              <p className="sage-msg-body">{m.content}</p>
            </div>
          ))}

          {sending && (
            <div className="sage-msg sage-msg--assistant sage-msg--pending">
              <span className="sage-msg-role">Vyasa</span>
              <p className="sage-msg-body sage-thinking">
                Vyasa is reflecting
                <span className="sage-dots" aria-hidden="true">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </p>
            </div>
          )}

          {error && <p className="sage-error">{error}</p>}
        </div>

        <form className="sage-composer" onSubmit={sendMessage}>
          <textarea
            ref={inputRef}
            className="sage-textarea"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask Vyasa…"
            rows={2}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
              }
            }}
          />
          <div className="sage-composer-actions">
            <button
              type="button"
              className="text-button sage-clear"
              onClick={clearConversation}
              disabled={messages.length === 0 && !error}
            >
              Clear
            </button>
            <button
              type="submit"
              className="sage-send"
              disabled={!draft.trim() || sending}
            >
              {sending ? 'Sending…' : 'Ask'}
            </button>
          </div>
        </form>
      </aside>

      {open && (
        <button
          type="button"
          className="sage-scrim"
          aria-label="Close drawer"
          onClick={() => setOpen(false)}
        />
      )}
    </>
  );
}
