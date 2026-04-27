import { useEffect, useMemo, useRef, useState } from 'react';

const SYSTEM_PROMPT =
  'You are "Base Camp" — a calm, knowledgeable field assistant communicating over a walkie-talkie. Keep every response to 1-3 sentences, direct and clear. Occasionally use light radio lingo naturally (e.g. "Copy that", "Understood", "Over") but don\'t overdo it. Never break character. /no_think';

const MODEL = 'qwen3:latest';
const STORAGE_KEY = 'walkie-talkie:ollama-url';
const ENV_OLLAMA_URL = import.meta.env.VITE_OLLAMA_URL || 'http://localhost:11434';

const STATES = {
  READY: 'Ready',
  LISTENING: 'Listening…',
  CONTACTING: 'Contacting Base Camp…',
  TRANSMITTING: 'Base Camp transmitting…',
  ERROR: 'Error',
};

function stripThinkBlocks(text) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/<think>[\s\S]*$/i, '')
    .trim();
}

function getRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function pickVoice(voices) {
  const en = voices.filter((v) => /^en[-_]?/i.test(v.lang));
  const preferredNames = [
    'Daniel',
    'Google UK English Male',
    'Microsoft Guy',
    'Microsoft David',
    'Alex',
    'Fred',
  ];
  for (const name of preferredNames) {
    const match = en.find((v) => v.name.includes(name));
    if (match) return match;
  }
  const male = en.find((v) => /male/i.test(v.name));
  if (male) return male;
  return en[0] || voices[0] || null;
}

export default function App() {
  const [ollamaUrl, setOllamaUrl] = useState(
    () => localStorage.getItem(STORAGE_KEY) || ENV_OLLAMA_URL
  );
  const [showSettings, setShowSettings] = useState(false);
  const [draftUrl, setDraftUrl] = useState(ollamaUrl);

  const [status, setStatus] = useState(STATES.READY);
  const [statusDetail, setStatusDetail] = useState('');
  const [messages, setMessages] = useState([]);
  const [supported, setSupported] = useState(true);
  const [micError, setMicError] = useState('');

  const recognitionRef = useRef(null);
  const transcriptRef = useRef('');
  const messagesRef = useRef(messages);
  const isBusyRef = useRef(false);
  const transcriptLogRef = useRef(null);
  const voiceRef = useRef(null);
  const ollamaUrlRef = useRef(ollamaUrl);
  const micErrorRef = useRef(micError);
  const handleUtteranceRef = useRef(() => {});

  useEffect(() => {
    ollamaUrlRef.current = ollamaUrl;
  }, [ollamaUrl]);

  useEffect(() => {
    micErrorRef.current = micError;
  }, [micError]);

  useEffect(() => {
    messagesRef.current = messages;
    if (transcriptLogRef.current) {
      transcriptLogRef.current.scrollTop = transcriptLogRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      setSupported(false);
      return;
    }
    const recog = new Ctor();
    recog.continuous = false;
    recog.interimResults = false;
    recog.lang = 'en-US';
    recog.maxAlternatives = 1;

    recog.onresult = (event) => {
      const result = event.results[0];
      if (result && result[0]) {
        transcriptRef.current = result[0].transcript.trim();
      }
    };

    recog.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setMicError('Microphone permission denied. Please allow mic access and reload.');
      } else if (event.error === 'no-speech') {
        setStatusDetail('Nothing heard — try again.');
      } else if (event.error !== 'aborted') {
        setStatusDetail(`Mic error: ${event.error}`);
      }
    };

    recog.onend = () => {
      const text = transcriptRef.current;
      transcriptRef.current = '';
      if (!text) {
        if (!micErrorRef.current) {
          setStatus(STATES.READY);
          setStatusDetail((d) => d || 'Nothing heard — try again.');
        }
        isBusyRef.current = false;
        return;
      }
      handleUtteranceRef.current(text);
    };

    recognitionRef.current = recog;
    return () => {
      try {
        recog.abort();
      } catch {
        /* noop */
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!('speechSynthesis' in window)) return;
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      voiceRef.current = pickVoice(voices);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  async function callOllama(history) {
    const url = `${ollamaUrlRef.current.replace(/\/$/, '')}/api/chat`;
    const body = {
      model: MODEL,
      stream: false,
      think: false,
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...history],
    };
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`Ollama responded ${res.status}`);
    }
    const data = await res.json();
    const raw = data?.message?.content ?? '';
    return stripThinkBlocks(raw);
  }

  function speak(text, onDone) {
    if (!('speechSynthesis' in window) || !text) {
      onDone?.();
      return;
    }
    const utter = new SpeechSynthesisUtterance(text);
    if (voiceRef.current) utter.voice = voiceRef.current;
    utter.rate = 0.95;
    utter.pitch = 1.0;
    utter.lang = voiceRef.current?.lang || 'en-US';
    utter.onend = () => onDone?.();
    utter.onerror = () => onDone?.();
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  }

  handleUtteranceRef.current = handleUserUtterance;

  async function handleUserUtterance(text) {
    setStatusDetail('');
    const nextHistory = [...messagesRef.current, { role: 'user', content: text }];
    setMessages(nextHistory);
    setStatus(STATES.CONTACTING);
    try {
      const reply = await callOllama(nextHistory);
      const cleaned = reply || '…';
      const withReply = [...nextHistory, { role: 'assistant', content: cleaned }];
      setMessages(withReply);
      setStatus(STATES.TRANSMITTING);
      speak(cleaned, () => {
        setStatus(STATES.READY);
        isBusyRef.current = false;
      });
    } catch (err) {
      console.error(err);
      setStatus(STATES.ERROR);
      setStatusDetail('Could not reach Base Camp. Check your Ollama URL.');
      isBusyRef.current = false;
    }
  }

  function startListening() {
    if (!supported || micError) return;
    if (isBusyRef.current) return;
    if (status === STATES.CONTACTING || status === STATES.TRANSMITTING) return;

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    isBusyRef.current = true;
    transcriptRef.current = '';
    setStatusDetail('');
    setStatus(STATES.LISTENING);
    try {
      recognitionRef.current?.start();
    } catch {
      // already started — ignore
    }
  }

  function stopListening() {
    if (status !== STATES.LISTENING) return;
    try {
      recognitionRef.current?.stop();
    } catch {
      /* noop */
    }
  }

  function saveSettings() {
    const trimmed = draftUrl.trim().replace(/\/$/, '');
    if (!trimmed) return;
    localStorage.setItem(STORAGE_KEY, trimmed);
    setOllamaUrl(trimmed);
    setShowSettings(false);
  }

  function clearTranscript() {
    setMessages([]);
    setStatusDetail('');
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  }

  const buttonClass = useMemo(() => {
    const base = 'ptt';
    if (status === STATES.LISTENING) return `${base} ptt--listening`;
    if (status === STATES.CONTACTING) return `${base} ptt--processing`;
    if (status === STATES.TRANSMITTING) return `${base} ptt--playing`;
    return base;
  }, [status]);

  const buttonLabel = useMemo(() => {
    if (status === STATES.LISTENING) return 'LISTENING';
    if (status === STATES.CONTACTING) return '• • •';
    if (status === STATES.TRANSMITTING) return 'TRANSMIT';
    return 'HOLD TO TALK';
  }, [status]);

  const disabled =
    !supported ||
    !!micError ||
    status === STATES.CONTACTING ||
    status === STATES.TRANSMITTING;

  return (
    <div className="app">
      <div className="noise" aria-hidden="true" />
      <div className="frame">
        <header className="header">
          <div className="brand">
            <span className="led" />
            <span className="brand-text">BASE CAMP · CH 1</span>
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={() => {
              setDraftUrl(ollamaUrl);
              setShowSettings((s) => !s);
            }}
            aria-label="Settings"
            title="Settings"
          >
            ⚙
          </button>
        </header>

        {showSettings && (
          <div className="settings">
            <label htmlFor="ollama-url">Ollama URL</label>
            <input
              id="ollama-url"
              type="text"
              value={draftUrl}
              onChange={(e) => setDraftUrl(e.target.value)}
              placeholder="http://192.168.1.x:11434"
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
            <div className="settings-actions">
              <button type="button" onClick={saveSettings}>
                Save
              </button>
              <button type="button" onClick={() => setShowSettings(false)}>
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="status-bar">
          <span className="status-label">{status}</span>
          {statusDetail && <span className="status-detail"> · {statusDetail}</span>}
        </div>

        <div className="transcript" ref={transcriptLogRef}>
          {messages.length === 0 && (
            <div className="empty">— No transmissions yet —</div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={`line ${m.role === 'user' ? 'line--user' : 'line--base'}`}
            >
              {m.role === 'assistant' && (
                <span className="prefix">📡 Base Camp:</span>
              )}
              <span className="content">{m.content}</span>
            </div>
          ))}
        </div>

        <div className="ptt-wrap">
          <button
            type="button"
            className={buttonClass}
            disabled={disabled}
            onPointerDown={(e) => {
              e.preventDefault();
              e.currentTarget.setPointerCapture?.(e.pointerId);
              startListening();
            }}
            onPointerUp={stopListening}
            onPointerLeave={stopListening}
            onPointerCancel={stopListening}
            onContextMenu={(e) => e.preventDefault()}
            aria-label="Push to talk"
          >
            <span className="ptt-label">{buttonLabel}</span>
          </button>
        </div>

        <footer className="footer">
          <span className="footer-url" title={ollamaUrl}>
            {ollamaUrl}
          </span>
          {messages.length > 0 && (
            <button type="button" className="link-btn" onClick={clearTranscript}>
              clear log
            </button>
          )}
        </footer>

        {!supported && (
          <div className="banner">
            Speech recognition not supported in this browser. Use Chrome or Safari.
          </div>
        )}
        {micError && <div className="banner">{micError}</div>}
      </div>
    </div>
  );
}
