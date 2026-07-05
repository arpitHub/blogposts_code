import { useEffect, useRef, useState } from 'react';

const LISTEN_SECONDS = 8;

const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
];

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  return MIME_CANDIDATES.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

function extensionForMime(mimeType) {
  if (mimeType.includes('mp4')) return 'mp4';
  if (mimeType.includes('ogg')) return 'ogg';
  return 'webm';
}

async function recognize(blob, mimeType) {
  const token = import.meta.env.VITE_AUDD_API_TOKEN;
  if (!token) {
    throw new Error(
      'Missing AudD API token. Set VITE_AUDD_API_TOKEN in your .env file.'
    );
  }

  const formData = new FormData();
  formData.append('file', blob, `sample.${extensionForMime(mimeType)}`);
  formData.append('api_token', token);
  formData.append('return', 'spotify');

  const response = await fetch('https://api.audd.io/', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`AudD error (${response.status})`);
  }

  const data = await response.json();

  if (data.status !== 'success') {
    throw new Error(data.error?.error_message || 'Recognition failed.');
  }

  if (!data.result) {
    return null;
  }

  const { result } = data;
  const spotify = result.spotify;

  return {
    title: result.title,
    artist: result.artist,
    album: result.album,
    releaseDate: result.release_date,
    artwork: spotify?.album?.images?.[0]?.url,
    spotifyUrl: spotify?.external_urls?.spotify,
    songLink: result.song_link,
  };
}

export default function App() {
  const [status, setStatus] = useState('idle'); // idle | listening | processing | result | error
  const [secondsLeft, setSecondsLeft] = useState(LISTEN_SECONDS);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const cancelledRef = useRef(false);

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  useEffect(() => stopStream, []);

  const handleListen = async () => {
    setError('');
    setResult(null);
    cancelledRef.current = false;

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setStatus('error');
      setError(
        'Microphone access was denied or is unavailable. Please allow microphone access and try again.'
      );
      return;
    }

    streamRef.current = stream;
    const mimeType = pickMimeType();
    const recorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    chunksRef.current = [];
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      stopStream();
      clearInterval(timerRef.current);

      if (cancelledRef.current) {
        setStatus('idle');
        return;
      }

      setStatus('processing');
      const blob = new Blob(chunksRef.current, {
        type: recorder.mimeType || 'audio/webm',
      });

      try {
        const match = await recognize(blob, recorder.mimeType || 'audio/webm');
        setResult(match);
        setStatus('result');
      } catch (err) {
        console.error(err);
        setError(err.message || 'Something went wrong while identifying the song.');
        setStatus('error');
      }
    };

    recorder.start();
    setStatus('listening');
    setSecondsLeft(LISTEN_SECONDS);

    timerRef.current = setInterval(() => {
      setSecondsLeft((prev) => {
        if (prev <= 1) {
          recorder.stop();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleCancel = () => {
    cancelledRef.current = true;
    clearInterval(timerRef.current);
    mediaRecorderRef.current?.stop();
  };

  const handleReset = () => {
    setStatus('idle');
    setResult(null);
    setError('');
  };

  return (
    <div className="page">
      <header className="header">
        <h1>Song Recognition</h1>
        <p className="tagline">Play a song nearby and we'll name it, then send you to Spotify.</p>
      </header>

      <main className="card">
        {status === 'idle' && (
          <div className="listen-panel">
            <button type="button" className="listen-button" onClick={handleListen}>
              <MicIcon />
            </button>
            <p className="hint">Tap to start listening</p>
          </div>
        )}

        {status === 'listening' && (
          <div className="listen-panel">
            <button type="button" className="listen-button listening" onClick={handleCancel}>
              <MicIcon />
              <span className="pulse-ring" />
              <span className="pulse-ring delay" />
            </button>
            <p className="hint">Listening… {secondsLeft}s</p>
            <button type="button" className="link-button" onClick={handleCancel}>
              Cancel
            </button>
          </div>
        )}

        {status === 'processing' && (
          <div className="listen-panel">
            <span className="examining">
              Identifying<span className="dots">...</span>
            </span>
          </div>
        )}

        {status === 'error' && (
          <div className="listen-panel">
            <div className="error">{error}</div>
            <button type="button" className="primary-button" onClick={handleReset}>
              Try Again
            </button>
          </div>
        )}

        {status === 'result' && !result && (
          <div className="result">
            <p className="no-match">
              We couldn't find a match. Try getting closer to the speaker and reducing
              background noise.
            </p>
            <button type="button" className="primary-button" onClick={handleReset}>
              Try Again
            </button>
          </div>
        )}

        {status === 'result' && result && (
          <div className="result">
            <div className="track">
              {result.artwork ? (
                <img className="artwork" src={result.artwork} alt={`${result.album || result.title} cover`} />
              ) : (
                <div className="artwork artwork-placeholder">
                  <NoteIcon />
                </div>
              )}
              <div className="track-info">
                <h2 className="track-title">{result.title}</h2>
                <p className="track-artist">{result.artist}</p>
                {result.album && (
                  <p className="track-album">
                    {result.album}
                    {result.releaseDate ? ` · ${result.releaseDate.slice(0, 4)}` : ''}
                  </p>
                )}
              </div>
            </div>

            {(result.spotifyUrl || result.songLink) && (
              <a
                className="spotify-button"
                href={result.spotifyUrl || result.songLink}
                target="_blank"
                rel="noopener noreferrer"
              >
                <SpotifyIcon />
                Open in Spotify
              </a>
            )}

            <button type="button" className="secondary-button" onClick={handleReset}>
              Listen Again
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Music recognition powered by AudD.</p>
      </footer>
    </div>
  );
}

function MicIcon() {
  return (
    <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor">
      <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z" />
      <path d="M19 11a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V20H9a1 1 0 0 0 0 2h6a1 1 0 0 0 0-2h-2v-2.08A7 7 0 0 0 19 11z" />
    </svg>
  );
}

function NoteIcon() {
  return (
    <svg viewBox="0 0 24 24" width="28" height="28" fill="currentColor">
      <path d="M9 18V5l12-2v13" />
      <circle cx="6" cy="18" r="3" />
      <circle cx="18" cy="16" r="3" />
    </svg>
  );
}

function SpotifyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
      <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm4.59 14.44a.62.62 0 0 1-.86.21c-2.36-1.44-5.33-1.77-8.83-.97a.62.62 0 1 1-.28-1.21c3.83-.88 7.12-.5 9.76 1.11.3.18.4.57.21.86zm1.22-2.72a.78.78 0 0 1-1.07.26c-2.7-1.66-6.82-2.14-10.02-1.17a.78.78 0 1 1-.45-1.49c3.65-1.11 8.19-.57 11.28 1.33.37.23.49.72.26 1.07zm.11-2.83C14.9 9.03 9.2 8.84 5.9 9.86a.93.93 0 1 1-.55-1.78c3.8-1.17 10.1-.94 14.09 1.44a.93.93 0 1 1-.97 1.59z" />
    </svg>
  );
}
