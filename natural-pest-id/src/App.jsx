import { useState, useRef } from 'react';

const SYSTEM_PROMPT = `You are a knowledgeable entomologist and garden ecologist. When given a description or image of an insect or garden creature, respond ONLY in valid JSON with this structure:
{
  "name": "Common name (Scientific name)",
  "verdict": "friend" | "foe" | "neutral",
  "summary": "One sentence description of what this creature is.",
  "what_it_does": "What it does in the garden — damage caused or benefit provided.",
  "what_to_do": "Practical advice: leave it, encourage it, or how to deal with it naturally.",
  "confidence": "high" | "medium" | "low"
}
If you cannot identify the creature, return the JSON with name as "Unknown" and verdict as "neutral". Do not return anything other than the JSON object.`;

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function extractJson(text) {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const start = trimmed.indexOf('{');
    const end = trimmed.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
      return JSON.parse(trimmed.slice(start, end + 1));
    }
    throw new Error('No JSON object found in response');
  }
}

export default function App() {
  const [description, setDescription] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result);
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleReset = () => {
    setDescription('');
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleIdentify = async () => {
    setError('');
    setResult(null);

    if (!description.trim() && !imageFile) {
      setError('Please describe the creature or upload a photo (or both).');
      return;
    }

    const baseUrl = (
      import.meta.env.VITE_OLLAMA_URL || 'http://localhost:11434'
    ).replace(/\/$/, '');
    const model = import.meta.env.VITE_OLLAMA_MODEL || 'gemma3:4b';

    setLoading(true);

    try {
      const userMessage = {
        role: 'user',
        content: description.trim()
          ? description.trim()
          : 'Please identify the creature in this image.',
      };
      if (imageFile) {
        const base64 = await fileToBase64(imageFile);
        userMessage.images = [base64];
      }

      const response = await fetch(`${baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          stream: false,
          format: 'json',
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            userMessage,
          ],
          options: { num_predict: 1000 },
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Ollama error (${response.status}): ${errText}`);
      }

      const data = await response.json();
      const text = data.message?.content;
      if (!text) {
        throw new Error('No text response from the model.');
      }

      const parsed = extractJson(text);
      setResult(parsed);
    } catch (err) {
      console.error(err);
      setError(
        "Something went wrong while identifying the creature. Make sure Ollama is running locally and the model is installed, then try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const verdictClass = (v) => {
    if (v === 'friend') return 'badge badge-friend';
    if (v === 'foe') return 'badge badge-foe';
    return 'badge badge-neutral';
  };

  const verdictLabel = (v) => {
    if (v === 'friend') return 'Friend';
    if (v === 'foe') return 'Foe';
    return 'Neutral';
  };

  return (
    <div className="page">
      <header className="header">
        <h1>Natural Pest ID</h1>
        <p className="tagline">
          Friend or foe? Identify the creatures in your garden.
        </p>
      </header>

      <main className="card">
        {!result && (
          <>
            <label className="field-label" htmlFor="description">
              Describe what you saw
            </label>
            <textarea
              id="description"
              className="textarea"
              placeholder="e.g. small green bug with wings on my rose leaves"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={4}
              disabled={loading}
            />

            <label className="field-label">Add a photo (optional)</label>
            <div className="upload-row">
              <label className="upload-button">
                {imageFile ? 'Change photo' : 'Choose photo'}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  disabled={loading}
                  hidden
                />
              </label>
              {imagePreview && (
                <button
                  type="button"
                  className="link-button"
                  onClick={handleRemoveImage}
                  disabled={loading}
                >
                  Remove
                </button>
              )}
            </div>

            {imagePreview && (
              <div className="preview-wrap">
                <img src={imagePreview} alt="Uploaded preview" className="preview" />
              </div>
            )}

            {error && <div className="error">{error}</div>}

            <button
              type="button"
              className="primary-button"
              onClick={handleIdentify}
              disabled={loading}
            >
              {loading ? (
                <span className="examining">Examining<span className="dots">...</span></span>
              ) : (
                'Identify'
              )}
            </button>
          </>
        )}

        {result && (
          <div className="result">
            <div className="result-header">
              <h2 className="creature-name">{result.name || 'Unknown'}</h2>
              <span className={verdictClass(result.verdict)}>
                {verdictLabel(result.verdict)}
              </span>
            </div>

            {result.summary && (
              <section className="result-section">
                <h3>Summary</h3>
                <p>{result.summary}</p>
              </section>
            )}

            {result.what_it_does && (
              <section className="result-section">
                <h3>What it does</h3>
                <p>{result.what_it_does}</p>
              </section>
            )}

            {result.what_to_do && (
              <section className="result-section">
                <h3>What to do</h3>
                <p>{result.what_to_do}</p>
              </section>
            )}

            {result.confidence && (
              <div className="confidence">
                Confidence:{' '}
                <span className={`confidence-${result.confidence}`}>
                  {result.confidence}
                </span>
              </div>
            )}

            <button type="button" className="secondary-button" onClick={handleReset}>
              Identify Another
            </button>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by Claude — leaves, soil, and curious eyes.</p>
      </footer>
    </div>
  );
}
