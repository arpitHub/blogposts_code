# blogposts_code

Companion code for blog posts on [iarpit.com](https://www.iarpit.com). Each top-level
folder is a self-contained project — most are interactive single-page web apps
(Vite + React) that can be run and deployed independently.

## Projects

| Project | What it is |
| --- | --- |
| [`lens-craft`](lens-craft/) | **Lens Craft** — an interactive photography school in the "explorable explanations" style. 14 modules (exposure triangle, depth of field, shutter speed, ISO, composition, light, white balance, focal length, histogram, autofocus, RAW vs JPEG, editing, genre guides, sensor sizes), each with a live canvas widget you drag instead of a wall of text. |
| [`court-craft`](court-craft/) | **Court Craft** — tennis education, same explorable-explanations approach: every concept ships with a widget you can drag, scrub, or play with. |
| [`how-llms-work`](how-llms-work/) | Interactive explainer of how large language models work — tokens, embeddings, attention, the transformer stack, and next-token prediction — for a mixed audience. |
| [`halflife`](halflife/) | **HalfLife** — interactive explainer of radiometric dating and deep time: how counting atoms tells us the Earth is 4.55 billion years old. |
| [`ekasutra`](ekasutra/) | **Ekasutra (One Thread)** — an immersive exploration of the universal themes connecting the Ramayana and the Mahabharata. |
| [`chidiya-udd`](chidiya-udd/) | **Chidiya Udd (Bird, Fly!)** — nostalgic pass-and-play reflex game for 2–8 players on one shared screen: raise a hand only when the called-out thing can fly. |
| [`natural-pest-id`](natural-pest-id/) | Garden insect identifier that labels each creature friend, foe, or neutral. Runs fully in the browser against a local Ollama instance — no cloud API or key. |
| [`unit-testing-datascience`](unit-testing-datascience/) | Python example (pytest) showing how to unit-test data-science code — preprocessing and model logic tested against the Iris dataset. |

## Running a web project

All web apps follow the same pattern:

```bash
cd <project>
npm install
npm run dev
```

Most include a `vercel.json` and deploy directly to Vercel with the project's
folder set as the root directory.

## Running the Python project

```bash
cd unit-testing-datascience
pip install -r requirements.txt
pytest
```
