export const WORDS = [
  { word: 'Chidiya', translation: 'Bird', flying: true },
  { word: 'Maina', translation: 'Myna', flying: true },
  { word: 'Tota', translation: 'Parrot', flying: true },
  { word: 'Kauwa', translation: 'Crow', flying: true },
  { word: 'Titli', translation: 'Butterfly', flying: true },
  { word: 'Machhar', translation: 'Mosquito', flying: true },
  { word: 'Cheel', translation: 'Kite (bird)', flying: true },
  { word: 'Baaz', translation: 'Hawk', flying: true },
  { word: 'Ullu', translation: 'Owl', flying: true },
  { word: 'Hawai Jahaz', translation: 'Airplane', flying: true },
  { word: 'Madhumakhi', translation: 'Bee', flying: true },
  { word: 'Bagula', translation: 'Heron', flying: true },
  { word: 'Billi', translation: 'Cat', flying: false },
  { word: 'Kutta', translation: 'Dog', flying: false },
  { word: 'Hathi', translation: 'Elephant', flying: false },
  { word: 'Kursi', translation: 'Chair', flying: false },
  { word: 'Mez', translation: 'Table', flying: false },
  { word: 'Gadha', translation: 'Donkey', flying: false },
  { word: 'Bakri', translation: 'Goat', flying: false },
  { word: 'Machhli', translation: 'Fish', flying: false },
  { word: 'Saanp', translation: 'Snake', flying: false },
  { word: 'Ghoda', translation: 'Horse', flying: false },
  { word: 'Gaay', translation: 'Cow', flying: false },
  { word: 'Cycle', translation: 'Bicycle', flying: false },
];

const FLYING_WORDS = WORDS.filter((w) => w.flying);
const GROUND_WORDS = WORDS.filter((w) => !w.flying);

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Shuffle-bag picker, one bag per category. Each call flips a coin to pick a
 * category (roughly balances flying vs non-flying over time), then pops the
 * category's shuffled bag, refilling it from the full category list when
 * empty. Swaps the top two entries on a repeat to avoid an immediate
 * back-to-back repeat across a bag-refill boundary.
 */
export function createWordPicker() {
  let flyingBag = shuffle(FLYING_WORDS);
  let groundBag = shuffle(GROUND_WORDS);
  let lastWord = null;

  return function pickNext() {
    const useFlying = Math.random() < 0.5;
    if (useFlying && flyingBag.length === 0) flyingBag = shuffle(FLYING_WORDS);
    if (!useFlying && groundBag.length === 0) groundBag = shuffle(GROUND_WORDS);

    const bag = useFlying ? flyingBag : groundBag;
    if (lastWord && bag[bag.length - 1].word === lastWord.word && bag.length > 1) {
      [bag[bag.length - 1], bag[bag.length - 2]] = [bag[bag.length - 2], bag[bag.length - 1]];
    }
    const candidate = bag.pop();
    lastWord = candidate;
    return candidate;
  };
}

export function getIntervalMs(callsMade) {
  return Math.max(850, 2400 - 150 * Math.floor(callsMade / 5));
}

export function getLevel(callsMade) {
  return Math.floor(callsMade / 5) + 1;
}
