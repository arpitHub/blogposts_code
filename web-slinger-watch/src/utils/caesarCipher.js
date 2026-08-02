// Caesar shift over A-Z / a-z only. Spaces, digits, and punctuation pass
// through untouched so the shape of a message stays readable.

const ALPHABET_SIZE = 26;
const UPPER_A = 65;
const LOWER_A = 97;

function shiftChar(char, shift) {
  const code = char.charCodeAt(0);

  if (code >= UPPER_A && code < UPPER_A + ALPHABET_SIZE) {
    return String.fromCharCode(
      ((code - UPPER_A + shift) % ALPHABET_SIZE) + UPPER_A
    );
  }
  if (code >= LOWER_A && code < LOWER_A + ALPHABET_SIZE) {
    return String.fromCharCode(
      ((code - LOWER_A + shift) % ALPHABET_SIZE) + LOWER_A
    );
  }
  return char;
}

function normalizeShift(shift) {
  return ((Math.trunc(shift) % ALPHABET_SIZE) + ALPHABET_SIZE) % ALPHABET_SIZE;
}

export function encode(text, shift) {
  const normalized = normalizeShift(shift);
  return String(text)
    .split('')
    .map((char) => shiftChar(char, normalized))
    .join('');
}

export function decode(text, shift) {
  return encode(text, -normalizeShift(shift));
}

export { ALPHABET_SIZE };
