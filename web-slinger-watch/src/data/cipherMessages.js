// Intercepted "chatter" for the Decode the Signal mini-game. Each message
// names the borough the tip points at, so a solved case can spawn a bonus
// sighting there. Kept to 3-5 short words for small-screen readability.

export const CIPHER_MESSAGES = [
  { borough: 'queens', plaintext: 'NEXT STRIKE WAREHOUSE DISTRICT' },
  { borough: 'queens', plaintext: 'MEET UNDER THE TRESTLE' },
  { borough: 'queens', plaintext: 'VAN PARKED NEAR STADIUM' },
  { borough: 'queens', plaintext: 'CARGO MOVES AFTER MIDNIGHT' },
  { borough: 'queens', plaintext: 'WATCH THE NIGHT MARKET' },
  { borough: 'brooklyn', plaintext: 'CROSSING THE BIG BRIDGE' },
  { borough: 'brooklyn', plaintext: 'ROOFTOP DEAL AT DUSK' },
  { borough: 'brooklyn', plaintext: 'FOLLOW THE BOARDWALK LIGHTS' },
  { borough: 'brooklyn', plaintext: 'LOADING DOCK GOES DARK' },
  { borough: 'brooklyn', plaintext: 'SIGNAL FROM THE PROMENADE' },
  { borough: 'manhattan', plaintext: 'EYES ON MIDTOWN SCAFFOLD' },
  { borough: 'manhattan', plaintext: 'THEY MOVE AT MARQUEE' },
  { borough: 'manhattan', plaintext: 'WATER TOWER IS COMPROMISED' },
  { borough: 'manhattan', plaintext: 'BODEGA CORNER GOES QUIET' },
  { borough: 'manhattan', plaintext: 'ELEVATED PARK EXIT SOUTH' },
];

// Static sample used by the instructions modal's non-functional demo dial.
export const EXAMPLE_CIPHER = {
  plaintext: 'ROOFTOP MEET',
  shift: 4,
};

export const CIPHER_LABELS = {
  panelTitle: 'Decode the Signal',
  intercepted: 'Intercepted transmission',
  decodedAs: 'Reads as',
  lockIn: 'Lock In',
  solved: 'Case Solved',
  lost: 'Signal Lost',
  shift: 'Shift',
};

export function randomCipherMessage() {
  return CIPHER_MESSAGES[Math.floor(Math.random() * CIPHER_MESSAGES.length)];
}

// Shift of 0 would leave the ciphertext readable, so puzzles use 1-25.
export function randomShift() {
  return 1 + Math.floor(Math.random() * 25);
}
