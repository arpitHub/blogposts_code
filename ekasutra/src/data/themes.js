// Placeholder theme data. Real content for all 7 themes will be filled in later.
// Schema:
//   id, name, symbol, provocation,
//   ramayana: { content: [paragraphs], characters: [], moment: "" },
//   mahabharata: { content: [paragraphs], characters: [], moment: "" },
//   synthesis: "",
//   shloka: { sanskrit, translation, source }

export const themes = [
  {
    id: 'dharma',
    name: 'Dharma',
    symbol: '⚖️',
    provocation: 'What is right when every choice has a cost?',
    ramayana: {
      content: [
        'Placeholder paragraph 1 about how Dharma appears in the Ramayana.',
        'Placeholder paragraph 2 about how Dharma appears in the Ramayana.',
        'Placeholder paragraph 3 about how Dharma appears in the Ramayana.',
      ],
      characters: ['Rama', 'Bharata'],
      moment:
        "Rama's acceptance of exile despite being anointed king — choosing dharma over personal desire.",
    },
    mahabharata: {
      content: [
        'Placeholder paragraph 1 about how Dharma appears in the Mahabharata.',
        'Placeholder paragraph 2 about how Dharma appears in the Mahabharata.',
        'Placeholder paragraph 3 about how Dharma appears in the Mahabharata.',
      ],
      characters: ['Yudhishthira', 'Arjuna', 'Karna'],
      moment:
        "Arjuna's crisis on the battlefield — and Krishna's answer in the Bhagavad Gita.",
    },
    synthesis:
      'Placeholder synthesis paragraph drawing the common thread between both epics on this theme.',
    shloka: {
      sanskrit: 'धर्म एव हतो हन्ति धर्मो रक्षति रक्षितः',
      translation: 'Dharma destroyed destroys; dharma protected protects.',
      source: 'Manusmriti',
    },
  },
];
