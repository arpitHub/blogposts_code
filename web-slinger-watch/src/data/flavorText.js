// Templated sighting flavor lines. `{landmark}` slots are filled in by the
// sightings generator with a landmark drawn from the matching borough.

export const LANDMARKS = {
  manhattan: [
    'a rooftop water tower',
    'the theater district marquee lights',
    'a fire escape zigzag',
    'a crosstown bus shelter',
    'a corner bodega awning',
    'the elevated park walkway',
    'a midtown scaffolding tunnel',
  ],
  brooklyn: [
    'the big suspension bridge cables',
    'a brownstone stoop line',
    'the promenade railing',
    'a warehouse loading dock',
    'the boardwalk lamp posts',
    'a rooftop water tank cluster',
    'the flea market tents',
  ],
  queens: [
    'the elevated train trestle',
    'a row of vinyl-sided rooftops',
    'the stadium light towers',
    'an airport flight-path overpass',
    'a night market food stall row',
    'the community garden fence',
    'a warehouse district water tower',
  ],
};

export const FLAVOR_TEMPLATES = [
  'Spotted web-slinging near {landmark}.',
  'Reported swinging between rooftops by {landmark}.',
  'Our hero was seen zipping past {landmark}.',
  'Witnesses say a red-and-blue blur cleared {landmark} in one bound.',
  'Grainy phone footage shows web-lines anchored to {landmark}.',
  'A quick assist reported near {landmark} — no name left behind, as usual.',
  'Something stuck to the underside of {landmark}. Guy in the chair says: "yep, that tracks."',
  'Sighted taking the scenic route over {landmark}.',
  'Neighborhood watch chatter: hero spotted resting on {landmark}.',
  'Kids on the block swear they saw a web-line snap taut across {landmark}.',
  'Local livestream caught a blur vaulting off {landmark}.',
  'Traffic cam glimpsed a figure swinging low past {landmark}.',
  'Delivery drone footage: unexplained shadow over {landmark}.',
  'Rooftop pigeons scattered near {landmark} — usual sign he was through.',
  'A dropped web-cartridge turned up near {landmark}. Souvenir hunters, beware.',
  'Late-night dog walker reports a "thwip" sound near {landmark}.',
  'Security footage: brief red streak crossing {landmark}.',
  'Someone taped a "thank you" note to {landmark} again.',
  'A car alarm near {landmark} cut off mid-wail — he landed, then left.',
  'Spotted doing a lazy loop around {landmark} before vanishing uptown.',
  'Storefront camera near {landmark} caught two seconds of red and blue.',
  'A construction crew at {landmark} says a "guy in pajamas" borrowed their scaffolding.',
  'Overheard on the scanner: unusual activity reported at {landmark}.',
  'A stray web-line still dangles from {landmark} this morning.',
  'Skyline watchers logged a sighting arcing over {landmark}.',
  'Someone livestreamed a blur bouncing off {landmark} for six whole seconds.',
  "Local paper's \"mystery photo of the week\": a smudge near {landmark}.",
  'Reports of a rescue in progress near {landmark} — no injuries, lots of gossip.',
  'A rooftop party at {landmark} got an uninvited, very brief guest.',
  '"He\'s headed toward {landmark}, again," says the scanner chatter.',
];

export function generateFlavorText(boroughId) {
  const landmarks = LANDMARKS[boroughId] || LANDMARKS.manhattan;
  const landmark = landmarks[Math.floor(Math.random() * landmarks.length)];
  const template =
    FLAVOR_TEMPLATES[Math.floor(Math.random() * FLAVOR_TEMPLATES.length)];
  return template.replace('{landmark}', landmark);
}
