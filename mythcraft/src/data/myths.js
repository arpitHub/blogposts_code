// Every myth follows one shape: a claim, where it came from, three pieces of
// evidence, and a verdict that reacts to the visitor's guess.
export const CATEGORIES = [
  'All',
  'Science',
  'History',
  'Space',
  'Psychology',
  'Health',
  'Tech',
  'Money',
];

export const myths = [
  {
    slug: 'ten-percent-brain',
    category: 'Science',
    claim: 'You only use 10% of your brain.',
    hook: 'The most repeated statistic about the human brain, and nobody can find where it came from.',
    origin: [
      "No researcher has ever been traced as the source. The closest thing to an origin is the 1936 foreword to Dale Carnegie's How to Win Friends and Influence People, where Lowell Thomas wrote that the average person develops only about ten percent of their latent mental ability, and attached the psychologist William James's name to the idea.",
      "James wrote about untapped human potential, never about a percentage of unused brain. Early neuroscience helped the number stick: for decades, large stretches of cortex had no known function and were described as 'silent' — a limit of the instruments of the time, not a finding about idle tissue.",
    ],
    evidence: [
      {
        label: 'Imaging',
        title: 'Scans find activity across the whole brain, not a tenth of it',
        body: 'Functional imaging shows different regions carrying different loads from moment to moment, but across the course of an ordinary day virtually every area does measurable work. No large stretch sits permanently idle.',
        stat: '100%',
        statLabel: 'of the brain shows measurable activity over a normal day',
      },
      {
        label: 'Energy',
        title: 'The brain is far too expensive to run at a tenth',
        body: "It makes up about 2% of body weight but uses roughly 20% of the body's energy at rest. Evolution does not maintain an organ at that price and leave nine tenths of it unused.",
        stat: '20%',
        statLabel: "of the body's resting energy, for 2% of its weight",
      },
      {
        label: 'Damage',
        title: 'There is no spare 90% to lose',
        body: 'Strokes and injuries settle the question: damage to even a small area can cost speech, movement, memory, or the ability to recognise a face. If most of the brain were surplus, small lesions almost anywhere would be harmless.',
        stat: '—',
        statLabel: 'small lesions cause real deficits nearly anywhere',
      },
    ],
    verdict: 'busted',
    correctReaction:
      'Your guess was right. The number has no study behind it, only a long chain of repetition.',
    incorrectReaction:
      "You called it fact. It's the most widely believed claim about the brain, and one of the least supported.",
    closing:
      'You use all of your brain, just not all of it at once. Different tasks recruit different networks, which is a genuinely interesting fact that the 10% story replaced with a much flatter one.',
  },
  {
    slug: 'napoleon-height',
    category: 'History',
    claim: 'Napoleon was famously, comically short.',
    hook: "The tiny emperor with something to prove — one of history's stickiest images.",
    origin: [
      'The idea took hold in British wartime cartoons that mocked Napoleon as a temperamental miniature tyrant.',
      "It got a second life from a units mix-up: French records gave his height in French pouces, a slightly longer inch than the English one, so translators who didn't adjust made him look shorter than he was.",
    ],
    evidence: [
      {
        label: 'Records',
        title: 'His own doctors measured him at about average height',
        body: "Napoleon's post-mortem measurements convert to roughly 5'7\" (170cm) in modern units, right around the average for a Frenchman of his generation.",
        stat: "5'7\"",
        statLabel: 'his recorded height, converted to modern units',
      },
      {
        label: 'Language',
        title: "'Le Petit Caporal' was a term of affection",
        body: "His troops' nickname for him referenced his early rank and his habit of fighting alongside ordinary soldiers, not his stature. Soldiers' letters and diaries don't mention unusual shortness.",
        stat: '—',
        statLabel: 'no contemporary account remarks on his height',
      },
      {
        label: 'Optics',
        title: 'He was often pictured beside an elite guard chosen for height',
        body: "Napoleon's Imperial Guard had a minimum height requirement, so paintings and cartoons placing him next to them exaggerate the contrast.",
        stat: '—',
        statLabel: 'comparison flatters the myth, not the facts',
      },
    ],
    verdict: 'busted',
    correctReaction: "Your guess was right. The 'tiny tyrant' image is mostly propaganda.",
    incorrectReaction:
      "You called it fact. It's one of history's most successful smear campaigns.",
    closing:
      "Napoleon was an ordinary height for his time and place. The 'short' story survives because it's a satisfying way to explain a man widely seen as overreaching, not because it's true.",
  },
  {
    slug: 'no-gravity-in-space',
    category: 'Space',
    claim: "There's no gravity in space, that's why astronauts float.",
    hook: 'The reason floating astronauts get taught wrong in almost every classroom.',
    origin: [
      "The word 'zero-gravity' gets used loosely for anything in orbit, and the visual of floating makes 'no gravity up there' feel obvious.",
      "But gravity doesn't switch off a few hundred kilometers up; it's still doing most of its job.",
    ],
    evidence: [
      {
        label: 'Physics',
        title: 'Gravity at the ISS is nearly as strong as on the ground',
        body: "The International Space Station orbits about 400km up, where Earth's gravity is still roughly 90% as strong as at sea level. If gravity had switched off, the station would fly off in a straight line, not stay in orbit.",
        stat: '~90%',
        statLabel: 'of surface gravity, at ISS altitude',
      },
      {
        label: 'Mechanics',
        title: 'Floating is free-fall, not weightlessness',
        body: 'Astronauts and their station are constantly falling toward Earth, but moving sideways fast enough to keep missing it. That continuous fall is what feels weightless.',
        stat: '—',
        statLabel: 'gravity is pulling the whole time',
      },
      {
        label: 'Everyday version',
        title: 'You can feel the same effect on Earth',
        body: 'The stomach-drop at the top of a roller coaster, or inside a fast elevator that just started descending, is the same free-fall sensation, just for a second instead of months.',
        stat: '—',
        statLabel: 'same physics, much shorter fall',
      },
    ],
    verdict: 'busted',
    correctReaction:
      "Your guess was right. 'Zero gravity' is a misleading nickname for free-fall.",
    incorrectReaction:
      "You called it fact. It's one of the most common misconceptions taught almost everywhere.",
    closing:
      "Gravity is very much still there in low Earth orbit. What's missing isn't gravity, it's anything to push back against it.",
  },
  {
    slug: 'left-brain-right-brain',
    category: 'Psychology',
    claim: "Some people are 'left-brained,' others are 'right-brained.'",
    hook: 'The personality test that neuroscience never actually ran.',
    origin: [
      'The idea borrows a real fact, that some functions like most language processing lean into one hemisphere, and stretches it into a personality typology it was never meant to support.',
    ],
    evidence: [
      {
        label: 'Imaging',
        title: "A large brain-scan study looked for dominant-side people and didn't find them",
        body: "Researchers scanned over 1,000 brains looking for evidence that some people consistently rely more on one hemisphere. They found strong specialization for specific tasks, but no sign of a 'left-brained' or 'right-brained' type of person.",
        stat: '1,000+',
        statLabel: 'brains scanned, no dominant-side people found',
      },
      {
        label: 'Function',
        title: 'Complex tasks recruit both sides together',
        body: 'Creativity and logic both depend on networks that span both hemispheres working in coordination, not one side switched on and the other idle.',
        stat: '—',
        statLabel: 'both hemispheres active for most complex tasks',
      },
      {
        label: 'Anatomy',
        title: 'The two halves are in constant contact',
        body: 'A thick bundle of nerve fibers, the corpus callosum, connects the hemispheres and carries a continuous stream of information between them during almost everything you do.',
        stat: '—',
        statLabel: 'constant cross-talk between hemispheres',
      },
    ],
    verdict: 'busted',
    correctReaction: "Your guess was right. It's a tidy story with no scan to back it up.",
    incorrectReaction:
      "You called it fact. The specific-function part is real, the personality-type part isn't.",
    closing:
      "Hemispheres do specialize, language usually leans left, spatial attention often leans right, but that's a division of labor within one working brain, not two competing personalities.",
  },
  {
    slug: 'knuckle-cracking-arthritis',
    category: 'Health',
    claim: 'Cracking your knuckles causes arthritis.',
    hook: "The warning almost everyone's heard from a parent, at least once.",
    origin: [
      "The sound itself is unsettling enough that it's easy to assume something is being damaged.",
      'The myth likely persists because arthritis is common with age, and knuckle-cracking habits are common too, so the two get blamed on each other.',
    ],
    evidence: [
      {
        label: 'Self-experiment',
        title: 'One doctor tested it on himself for 60 years',
        body: 'Physician Donald Unger cracked the knuckles on only one hand for decades, leaving the other alone, specifically to test this claim. Neither hand developed arthritis.',
        stat: '60 yrs',
        statLabel: 'of one-sided knuckle-cracking, no difference found',
      },
      {
        label: 'Studies',
        title: 'Larger studies found no link either',
        body: 'Research comparing regular knuckle-crackers to non-crackers has consistently failed to find a higher rate of arthritis in the cracking group.',
        stat: '—',
        statLabel: 'no elevated arthritis rate found in crackers',
      },
      {
        label: 'Mechanism',
        title: 'The sound is gas, not damage',
        body: "The pop comes from a bubble collapsing in the fluid that lubricates the joint, not bone grinding on bone. That's also why a joint can't be cracked again right away, the gas needs time to redissolve.",
        stat: '—',
        statLabel: 'a gas bubble, not joint wear',
      },
    ],
    verdict: 'busted',
    correctReaction: 'Your guess was right. Go ahead and crack away, guilt-free.',
    incorrectReaction:
      "You called it fact. This one's been repeated so often it feels obviously true.",
    closing:
      "Knuckle-cracking might be annoying to sit next to, but the evidence doesn't connect it to arthritis. The sound is just gas moving, not joints wearing down.",
  },
  {
    slug: 'incognito-anonymity',
    category: 'Tech',
    claim: 'Private/incognito browsing makes you anonymous online.',
    hook: 'The setting most people over-trust, until they read what it actually promises.',
    origin: [
      "The name does a lot of the misleading work. 'Private' and 'incognito' sound like a cloak, when the feature was really built to stop one specific thing: your own device keeping a local record.",
    ],
    evidence: [
      {
        label: 'Scope',
        title: 'It only clears local traces',
        body: "Incognito mode skips saving your history, cookies, and site data on your own device once you close the window. That's the entire promise.",
        stat: '—',
        statLabel: 'local history and cookies only',
      },
      {
        label: 'Visibility',
        title: 'Your network can still see everything',
        body: "Your internet provider, your employer or school's network, and the websites you visit can all still see your activity. Incognito mode has no effect on any of them.",
        stat: '—',
        statLabel: 'no protection from network-level visibility',
      },
      {
        label: 'Legal record',
        title: 'This distinction ended up in court',
        body: 'Google faced a major lawsuit over users believing incognito mode meant untracked browsing, and settled by agreeing to destroy related data and clarify what the mode actually does.',
        stat: '$0',
        statLabel: 'extra anonymity added against your network or the sites you visit',
      },
    ],
    verdict: 'busted',
    correctReaction:
      'Your guess was right. It hides your history from the next person on your device, nothing more.',
    incorrectReaction:
      'You called it fact. The name oversells it, and a lot of people believed it too.',
    closing:
      "Incognito mode is a 'don't save this locally' switch, not an invisibility cloak. For real privacy from your network, that's a different tool entirely.",
  },
  {
    slug: 'credit-card-balance-score',
    category: 'Money',
    claim: 'Carrying a balance on your credit card helps your credit score.',
    hook: 'The advice that quietly costs people interest for no scoring benefit.',
    origin: [
      "It's a reasonable-sounding leap: using credit responsibly helps your score, so surely using it more, by carrying a balance, helps more. Credit scoring doesn't work that way.",
    ],
    evidence: [
      {
        label: 'Scoring model',
        title: 'Utilization is measured, not interest paid',
        body: "Scoring models look at how much of your available credit you're using at a given moment, not whether you paid it off or let it carry over with interest.",
        stat: '—',
        statLabel: "utilization ratio is what's scored",
      },
      {
        label: 'Best practice',
        title: 'Paying in full is treated the same or better',
        body: "Paying your statement balance in full each month keeps utilization low and avoids interest entirely. There's no scoring credit for choosing to pay interest instead.",
        stat: '—',
        statLabel: 'no scoring benefit to carrying a balance',
      },
      {
        label: 'Cost',
        title: 'The only guaranteed effect is the interest charge',
        body: "Carrying a balance has one certain outcome: interest accrues on whatever's left unpaid, with no offsetting score benefit to make that trade worthwhile.",
        stat: '0',
        statLabel: 'score points gained by paying interest instead of paying in full',
      },
    ],
    verdict: 'busted',
    correctReaction:
      "Your guess was right. Utilization matters, carrying interest doesn't add anything on top.",
    incorrectReaction:
      'You called it fact. This one costs real money if you act on it, worth double-checking.',
    closing:
      'Keeping your utilization low and paying on time drives your score. Carrying a balance just adds interest, with no scoring upside attached to it.',
  },
  {
    slug: 'viking-horned-helmets',
    category: 'History',
    claim: 'Vikings wore horned helmets into battle.',
    hook: "The single most recognizable image of Vikings, and the one archaeologists can't find any evidence for.",
    origin: [
      "The look comes from 19th-century Romantic art and costume design, most notably Carl Emil Doepler's costumes for an 1876 production of Wagner's Ring Cycle, centuries after the Viking Age had ended.",
    ],
    evidence: [
      {
        label: 'Archaeology',
        title: 'No Viking-Age helmet with horns has ever been found',
        body: 'Excavated Viking helmets, and there are very few complete ones, are simple rounded or conical shapes built for protection, with no horns anywhere on them.',
        stat: '0',
        statLabel: 'horned helmets recovered from Viking-Age sites',
      },
      {
        label: 'Practicality',
        title: 'Horns are a liability in close combat',
        body: 'A horn gives an opponent something to grab or strike for extra leverage, exactly the opposite of what protective headgear is for.',
        stat: '—',
        statLabel: 'a combat disadvantage, not an advantage',
      },
      {
        label: 'Origin',
        title: 'The image is 19th-century opera costume design',
        body: 'The horned look was designed for stage spectacle nearly a thousand years after the Viking Age, and 19th-century Romantic painting cemented it as the default image.',
        stat: '~1876',
        statLabel: 'when the horned costume design took hold',
      },
    ],
    verdict: 'busted',
    correctReaction:
      "Your guess was right. It's a costume designer's invention, not a historical record.",
    incorrectReaction:
      "You called it fact. It's one of the most convincing costume decisions ever made.",
    closing:
      "Real Viking helmets were plain and functional. The horns arrived a thousand years late, courtesy of opera, and never left pop culture's imagination.",
  },
  {
    slug: 'sharks-older-than-trees',
    category: 'Science',
    claim: 'Sharks have been around longer than trees.',
    hook: "A comparison so lopsided it sounds made up. It isn't.",
    origin: [
      'This one flips the usual myth-busting script: it sounds like an exaggeration built for a headline, but the fossil record backs it up.',
    ],
    evidence: [
      {
        label: 'Fossil record',
        title: 'Shark ancestors go back roughly 400 million years',
        body: 'Fossil evidence places early shark relatives in the oceans during the Devonian period, tens of millions of years before trees existed anywhere on land.',
        stat: '~400M',
        statLabel: 'years since early shark ancestors appear in the fossil record',
      },
      {
        label: 'Botany',
        title: 'The first tree-like plants show up tens of millions of years later',
        body: 'Early trees, like Archaeopteris, appear in the fossil record roughly 350-390 million years ago, well after sharks were already established predators.',
        stat: '~370M',
        statLabel: 'years since the first tree-like plants appear',
      },
      {
        label: 'Persistence',
        title: 'Sharks outlasted several mass extinctions',
        body: 'Sharks have survived multiple mass extinction events, including the one that ended the dinosaurs, while individual tree species and forests have come and gone many times over.',
        stat: '5',
        statLabel: 'mass extinctions sharks have survived',
      },
    ],
    verdict: 'confirmed',
    correctReaction: 'Your guess was right. This one actually holds up.',
    incorrectReaction:
      'You called it myth, understandable, but the timeline genuinely checks out.',
    closing:
      'Not every surprising claim is false. This is one of the rare cases where the wild-sounding version is also the accurate one.',
  },
  {
    slug: 'opposites-attract',
    category: 'Psychology',
    claim: 'Opposites attract, in the long run, people end up with someone very different from them.',
    hook: "A comforting idea for anyone who's ever felt like they don't match their type, but does it hold up?",
    origin: [
      "The phrase captures a real, minor pattern, people sometimes seek a partner who's strong where they're weak, and stretches it into a general law of attraction that the research doesn't fully support.",
    ],
    evidence: [
      {
        label: 'Research',
        title: 'Similarity predicts attraction more reliably',
        body: 'Decades of social-psychology research on the similarity-attraction effect consistently find that people are drawn to, and stay with, partners who share their values, interests, and habits more than people who differ from them.',
        stat: '—',
        statLabel: 'similarity outpredicts difference across studies',
      },
      {
        label: 'Long-term data',
        title: 'Couples tend to become more alike, not less, over time',
        body: "Long-term partners often converge on habits, attitudes, and even some physical patterns over years together, the opposite of what a strict 'opposites attract' story would predict.",
        stat: '—',
        statLabel: 'convergence, not divergence, over time',
      },
      {
        label: 'Nuance',
        title: 'A few traits really do work better as a mismatch',
        body: "Some complementary pairings, like one partner being more spontaneous and the other more organized, can work well together. That's a narrow exception, not the general rule the phrase implies.",
        stat: '—',
        statLabel: 'complementary traits: the exception, not the rule',
      },
    ],
    verdict: 'mixed',
    correctReaction:
      'Your guess was right. The full picture is more complicated than a catchy phrase.',
    incorrectReaction: 'You called it clean fact, the truth has more nuance than that.',
    closing:
      'A few traits genuinely work better as a mismatch, but overall, similarity is the far stronger predictor of who ends up together, and who stays together.',
  },
];

export const VERDICTS = {
  busted: { word: 'Busted', color: 'busted' },
  confirmed: { word: 'Confirmed', color: 'confirmed' },
  mixed: { word: "It's Complicated", color: 'mixed' },
};

export function getMyth(slug) {
  return myths.find((m) => m.slug === slug);
}

// Two more to read: same category first, then whatever follows in the list.
// Walking the list in order (rather than picking at random) keeps the footer
// stable across re-renders.
export function getRelatedMyths(slug, count = 2) {
  const current = getMyth(slug);
  if (!current) return [];
  const others = myths.filter((m) => m.slug !== slug);
  const sameCategory = others.filter((m) => m.category === current.category);
  const rest = others.filter((m) => m.category !== current.category);
  const startAt = myths.findIndex((m) => m.slug === slug);
  const rotated = [...rest.slice(startAt), ...rest.slice(0, startAt)];
  return [...sameCategory, ...rotated].slice(0, count);
}
