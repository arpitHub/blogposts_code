import type { Player, Source } from "../types";

// Every fact below is traced to content/research/laver.md. Items marked ❗ there are either left
// out or shown under "Where sources differ".

const S = {
  ithf: { label: "International Tennis Hall of Fame: Rod Laver", url: "https://www.tennisfame.com/hall-of-famers/inductees/rod-laver" },
  enc: { label: "Encyclopedia.com: Rod Laver", url: "https://www.encyclopedia.com/people/sports-and-games/sports-biographies/rod-laver" },
  ebsco: { label: "EBSCO Research Starters: Rod Laver", url: "https://www.ebsco.com/research-starters/biography/rod-laver/" },
  brit: { label: "Britannica: Rod Laver", url: "https://www.britannica.com/biography/Rod-Laver" },
  wp: { label: "Wikipedia: Rod Laver", url: "https://en.wikipedia.org/wiki/Rod_Laver" },
  ta: { label: "Tennis Australia: Rod Laver", url: "https://www.tennis.com.au/fan-zone/australian-players/rod-laver" },
  sahof: { label: "Sport Australia Hall of Fame: Rod Laver", url: "https://sahof.org.au/hall-of-fame-member/rod-laver/" },
  myth: { label: "Bleacher Report: Dispelling the myths of “Rocket” Rod Laver", url: "https://bleacherreport.com/articles/213115-dispelling-the-myths-of-rocket-rod-laver" },
  tmBorn: { label: "Tennis Majors: The day Rod Laver was born", url: "https://www.tennismajors.com/atp/august-9-1938-the-day-rod-laver-was-born-tennis-majors-279971.html" },
  hollis: { label: "EssentiallySports: Laver on Charlie Hollis and the topspin backhand", url: "https://www.essentiallysports.com/atp-tennis-news-youll-never-win-wimbledon-with-a-sliced-backhand-rod-laver-reveals-mastering-his-skills-with-a-wooden-racket/" },
  hopman: { label: "Sport Australia Hall of Fame: Harry Hopman", url: "https://sahof.org.au/hall-of-fame-member/harry-hopman/" },
  hopmanItHF: { label: "ITHF: Harry Hopman", url: "https://www.tennisfame.com/hall-of-famers/inductees/harry-hopman" },
  w1959: { label: "Wikipedia: 1959 Wimbledon men's singles", url: "https://en.wikipedia.org/wiki/1959_Wimbledon_Championships_%E2%80%93_Men%27s_singles" },
  a1960: { label: "Wikipedia: 1960 Australian Championships men's singles", url: "https://en.wikipedia.org/wiki/1960_Australian_Championships_%E2%80%93_Men%27s_singles" },
  w1960: { label: "Wikipedia: 1960 Wimbledon men's singles", url: "https://en.wikipedia.org/wiki/1960_Wimbledon_Championships_%E2%80%93_Men%27s_singles" },
  w1961: { label: "Wikipedia: 1961 Wimbledon men's singles", url: "https://en.wikipedia.org/wiki/1961_Wimbledon_Championships_%E2%80%93_Men%27s_singles" },
  f1962: { label: "Wikipedia: 1962 French Championships men's singles", url: "https://en.wikipedia.org/wiki/1962_French_Championships_%E2%80%93_Men%27s_singles" },
  w1962: { label: "Wikipedia: 1962 Wimbledon men's singles", url: "https://en.wikipedia.org/wiki/1962_Wimbledon_Championships_%E2%80%93_Men%27s_singles" },
  atp1962: { label: "ATP draws archive: Wimbledon 1962", url: "https://www.atptour.com/en/scores/archive/wimbledon/540/1962/draws" },
  jrank: { label: "JRank: Laver turns pro", url: "https://sports.jrank.org/pages/2749/Laver-Rod-Turns-Pro.html" },
  wbur: { label: "WBUR Only A Game: Rod Laver on the pro circuit's early days", url: "https://www.wbur.org/onlyagame/2016/04/30/rod-laver-book-tennis" },
  racketeers: { label: "Laver Cup: The Racketeers", url: "https://lavercup.com/news/2017/09/19/the-racketeers" },
  proMajors: { label: "Wikipedia: Major professional tennis tournaments before the Open Era", url: "https://en.wikipedia.org/wiki/Major_professional_tennis_tournaments_before_the_Open_Era" },
  wpro: { label: "Wikipedia: Wimbledon Pro", url: "https://en.wikipedia.org/wiki/Wimbledon_Pro" },
  bournemouth: { label: "ATP Heritage: Bournemouth 1968, the start of the Open Era", url: "https://www.atptour.com/en/news/atp-heritage-open-tennis-laver-rosewall-cox-1968-bournemouth" },
  tcBournemouth: { label: "Tennis.com: 1968, Open Era begins in Bournemouth", url: "https://www.tennis.com/news/articles/1968-open-era-begins-in-bournemouth" },
  ithfOpen: { label: "ITHF: 5 things to know about the dawn of the Open Era", url: "https://www.tennisfame.com/blog/2018/4/5-things-to-know-the-dawn-of-the-open-era" },
  f1968: { label: "Tennis Majors: Rosewall edges Laver, 1968", url: "https://www.tennismajors.com/our-features/on-this-day/april-28-1968-ken-rosewall-edges-rod-laver-to-win-the-first-open-tournament-tennis-majors-88239.html" },
  wpf1968: { label: "Wikipedia: 1968 French Open men's singles", url: "https://en.wikipedia.org/wiki/1968_French_Open_%E2%80%93_Men%27s_singles" },
  w1968: { label: "Wikipedia: 1968 Wimbledon men's singles", url: "https://en.wikipedia.org/wiki/1968_Wimbledon_Championships_%E2%80%93_Men%27s_singles" },
  cnbc: { label: "CNBC: The Wimbledon men's champ earned $2,621 in 1968", url: "https://www.cnbc.com/2018/07/12/wimbledon-mens-champ-got-2621-in-1968heres-what-he-wins-this-year.html" },
  mlk: { label: "Stanford King Institute: Assassination of Martin Luther King Jr.", url: "https://kinginstitute.stanford.edu/assassination-martin-luther-king-jr" },
  a1969: { label: "Wikipedia: 1969 Australian Open men's singles", url: "https://en.wikipedia.org/wiki/1969_Australian_Open_%E2%80%93_Men%27s_singles" },
  f1969: { label: "Wikipedia: 1969 French Open men's singles", url: "https://en.wikipedia.org/wiki/1969_French_Open_%E2%80%93_Men%27s_singles" },
  u1969: { label: "Wikipedia: 1969 US Open men's singles", url: "https://en.wikipedia.org/wiki/1969_US_Open_%E2%80%93_Men%27s_singles" },
  wim1969: { label: "Wimbledon archive: 1969 gentlemen's singles draw (PDF)", url: "https://assets.wimbledon.com/archive/draws/pdfs/draws/1969_MS_A4.pdf" },
  tc1969: { label: "Tennis.com: 1969, Rod Laver wins his second Grand Slam", url: "https://www.tennis.com/news/articles/1969-rod-laver-wins-his-second-grand-slam" },
  revolt: { label: "Tennis.com: Rocket's Revolt, the summer of 1969", url: "https://www.tennis.com/news/articles/rocket-s-revolt-rod-laver-s-miraculous-slam-winning-summer-of-1969" },
  tnow: { label: "Tennis Now: The greatest matches of the Open Era", url: "https://tennisnow.com/the-greatest-tennis-matches-in-open-era-history-2/" },
  apollo: { label: "NASA: July 20, 1969, one giant leap for mankind", url: "https://www.nasa.gov/history/july-20-1969-one-giant-leap-for-mankind/" },
  wct71: { label: "Wikipedia: 1971 WCT Finals singles", url: "https://en.wikipedia.org/wiki/1971_World_Championship_Tennis_Finals_%E2%80%93_Singles" },
  wct72: { label: "Tennis.com: The 1972 Laver vs Rosewall WCT Final in Dallas", url: "https://www.tennis.com/news/articles/1972-the-rod-laver-vs-ken-rosewall-wct-final-in-dallas" },
  wct72b: { label: "Tennis.com: 1972 WCT Final draws 21 million viewers", url: "https://www.tennis.com/pro-game/2020/05/tbt-1972-wct-finals-dallas-ken-rosewall-rod-laver-classic-draws-21-million-viewers/88778/" },
  dc73: { label: "Tennis.com: Newcombe and Laver open the 1973 Davis Cup final", url: "https://www.tennis.com/news/articles/on-this-day-aussie-legends-newcombe-laver-take-five-setters-1973-davis-cup-final" },
  wpdc73: { label: "Wikipedia: 1973 Davis Cup", url: "https://en.wikipedia.org/wiki/1973_Davis_Cup" },
  rankings: { label: "ATP: Nastase becomes the first No. 1 of computer rankings", url: "https://www.atptour.com/en/news/nastase-number-one-club-rise" },
  tmRank: { label: "Tennis Majors: The first ATP rankings, 23 Aug 1973", url: "https://www.tennismajors.com/atp/august-23-1973-the-first-atp-ranking-is-released-459775.html" },
  friars: { label: "Wikipedia: San Diego Friars (1975–1978)", url: "https://en.wikipedia.org/wiki/San_Diego_Friars_(1975%E2%80%931978)" },
  siMary: { label: "Sports Illustrated: Mary Laver dies at 84", url: "https://www.si.com/tennis/2012/11/13/mary-laver-dies" },
  taMary: { label: "Tennis Australia: Mary Laver passes away", url: "https://www.tennis.com.au/news/2012/11/14/mary-laver-wife-of-rod-passes-away" },
  wapoStroke: { label: "Washington Post: Laver suffers stroke during interview (1998)", url: "https://www.washingtonpost.com/archive/sports/1998/07/28/tennis-star-laver-suffers-stroke-during-interview/a264166d-4ee5-48ef-b837-7c20730ef32e/" },
  cbsStroke: { label: "CBS News: Tennis legend Laver suffers stroke", url: "https://www.cbsnews.com/news/tennis-legend-laver-suffers-stroke/" },
  mp: { label: "Melbourne Park: Our history", url: "https://www.melbournepark.com.au/about/our-history/" },
  rla: { label: "Wikipedia: Rod Laver Arena", url: "https://en.wikipedia.org/wiki/Rod_Laver_Arena" },
  abc: { label: "ABC News: Laver given Australia's highest honour (2016)", url: "https://www.abc.net.au/news/2016-01-26/rod-laver-leads-sports-stars-honoured-on-australia-day/7114336" },
  lc17: { label: "Wikipedia: 2017 Laver Cup", url: "https://en.wikipedia.org/wiki/2017_Laver_Cup" },
  lc: { label: "Laver Cup: Rod Laver", url: "https://lavercup.com/rod-laver" },
  o2: { label: "O2 arena: Laver Cup 2017 highlights", url: "https://www.o2arena.cz/en/laver-cup-2017-prague-highlights/" },
  atHof: { label: "Wikipedia: Australian Tennis Hall of Fame", url: "https://en.wikipedia.org/wiki/Australian_Tennis_Hall_of_Fame" },
  titles: { label: "Yahoo Sports: Rod Laver's career and titles", url: "https://sports.yahoo.com/articles/rod-laver-career-titles-grand-161000137.html" },
  ta128: { label: "Tennis Abstract: The Tennis 128, No. 1 Rod Laver", url: "https://www.tennisabstract.com/blog/2022/12/21/the-tennis-128-no-1-rod-laver/" },
  dunlop: { label: "Dunlop: Rod Laver", url: "https://dunlopsports.com/team/tennis/ambassadors/rod-laver/" },
  us2026: { label: "Wikipedia: 2026 US Open men's singles", url: "https://en.wikipedia.org/wiki/2026_US_Open_%E2%80%93_Men%27s_singles" },
  records: { label: "Wikipedia: All-time tennis records, men's singles", url: "https://en.wikipedia.org/wiki/All-time_tennis_records_%E2%80%93_Men's_singles" },
} satisfies Record<string, Source>;

const laver: Player = {
  slug: "laver",
  name: "Rod Laver",
  shortName: "Laver",
  born: { date: "9 August 1938", place: "Rockhampton, Queensland" },
  country: "Australia",
  lifespan: "Born 1938",
  identity: "The only player, man or woman, to win the calendar Grand Slam in singles twice.",
  theme: "laver",
  eraLabel: "The 1960s",
  heroFacts: [
    { label: "Majors", value: "11" },
    { label: "Calendar Slams", value: "1962 · 1969" },
    { label: "Plays", value: "Left-handed" },
    { label: "Nickname", value: "“Rocket”" },
  ],
  portrait: {
    monogram: "RL",
    alt: "Photograph of Rod Laver",
    commons: {
      files: [],
      category: "Category:Rod Laver",
      exclude: "arena|cup|stamp|statue|trophy|logo|court|stadium",
      era: [1959, 1970],
    },
  },

  journey: [
    {
      set: 1,
      title: "Roots",
      years: [1938, 1955],
      headline: "A court beside every house",
      paragraphs: [
        "Rodney George Laver was born on 9 August 1938 in Rockhampton, Queensland. Tennis ran through the family. His father Roy, a cattle rancher, was one of 13 children, all of whom played. His mother Melba (née Roffey) was a tournament-standard player. Roy and Melba had met at a tournament in the Queensland town of Dingo, and wherever they lived afterwards, there was a court next to the house.",
        "His older brothers Trevor and Robert and younger sister Lois all played too. Rod started at about six, taking on his brothers with a hand-me-down racquet whose handle had been sawn down to fit his hand.",
        "His early coach, Charlie Hollis, pushed the small left-hander towards the shot that would define him. As Laver later recalled it: “You'll never win Wimbledon with a slice backhand. You must learn to put topspin on the ball.” He was then picked for a junior camp sponsored by the Brisbane Courier-Mail and run by Harry Hopman, Australia's Davis Cup captain.",
      ],
      tile: { label: "Children in his father's family, all tennis players", value: "13" },
      eraNote: {
        text: "The year Laver was born, 1938, Don Budge became the first man to win all four major titles in a single calendar year. The next man to do it would be Laver himself, in 1962.",
        sources: [S.f1962, S.ithf],
      },
      sources: [S.enc, S.ithf, S.tmBorn, S.hollis],
    },
    {
      set: 2,
      title: "Rocket",
      years: [1956, 1958],
      headline: "An ironic nickname and two junior crowns",
      paragraphs: [
        "Hopman gave the teenager the nickname that stuck for life: “Rocket”. It was partly a joke. It had nothing to do with speed around the court. It referred to the raw, untamed power in his left arm, which made his game wildly erratic as a youngster.",
        "The power came with results. Laver won the junior championships of both the United States and Australia. Sources differ on the order: one puts the US title in 1956, on his first trip abroad at 17, and the Australian title in 1957, while others list both in 1957.",
      ],
      tile: { label: "Junior titles", value: "US · AUS", caption: "1956–57; sources differ on the year" },
      eraNote: {
        text: "Hopman was building a dynasty. Between 1938 and 1967 he captained 22 Australian Davis Cup teams, 16 of which won the Cup, with players including Sedgman, Hoad, Rosewall, Emerson and, soon, Laver.",
        sources: [S.hopman, S.hopmanItHF],
      },
      sources: [S.ithf, S.ta, S.myth, S.sahof, S.enc],
    },
    {
      set: 3,
      title: "Breakthrough",
      years: [1959, 1960],
      headline: "Losing Wimbledon finals, then winning from two sets down",
      paragraphs: [
        "In 1959 Laver reached his first Wimbledon singles final and lost it in straight sets to Alex Olmedo. He left London with a title all the same: the mixed doubles, won with Darlene Hard. The same year he took the Australian doubles with Bob Mark and helped Australia win the Davis Cup, the first of four straight Cup wins with him in the team.",
        "His first major singles title came at the 1960 Australian Championships, and he had to fight for it. Against Neale Fraser in the final he lost the first two sets, then won the next three, taking the last two 8–6, 8–6. At Wimbledon later that year it was Fraser's turn, and Laver was a beaten finalist for the second year running.",
      ],
      matches: [
        {
          year: 1959, event: "Wimbledon", round: "Final", opponent: "Alex Olmedo", score: "4–6, 3–6, 4–6", result: "L",
          why: "His first major singles final; the first of six straight Wimbledon finals in the years he entered.",
          sources: [S.w1959, S.ithf],
        },
        {
          year: 1960, event: "Australian Championships", round: "Final", opponent: "Neale Fraser", score: "5–7, 3–6, 6–3, 8–6, 8–6", result: "W",
          why: "Major title No. 1, won from two sets down.",
          sources: [S.a1960],
        },
        {
          year: 1960, event: "Wimbledon", round: "Final", opponent: "Neale Fraser", result: "L",
          why: "A second straight Wimbledon final lost.",
          sources: [S.w1960, S.ithf],
        },
      ],
      tile: { label: "Came back from", value: "0–2 sets", caption: "To win his first major, Australian Championships 1960" },
      eraNote: {
        text: "The majors were amateur-only. The best players were not allowed to take prize money there, and those who turned professional were shut out of the Grand Slam events entirely. That rule would shape the next decade of Laver's life.",
        sources: [S.ithfOpen, S.tcBournemouth],
      },
      sources: [S.w1959, S.a1960, S.w1960, S.ithf, S.ta],
    },
    {
      set: 4,
      title: "Champion",
      years: [1961, 1961],
      headline: "Third final, first Wimbledon title",
      paragraphs: [
        "At the third attempt, Laver won Wimbledon. In the 1961 final he beat the American Chuck McKinley 6–3, 6–1, 6–4 for his second major title.",
        "The International Tennis Hall of Fame points to a remarkable run: in every year he entered Wimbledon from 1959 onwards, Laver reached the final, six times in a row. He also remained a fixture of Australia's Davis Cup side, which won the Cup every year from 1959 to 1962.",
      ],
      matches: [
        {
          year: 1961, event: "Wimbledon", round: "Final", opponent: "Chuck McKinley", score: "6–3, 6–1, 6–4", result: "W",
          why: "His first Wimbledon title, after losing the two previous finals.",
          sources: [S.w1961, S.ithf],
        },
      ],
      tile: { label: "Wimbledon final, 1961", value: "6–3 6–1 6–4", caption: "d. Chuck McKinley" },
      eraNote: {
        text: "Wimbledon was still closed to professionals and would stay that way until 1968, when the first open Wimbledon was played. Laver won that one too.",
        sources: [S.w1968],
      },
      sources: [S.ithf, S.w1961, S.ta],
    },
    {
      set: 5,
      title: "The First Slam",
      years: [1962, 1962],
      headline: "All four majors, and a match point saved in Paris",
      paragraphs: [
        "In 1962, aged 24, Laver won the Australian, French, Wimbledon and US Championships in a single year. That made him only the second man to complete the Grand Slam, after Don Budge in 1938. Three of his four finals were against his compatriot Roy Emerson.",
        "Paris was the closest call. In the quarter-final Laver saved a match point against Martin Mulligan, and in the final he lost the first two sets to Emerson before winning 3–6, 2–6, 6–3, 9–7, 6–2. At Wimbledon he gave Mulligan just five games in the final.",
        "That December, after winning the Davis Cup with Australia again, he turned professional. The decision would keep him out of the majors for the next five years.",
      ],
      matches: [
        {
          year: 1962, event: "Australian Championships", round: "Final", opponent: "Roy Emerson", score: "8–6, 0–6, 6–4, 6–4", result: "W",
          why: "Leg one of the Grand Slam.", sources: [S.f1962],
        },
        {
          year: 1962, event: "French Championships", round: "Quarter-final", opponent: "Martin Mulligan", result: "W",
          why: "Laver saved a match point; without it there is no 1962 Slam.", sources: [S.f1962],
        },
        {
          year: 1962, event: "French Championships", round: "Final", opponent: "Roy Emerson", score: "3–6, 2–6, 6–3, 9–7, 6–2", result: "W",
          why: "Leg two, won from two sets down.", sources: [S.f1962],
        },
        {
          year: 1962, event: "Wimbledon", round: "Final", opponent: "Martin Mulligan", score: "6–2, 6–2, 6–1", result: "W",
          why: "Leg three, and his second straight Wimbledon title.", sources: [S.ithf, S.w1962, S.atp1962],
        },
        {
          year: 1962, event: "US Championships", round: "Final", opponent: "Roy Emerson", score: "6–2, 6–4, 5–7, 6–4", result: "W",
          why: "Completed the Grand Slam, the first by a man since 1938.", sources: [S.f1962],
        },
      ],
      tile: { label: "Majors won in 1962", value: "4 / 4", caption: "Second man ever, after Don Budge" },
      eraNote: {
        text: "A Grand Slam was so rare that the only man to have done it before Laver, Don Budge, had done it 24 years earlier, the year Laver was born.",
        sources: [S.f1962, S.ithf],
      },
      sources: [S.ithf, S.f1962, S.w1962, S.wp],
    },
    {
      set: 6,
      title: "Exile",
      years: [1963, 1966],
      headline: "Turning pro, and learning to lose again",
      paragraphs: [
        "Laver turned professional in December 1962. Members of the pros' players' association reportedly clubbed together to guarantee him US$110,000 over three years. The price was steep: professionals were barred from the majors, so for five years (1963 to 1967) the reigning Grand Slam champion could not play any of them.",
        "The pros were a harder school than the amateurs. In the first half of 1963 Lew Hoad beat him in their first eight meetings, and Ken Rosewall won 11 of their first 13. Laver adapted. By the end of 1963, with three tournament wins, he was the No. 2 professional behind Rosewall. The circuit, as Laver later recalled, took him to places like bullrings and ice rinks.",
        "He rose to the top of the pro game. He won the US Pro in 1964 and 1966, and the Wembley Pro in London every year from 1964 to 1966. Off court, he married Mary Benson on 20 June 1966 in San Rafael, California; the two had met at the Jack Kramer Club near Los Angeles.",
      ],
      tile: { label: "His first matches against Lew Hoad as a pro", value: "0–8", caption: "Early 1963" },
      eraNote: {
        text: "Before the Open Era, professional tennis was a separate touring troupe. Players signed contracts and played one-night stands and their own “pro” championships, far from Wimbledon and Forest Hills.",
        sources: [S.racketeers, S.wbur],
      },
      sources: [S.wp, S.sahof, S.jrank, S.wbur, S.proMajors, S.siMary],
    },
    {
      set: 7,
      title: "King of the Pros",
      years: [1967, 1967],
      headline: "A professional Slam, and a pro final on Centre Court",
      paragraphs: [
        "1967 was Laver's peak year as a professional. He won the French Pro, the Wembley Pro and the US Pro, the three events historians now call the “pro majors”, for a clean sweep. (The label is a later convention, not an official title of the time.)",
        "In late August, from the 25th to the 28th, Wimbledon staged a professional event, the Wimbledon Pro. It was the first tournament there open to male professionals since 1930, and the only pro event ever held on Centre Court. Laver beat Rosewall 6–2, 6–2, 12–10 in the final. The tournament is widely credited with helping to open the door to Open tennis the next year.",
      ],
      matches: [
        {
          year: 1967, event: "Wimbledon Pro", round: "Final", opponent: "Ken Rosewall", score: "6–2, 6–2, 12–10", result: "W",
          why: "A professional final on Wimbledon's Centre Court, a step towards Open tennis.",
          sources: [S.wpro],
        },
      ],
      tile: { label: "Pro majors won in 1967", value: "3 / 3", caption: "French Pro, Wembley Pro, US Pro" },
      eraNote: {
        text: "The 1967 Wimbledon Pro was the first event at the All England Club open to male professionals since the British Professional Championships in 1930.",
        sources: [S.wpro],
      },
      sources: [S.proMajors, S.wpro, S.sahof],
    },
    {
      set: 8,
      title: "Open Doors",
      years: [1968, 1968],
      headline: "The Open Era begins, and Laver wins the first open Wimbledon",
      paragraphs: [
        "On 22 April 1968, at the West Hants Club in Bournemouth, the first tournament open to amateurs and professionals alike began. After five years away, Laver could play the majors again.",
        "At the first of them, the 1968 French Open, he reached the final and lost to Rosewall 6–3, 6–1, 2–6, 6–2. Wimbledon was different. In the first Wimbledon open to professionals he beat Tony Roche 6–3, 6–4, 6–2 for his seventh major title. The men's champion's prize that year was £2,000, about US$2,621 at the time.",
      ],
      matches: [
        {
          year: 1968, event: "French Open", round: "Final", opponent: "Ken Rosewall", score: "3–6, 1–6, 6–2, 2–6", result: "L",
          why: "The first major final of the Open Era.", sources: [S.f1968, S.wpf1968],
        },
        {
          year: 1968, event: "Wimbledon", round: "Final", opponent: "Tony Roche", score: "6–3, 6–4, 6–2", result: "W",
          why: "Champion of the first open Wimbledon.", sources: [S.w1968, S.ithf],
        },
      ],
      tile: { label: "Wimbledon champion's prize, 1968", value: "£2,000", caption: "About US$2,621 at the time" },
      eraNote: {
        text: "Open tennis arrived in a turbulent spring. Martin Luther King Jr. had been assassinated in Memphis on 4 April 1968, less than three weeks before play began at Bournemouth.",
        sources: [S.mlk, S.bournemouth],
      },
      sources: [S.bournemouth, S.tcBournemouth, S.f1968, S.w1968, S.cnbc],
    },
    {
      set: 9,
      title: "The Second Slam",
      years: [1969, 1969],
      headline: "Heat, cabbage leaves, spikes, and all four majors again",
      paragraphs: [
        "In 1969 Laver won all four majors again, this time against the full field of amateurs and professionals. It is still the only men's calendar Grand Slam of the Open Era. It nearly ended in Brisbane. His Australian Open semi-final against Tony Roche on the grass at Milton was played in heat of around 100°F and ran to 90 games over more than four hours: 7–5, 22–20, 9–11, 1–6, 6–3. Both men cooled off with wet cabbage leaves under their hats.",
        "At Wimbledon he was two sets down to India's Premjit Lall in the second round on Court 4, then won the last 15 games of the match. He took the French Open without dropping a set in the final against Rosewall and beat John Newcombe for the Wimbledon title.",
        "The last leg was played in the rain at Forest Hills. With the court wet and slippery, the referee Bill Talbert allowed Laver to change into spikes, and he beat Roche 7–9, 6–1, 6–2, 6–2. His winner's cheque was $16,000, then an astonishing sum. His son Rick was born about three weeks later.",
      ],
      matches: [
        {
          year: 1969, event: "Australian Open", round: "Semi-final", opponent: "Tony Roche", score: "7–5, 22–20, 9–11, 1–6, 6–3", result: "W",
          why: "Ninety games in fierce heat, the closest the second Slam came to ending.", sources: [S.tnow, S.revolt],
        },
        {
          year: 1969, event: "Australian Open", round: "Final", opponent: "Andrés Gimeno", score: "6–3, 6–4, 7–5", result: "W",
          why: "Leg one.", sources: [S.a1969, S.tc1969],
        },
        {
          year: 1969, event: "French Open", round: "Final", opponent: "Ken Rosewall", score: "6–4, 6–3, 6–4", result: "W",
          why: "Leg two, against the defending champion.", sources: [S.f1969, S.tc1969],
        },
        {
          year: 1969, event: "Wimbledon", round: "Second round", opponent: "Premjit Lall", result: "W",
          why: "Two sets down, he won the last 15 games.", sources: [S.tc1969, S.wim1969],
        },
        {
          year: 1969, event: "Wimbledon", round: "Final", opponent: "John Newcombe", score: "6–4, 5–7, 6–4, 6–4", result: "W",
          why: "Leg three, and his fourth Wimbledon singles title.", sources: [S.tc1969, S.wim1969],
        },
        {
          year: 1969, event: "US Open", round: "Final", opponent: "Tony Roche", score: "7–9, 6–1, 6–2, 6–2", result: "W",
          why: "Completed a second calendar Grand Slam, still unmatched.", sources: [S.u1969, S.tc1969],
        },
      ],
      tile: { label: "Calendar Grand Slams", value: "2", caption: "1962 and 1969; no one else has two" },
      eraNote: {
        text: "Between Laver's Wimbledon title and his US Open, Apollo 11 landed on the Moon on 20 July 1969.",
        sources: [S.apollo],
      },
      sources: [S.tc1969, S.revolt, S.tnow, S.a1969, S.f1969, S.u1969, S.wim1969, S.siMary],
    },
    {
      set: 10,
      title: "The Millionaire",
      years: [1970, 1978],
      headline: "The first tennis millionaire, and one more Davis Cup",
      paragraphs: [
        "Open tennis made the money real. In 1971 Laver won $292,717 in prize money, an unheard-of amount then, and became the first tennis player to pass US$1 million in career earnings. He also lost two famous finals at the WCT Finals in Dallas to his old rival Rosewall, in 1971 and 1972. The 1972 final, a fifth-set tiebreak, drew a US television audience in the tens of millions and is often called the match that made tennis in the United States.",
        "In 1973, when professionals were allowed into the Davis Cup for the first time, Laver came back for Australia. In the final in Cleveland he beat Tom Gorman in five sets, then clinched the Cup in doubles with John Newcombe. Australia won 5–0, and it was his fifth Davis Cup-winning team.",
        "In 1976, aged 38, he retired from the main tour. He kept playing World TeamTennis for the San Diego Friars from 1976 to 1978, and was named the league's Rookie of the Year in 1976.",
      ],
      matches: [
        {
          year: 1971, event: "WCT Finals, Dallas", round: "Final", opponent: "Ken Rosewall", score: "4–6, 6–1, 6–7(3), 6–7(4)", result: "L",
          why: "The first WCT Finals, a new showcase for the pro game.", sources: [S.wct71],
        },
        {
          year: 1972, event: "WCT Finals, Dallas", round: "Final", opponent: "Ken Rosewall", score: "6–4, 0–6, 3–6, 7–6(3), 6–7(5)", result: "L",
          why: "A classic watched by a huge US television audience.", sources: [S.wct72, S.wct72b],
        },
        {
          year: 1973, event: "Davis Cup final, Cleveland", round: "Singles", opponent: "Tom Gorman", score: "8–10, 8–6, 6–8, 6–3, 6–1", result: "W",
          why: "Aged 35, back in the Cup for the first time since 1962.", sources: [S.dc73, S.wpdc73],
        },
        {
          year: 1973, event: "Davis Cup final, Cleveland", round: "Doubles (with John Newcombe)", opponent: "Stan Smith / Erik van Dillen", result: "W",
          why: "Clinched the Cup; Australia won the final 5–0.", sources: [S.dc73, S.wpdc73],
        },
      ],
      tile: { label: "Career prize money, 1971", value: "$1M+", caption: "First tennis player past it" },
      eraNote: {
        text: "On 23 August 1973 the ATP published its first computer rankings, with Ilie Năstase at No. 1. Before that, world No. 1 was a matter of expert opinion, which is why claims about Laver's ranking in the 1960s vary.",
        sources: [S.rankings, S.tmRank],
      },
      sources: [S.brit, S.ebsco, S.enc, S.wct71, S.wct72, S.dc73, S.wpdc73, S.friars],
    },
    {
      set: 11,
      title: "Legacy",
      years: [1979, "present"],
      headline: "A stroke survived, an arena named, a cup in his honour",
      paragraphs: [
        "The honours followed: the International Tennis Hall of Fame in 1981, the Sport Australia Hall of Fame in 1985 (named a Legend in 2002), and the Australian Tennis Hall of Fame. On 27 July 1998, aged 59, Laver suffered a stroke while taping a television interview for ESPN. He was treated at UCLA Medical Center and faced a long rehabilitation.",
        "On 16 January 2000 the Centre Court at Melbourne Park was renamed Rod Laver Arena. His wife Mary died in November 2012, aged 84, at their home in Carlsbad, California, after 46 years of marriage. In 2016 he was made a Companion of the Order of Australia, Australia's highest honour.",
        "In September 2017 the first Laver Cup, a team event pitting Europe against the rest of the world, was played at the O2 Arena in Prague. Team Europe won it 15–9.",
      ],
      tile: { label: "Rod Laver Arena named", value: "2000", caption: "Melbourne Park's Centre Court" },
      eraNote: {
        text: "More than half a century later, no man has matched his 1969 feat. As of the end of the 2026 US Open, it remains the only men's calendar Grand Slam of the Open Era.",
        sources: [S.us2026, S.records],
      },
      sources: [S.ithf, S.sahof, S.atHof, S.wapoStroke, S.cbsStroke, S.mp, S.rla, S.siMary, S.taMary, S.abc, S.lc17, S.o2, S.lc],
    },
  ],

  records: [
    {
      label: "Calendar Grand Slams in singles",
      value: "2",
      countTo: 2,
      detail: "1962 and 1969. The only player, man or woman, with two.",
      status: "still-stands",
      sources: [S.ithf],
    },
    {
      label: "Men's calendar Slams in the Open Era",
      value: "1",
      countTo: 1,
      detail: "His 1969 Slam is the only one since 1968 (checked through the 2026 US Open).",
      status: "still-stands",
      sources: [S.u1969, S.us2026, S.records],
    },
    {
      label: "Major singles titles",
      value: "11",
      countTo: 11,
      detail: "Australian 1960, 1962, 1969 · French 1962, 1969 · Wimbledon 1961, 1962, 1968, 1969 · US 1962, 1969. Won despite missing five years of majors (1963–67) as a pro.",
      status: "stat",
      sources: [S.ithf, S.wp],
    },
    {
      label: "First tennis player past US$1M in career prize money",
      value: "1971",
      detail: "Helped by $292,717 won that season alone.",
      status: "first",
      sources: [S.brit, S.ebsco, S.enc],
    },
    {
      label: "Straight Wimbledon finals",
      value: "6",
      countTo: 6,
      detail: "In every Wimbledon he entered from 1959 to 1969: 1959, 1960, 1961, 1962, 1968, 1969.",
      status: "stat",
      sources: [S.ithf],
    },
    {
      label: "Davis Cup-winning teams",
      value: "5",
      countTo: 5,
      detail: "1959, 1960, 1961, 1962 and, as a pro, 1973.",
      status: "stat",
      sources: [S.ta, S.dc73],
    },
    {
      label: "Career singles titles",
      value: "~200",
      detail: "ITHF counts 200 across amateur, pre-Open pro and Open Era events; other sources say 198. We don't treat it as a record because the count depends on what is included.",
      status: "sources-differ",
      sources: [S.ithf, S.titles],
    },
  ],

  era: {
    onCourt: [
      {
        title: "Amateur vs professional",
        body: "Until April 1968 the four majors and the Davis Cup were amateur-only. Professionals toured separately and played their own championships, the US Pro, the Wembley Pro and the French Pro. That is why Laver, the best player of the mid-1960s, played no majors from 1963 to 1967.",
        sources: [S.ithfOpen, S.proMajors, S.racketeers],
      },
      {
        title: "The Open Era",
        body: "Open tennis began on 22 April 1968 at the British Hard Court Championships in Bournemouth. Wimbledon went open that summer, and professionals were admitted to the Davis Cup in 1973.",
        sources: [S.bournemouth, S.tcBournemouth, S.dc73],
      },
      {
        title: "Prize money",
        body: "The 1968 Wimbledon men's champion won £2,000 (about US$2,621). Laver's 1969 US Open cheque was $16,000. By 1971 he earned $292,717 in a season and became the sport's first millionaire.",
        sources: [S.cnbc, S.tc1969, S.brit],
      },
      {
        title: "Equipment and style",
        body: "Laver won both of his Grand Slams with Dunlop Maxply wooden racquets. His wristy topspin off both sides, and an attacking topspin lob, were an innovation in the 1960s. In 1968 his left wrist was measured at 7 inches around and his forearm at 12 inches.",
        sources: [S.dunlop, S.ta128, S.wp],
      },
      {
        title: "Rankings",
        body: "The ATP's computer rankings began on 23 August 1973. Before that, rankings were compiled by journalists and experts, so “world No. 1” claims for earlier years are judgments, not official positions.",
        sources: [S.rankings, S.tmRank],
      },
    ],
    inTheWorld: [
      {
        title: "1938",
        body: "The year of Laver's birth was the year Don Budge completed the first Grand Slam.",
        sources: [S.f1962],
      },
      {
        title: "April 1968",
        body: "Martin Luther King Jr. was assassinated in Memphis on 4 April, less than three weeks before open tennis began.",
        sources: [S.mlk],
      },
      {
        title: "July 1969",
        body: "Apollo 11 landed on the Moon on 20 July 1969, between Laver's Wimbledon and US Open titles.",
        sources: [S.apollo],
      },
    ],
  },

  beyond: [
    {
      title: "The first tennis millionaire",
      body: "In 1971 Laver became the first tennis player to earn more than US$1 million in career prize money, a sign of how quickly the Open Era changed the economics of the sport.",
      sources: [S.brit, S.ebsco],
    },
    {
      title: "Aggressive topspin tennis",
      body: "Coached from boyhood to hit a topspin backhand at a time when slice was the norm, Laver attacked off both wings with wristy topspin and used the topspin lob as a weapon, an innovation in the 1960s game.",
      sources: [S.hollis, S.ta128],
    },
    {
      title: "Helping open the game",
      body: "His victory in the 1967 Wimbledon Pro, the only professional event ever staged on Centre Court, is widely credited with helping make the case for Open tennis a year later.",
      sources: [S.wpro],
    },
    {
      title: "Rod Laver Arena",
      body: "Melbourne Park's Centre Court, home of the Australian Open final, has carried his name since 16 January 2000.",
      sources: [S.mp, S.rla],
    },
    {
      title: "The Laver Cup",
      body: "A team competition between Europe and the rest of the world, first played in Prague in September 2017 and named in his honour.",
      sources: [S.lc17, S.lc, S.o2],
    },
  ],

  legacy: [
    "Laver's record is defined by two things: what he won and what he was kept from. He won the Grand Slam as an amateur in 1962, spent five years barred from the majors as a professional, and won the Grand Slam again in 1969, when everyone was allowed to play. No one else has won two.",
    "Off the court he was a bridge between tennis eras. He was the dominant amateur of the early 1960s, one of the leading professionals of the touring years, and the first millionaire of the Open Era. Today his name is on the Australian Open's main stadium and on the Laver Cup.",
  ],

  sourcesDiffer: [
    {
      title: "How many titles?",
      body: "The International Tennis Hall of Fame credits Laver with 200 singles titles; other sources say 198. The total includes amateur events, pre-Open professional events and Open Era tournaments, which are counted differently. The ATP lists 72 Open Era titles, while ITHF text gives 77 for 1968–76.",
      sources: [S.ithf, S.titles],
    },
    {
      title: "World No. 1, 1964–70?",
      body: "ITHF describes Laver as world No. 1 from 1964 to 1970. Official computer rankings did not exist until August 1973, so this is an expert assessment rather than an official ranking.",
      sources: [S.ithf, S.rankings],
    },
    {
      title: "“Pro majors”",
      body: "Grouping the US Pro, Wembley Pro and French Pro as professional “majors” is a later convention used by historians. We use it with that caveat.",
      sources: [S.proMajors],
    },
    {
      title: "Junior title years",
      body: "Some accounts date Laver's US junior title to 1956 and his Australian junior title to 1957; others give 1957 for both.",
      sources: [S.sahof, S.enc],
    },
    {
      title: "The 1972 Dallas audience",
      body: "Accounts of the US television audience for the 1972 WCT Final vary between 21 and 23 million.",
      sources: [S.wct72, S.wct72b],
    },
  ],

  references: [S.ithf, S.brit, S.enc, S.ebsco, S.wp, S.ta, S.sahof, S.lc],
};

export default laver;
