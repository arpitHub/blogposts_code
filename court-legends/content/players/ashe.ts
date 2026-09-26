import type { Player, Source } from "../types";

// Every fact below is traced to content/research/ashe.md. Items marked ❗ there are either left
// out or shown under "Where sources differ". Match scores are written from Ashe's side.

const S = {
  ev: { label: "Encyclopedia Virginia: Arthur Ashe (1943–1993)", url: "https://encyclopediavirginia.org/entries/ashe-arthur-1943-1993/" },
  lva: { label: "Library of Virginia: Dictionary of Virginia Biography, Arthur Robert Ashe", url: "https://www.lva.virginia.gov/public/dvb/bio.asp?b=Ashe_Arthur_Robert" },
  bp: { label: "BlackPast: Arthur Ashe (1943–1993)", url: "https://blackpast.org/african-american-history/ashe-arthur-1943-1993/" },
  ithf: { label: "International Tennis Hall of Fame: Arthur Ashe", url: "https://www.tennisfame.com/hall-of-famers/inductees/arthur-ashe" },
  atp: { label: "ATP Tour: Arthur Ashe bio", url: "https://www.atptour.com/en/players/arthur-ashe/a063/bio" },
  atpTitles: { label: "ATP Tour: Arthur Ashe titles and finals", url: "https://www.atptour.com/en/players/arthur-ashe/a063/titles-and-finals" },
  wp: { label: "Wikipedia: Arthur Ashe", url: "https://en.wikipedia.org/wiki/Arthur_Ashe" },
  stl: { label: "St. Louis Sports Hall of Fame: Arthur Ashe", url: "https://www.stlshof.com/arthur-ashe/" },
  jrank: { label: "JRank: Ashe's early lessons", url: "https://sports.jrank.org/pages/186/Ashe-Arthur-Early-Lessons.html" },
  uclaHof: { label: "UCLA Athletics Hall of Fame: Arthur Ashe", url: "https://uclabruins.com/sports/hall-of-fame/roster/season/hof/player/arthur-ashe" },
  ncaa65: { label: "Wikipedia: 1965 NCAA tennis championships", url: "https://en.wikipedia.org/wiki/1965_NCAA_University_Division_tennis_championships" },
  a1966: { label: "Wikipedia: 1966 Australian Championships men's singles", url: "https://en.wikipedia.org/wiki/1966_Australian_Championships_%E2%80%93_Men%27s_singles" },
  a1967: { label: "Wikipedia: 1967 Australian Championships men's singles", url: "https://en.wikipedia.org/wiki/1967_Australian_Championships_%E2%80%93_Men%27s_singles" },
  vt: { label: "Veteran Tributes: Arthur R. Ashe", url: "http://veterantributes.org/TributeDetail.php?recordID=2478" },
  johnnie: { label: "UCLA Newsroom: Brother of Arthur Ashe made decision that kept him out of war", url: "https://newsroom.ucla.edu/stories/brother-of-arthur-ashe-recalls-decision-that-kept-tennis-legend-out-of-war" },
  nypl: { label: "New York Public Library: Arthur Ashe archive", url: "https://archives.nypl.org/scm/20922" },
  u1968: { label: "Wikipedia: 1968 US Open men's singles", url: "https://en.wikipedia.org/wiki/1968_US_Open_%E2%80%93_Men%27s_singles" },
  npr: { label: "NPR Code Switch: In 1968, Arthur Ashe made history at the US Open", url: "https://www.npr.org/sections/codeswitch/2018/09/10/646213901/in-1968-arthur-ashe-made-history-at-the-u-s-open" },
  uso50: { label: "USOpen.org: 50 for 50, Arthur Ashe", url: "https://www.usopen.org/en_US/news/articles/2018-01-16/2018-01-16_50_for_50_arthur_ashe_1968_mens_singles_champion.html" },
  mt: { label: "Mississippi Today: 1968, Arthur Ashe wins US Open", url: "https://mississippitoday.org/2023/09/09/on-this-day-in-1968-arthur-ashe-won-us-open-tennis-championship/" },
  wtm68: { label: "World Tennis Magazine: Ashe's historic 1968 US Open win", url: "https://worldtennismagazine.com/arthur-ashes-historic-five-set-1968-u-s-open-win-was-just-the-start-of-his-tennis-marathon/26054" },
  dc68: { label: "Wikipedia: 1968 Davis Cup", url: "https://en.wikipedia.org/wiki/1968_Davis_Cup" },
  mlk: { label: "Stanford King Institute: Assassination of Martin Luther King Jr.", url: "https://kinginstitute.stanford.edu/assassination-martin-luther-king-jr" },
  ucla: { label: "UCLA College: Arthur Ashe, champion for justice", url: "https://www.college.ucla.edu/2022/02/09/arthur-ashe-champion-for-justice" },
  saho: { label: "South African History Online: South Africa banned from the Davis Cup", url: "https://sahistory.org.za/dated-event/south-africa-banned-competing-davis-cup-because-its-apartheid-policy-sport" },
  sa73: { label: "Wikipedia: 1973 South African Open", url: "https://en.wikipedia.org/wiki/1973_South_African_Open_(tennis)" },
  mandela: { label: "South African History Online: Mandela released, 11 February 1990", url: "https://sahistory.org.za/content/nelson-mandela-released-prison-11-february-1990" },
  a1970: { label: "Wikipedia: 1970 Australian Open men's singles", url: "https://en.wikipedia.org/wiki/1970_Australian_Open_%E2%80%93_Men%27s_singles" },
  a1971: { label: "Wikipedia: 1971 Australian Open men's singles", url: "https://en.wikipedia.org/wiki/1971_Australian_Open_%E2%80%93_Men%27s_singles" },
  u1972: { label: "Wikipedia: 1972 US Open men's singles", url: "https://en.wikipedia.org/wiki/1972_US_Open_%E2%80%93_Men%27s_singles" },
  atpOrg: { label: "Wikipedia: Association of Tennis Professionals", url: "https://en.wikipedia.org/wiki/Association_of_Tennis_Professionals" },
  boycott: { label: "Tennis.com: 1973, the men boycott Wimbledon", url: "https://www.tennis.com/news/articles/1973-the-men-boycott-wimbledon-and-shift-power-to-the-players" },
  tmBoycott: { label: "Tennis Majors: The day Wimbledon began with a boycott", url: "https://www.tennismajors.com/wimbledon-news/june-25-1973-the-day-wimbledon-started-without-the-top-male-players-boycotting-the-tournament-267502.html" },
  wct75: { label: "Wikipedia: 1975 WCT Finals singles", url: "https://en.wikipedia.org/wiki/1975_World_Championship_Tennis_Finals_%E2%80%93_Singles" },
  atp1975: { label: "ATP Heritage: Ashe's Wimbledon win over Connors", url: "https://www.atptour.com/en/news/atp-heritage-ashe-connors-1975-wimbledon" },
  espn1975: { label: "ESPN: Remembering Arthur Ashe's historic 1975 Wimbledon title", url: "https://www.espn.com/tennis/story/_/id/45724971/arthur-ashe-wimbledon-1975-chris-eubanks-stan-smith-richard-evans" },
  wapo1975: { label: "Washington Post: How Arthur Ashe beat Jimmy Connors (2025)", url: "https://www.washingtonpost.com/sports/2025/07/04/arthur-ashe-wimbledon-black-champion/" },
  tc1975: { label: "Tennis.com: 1975, Ashe beats a seemingly invincible Connors", url: "https://www.tennis.com/news/articles/1975-beating-seemingly-invincible-jimmy-connors-arthur-ashe-thought-courage-ali" },
  rankings: { label: "ATP: Nastase becomes the first No. 1 of computer rankings", url: "https://www.atptour.com/en/news/nastase-number-one-club-rise" },
  nwe: { label: "New World Encyclopedia: Arthur Ashe", url: "https://www.newworldencyclopedia.org/entry/Arthur_Ashe" },
  jeanne: { label: "Wikipedia: Jeanne Moutoussamy-Ashe", url: "https://en.wikipedia.org/wiki/Jeanne_Moutoussamy-Ashe" },
  hm: { label: "The HistoryMakers: Jeanne Moutoussamy-Ashe", url: "https://www.thehistorymakers.org/biography/jeanne-moutoussamy-ashe-41" },
  cnn: { label: "CNN: Arthur Ashe retires from tennis, 16 April 1980", url: "https://www.cnn.com/2015/04/16/tennis/arthur-ashe-retires-tennis-april-1980/index.html" },
  tmDc: { label: "Tennis Majors: The day Ashe resigned as Davis Cup captain", url: "https://www.tennismajors.com/others-news/october-22-1985-the-day-arthur-ashe-resigned-as-usas-davis-cup-captain-299964.html" },
  mayo: { label: "Mayo Clinic Proceedings: Arthur Ashe Jr, tennis star and AIDS and urban health activist", url: "https://www.mayoclinicproceedings.org/article/S0025-6196(23)00006-X/fulltext" },
  upiAids: { label: "UPI (1992): Arthur Ashe confirms he has AIDS", url: "https://www.upi.com/Archives/1992/04/08/Arthur-Ashe-confirms-he-has-AIDS/8698702705600/" },
  stat: { label: "STAT News: Arthur Ashe and AIDS", url: "https://www.statnews.com/2022/06/26/arthur-ashe-and-aids-did-the-public-have-the-right-to-know-his-diagnosis/" },
  wapo1985: { label: "Washington Post (1985): Arthur Ashe jailed in apartheid protest", url: "https://www.washingtonpost.com/archive/local/1985/01/12/arthur-ashe-jailed-in-apartheid-protest/8a138f41-107c-475d-a8c8-9bd669fa3655/" },
  hrtg: { label: "A Hard Road to Glory (Warner Books, 1988): bibliographic record", url: "https://www.biblio.com/book/hard-road-glory-arthur-ashe/d/1583914138" },
  upiHaiti: { label: "UPI (1992): Arthur Ashe arrested in Haitian protest", url: "https://www.upi.com/Archives/1992/09/10/Arthur-Ashe-arrested-in-front-of-White-House-in-Haitian-protest/7069716097600" },
  desHaiti: { label: "Deseret News (1992): Ashe, other activists arrested", url: "https://www.deseret.com/1992/9/10/19004049/ashe-other-activists-arrested-after-protest/" },
  unPhoto: { label: "UN Photo: Ashe addresses the General Assembly on World AIDS Day", url: "https://media.un.org/photo/en/asset/oun7/oun7729384" },
  unPv: { label: "UN General Assembly record A/47/PV.76", url: "https://digitallibrary.un.org/record/157114/files/A_47_PV.76-EN.pdf" },
  prh: { label: "Penguin Random House: Days of Grace", url: "https://www.penguinrandomhouse.com/books/5414/days-of-grace-by-arthur-ashe-arnold-rampersad/" },
  legacy: { label: "Legacy.com: Arthur Ashe, a timeline of triumph", url: "https://www.legacy.com/news/arthur-ashe-a-timeline-of-triumph" },
  smith: { label: "Smithsonian: The death of a sports legend changed how Americans viewed AIDS", url: "https://www.smithsonianmag.com/smart-news/the-death-of-a-sports-legend-on-this-day-in-1993-changed-how-americans-viewed-aids-180985966/" },
  magic: { label: "HISTORY: Magic Johnson announces he is HIV-positive", url: "https://www.history.com/this-day-in-history/november-7/magic-johnson-announces-he-is-hiv-positive" },
  medal: { label: "Clinton White House archive: Medal of Freedom to Arthur Ashe", url: "https://clintonwhitehouse6.archives.gov/1993/06/1993-06-20-president-presents-medal-of-freedom-to-arthur-ashe.html" },
  espyAward: { label: "ESPN: Arthur Ashe Courage Award winners", url: "https://www.espn.com/espys/story/_/id/49350742/who-won-arthur-ashe-courage-award-espys-winners-year" },
  hmdb: { label: "HMDB: Arthur Ashe Monument", url: "https://www.hmdb.org/m.asp?m=22823" },
  valentine: { label: "The Valentine: Monument Avenue, Arthur Ashe Monument", url: "https://thevalentine.org/explore/richmond-stories/featured-stories/monument-avenue-arthur-ashe-monument/" },
  usoAshe: { label: "USOpen.org: 25 years of Arthur Ashe Stadium", url: "https://www.usopen.org/en_US/news/articles/2022-06-07/25_years_of_arthur_ashe_stadium_a_very_grand_opening_1997.html" },
  tmAshe: { label: "Tennis Majors: The day the US Open opened Arthur Ashe Stadium", url: "https://www.tennismajors.com/us-open-news/august-25-1997-day-us-open-officially-opened-arthur-ashe-stadium-court-282594.html" },
  dc1992: { label: "Wikipedia: 1992 Davis Cup", url: "https://en.wikipedia.org/wiki/1992_Davis_Cup" },
  mow: { label: "National Archives: The March on Washington for Jobs and Freedom", url: "https://www.archives.gov/legislative/features/march-on-washington" },
  cra: { label: "Miller Center: The Civil Rights Act of 1964", url: "https://millercenter.org/the-presidency/educational-resources/the-civil-rights-act-of-1964" },
  ta128: { label: "Tennis Abstract: The Tennis 128, No. 48 Arthur Ashe", url: "https://www.tennisabstract.com/blog/2022/09/14/the-tennis-128-no-48-arthur-ashe/" },
  gibson: { label: "Wikipedia: Robert Walter Johnson", url: "https://en.wikipedia.org/wiki/Robert_Walter_Johnson" },
} satisfies Record<string, Source>;

const ashe: Player = {
  slug: "ashe",
  name: "Arthur Ashe",
  shortName: "Ashe",
  born: { date: "10 July 1943", place: "Richmond, Virginia" },
  died: { date: "6 February 1993", place: "New York City" },
  country: "United States",
  lifespan: "1943 – 1993",
  identity: "Three majors, each a first for a Black man, and a life spent fighting for more than titles.",
  theme: "ashe",
  eraLabel: "The 1970s",
  heroFacts: [
    { label: "Majors", value: "3" },
    { label: "US Open", value: "1968" },
    { label: "Australian Open", value: "1970" },
    { label: "Wimbledon", value: "1975" },
  ],
  portrait: {
    monogram: "AA",
    alt: "Photograph of Arthur Ashe",
    commons: {
      files: [
        "File:Arthur Ashe (cropped).jpg",
        "File:ABN-wereldtennistoernooi in Rotterdam Arthur Ashe in actie, Bestanddeelnr 927-7839.jpg",
      ],
      category: "Category:Arthur Ashe",
      exclude: "stadium|court|statue|monument|stamp|arena|grave|logo",
      era: [1968, 1980],
    },
  },

  journey: [
    {
      set: 1,
      title: "Brook Field",
      years: [1943, 1950],
      headline: "A playground house in segregated Richmond",
      paragraphs: [
        "Arthur Robert Ashe Jr. was born on 10 July 1943 in Richmond, Virginia, to Mattie Cunningham Ashe and Arthur Robert Ashe Sr. They were a middle-class Black family in a strictly segregated city.",
        "When Arthur was four, the family moved into a house at Brook Field, Richmond's largest recreational facility for Black residents, where his father was the supervisor.",
        "When he was six, his mother died, aged 28, of a stroke caused by undiagnosed hypertensive vascular disease.",
      ],
      tile: { label: "Born", value: "10·7·43", caption: "Richmond, Virginia" },
      eraNote: {
        text: "Richmond's parks were segregated by race. Brook Field, where the Ashes lived, was the city's largest recreation ground for African Americans, and those were the courts he learned on.",
        sources: [S.ev],
      },
      sources: [S.ev, S.lva, S.bp],
    },
    {
      set: 2,
      title: "Lessons",
      years: [1950, 1960],
      headline: "A student's serve, and a doctor's summer camp",
      paragraphs: [
        "About a year after his mother's death, Arthur was watching Ronald Charity, a student at nearby Virginia Union University, practise his serve on the Brook Field courts. Charity became his first teacher.",
        "From 1953, when he was 10, until 1960, Ashe spent his summers at the Lynchburg home and tennis camp of Dr. Robert Walter Johnson, a Black physician and coach who had earlier developed Althea Gibson.",
      ],
      tile: { label: "Summers at Dr. Johnson's camp", value: "1953–60", caption: "Lynchburg, Virginia" },
      eraNote: {
        text: "Johnson's earlier pupil, Althea Gibson, won Wimbledon in 1957 and 1958, the first Black player to do so.",
        sources: [S.ev, S.gibson],
      },
      sources: [S.ev, S.lva, S.atp, S.gibson],
    },
    {
      set: 3,
      title: "St. Louis",
      years: [1960, 1961],
      headline: "A senior year away from home",
      paragraphs: [
        "In 1960, after three years at Maggie L. Walker High School in Richmond, Ashe accepted an offer from Richard Hudlin, a tennis coach in St. Louis, to move there for his senior year. He finished school at Sumner High.",
        "In 1961, after Dr. Johnson lobbied for his entry to the previously segregated US Interscholastic tournament, Ashe won it for Sumner. A tennis scholarship to UCLA followed.",
      ],
      tile: { label: "US Interscholastic champion", value: "1961", caption: "For Sumner High, St. Louis" },
      eraNote: {
        text: "The Interscholastic had been closed to Black players until Johnson's lobbying got Ashe in.",
        sources: [S.jrank, S.ev],
      },
      sources: [S.ev, S.stl, S.jrank],
    },
    {
      set: 4,
      title: "Bruin",
      years: [1961, 1966],
      headline: "UCLA, a national title, and the Davis Cup",
      paragraphs: [
        "At UCLA, Ashe was coached by J. D. Morgan and guided by the tennis great Pancho Gonzales. In 1963 he became the first Black player named to the United States Davis Cup team.",
        "In 1965 he won the NCAA singles title, beating Mike Belkin of Miami in the final, and teamed with Ian Crookenden to win the doubles. UCLA won the team championship. In 1966 he reached his first major final, at the Australian Championships, and lost to Roy Emerson in four sets.",
      ],
      matches: [
        {
          year: 1965, event: "NCAA Championships", round: "Final", opponent: "Mike Belkin", score: "6–4, 6–1, 6–1", result: "W",
          why: "National collegiate champion, and UCLA took the team title.", sources: [S.ev, S.ncaa65],
        },
        {
          year: 1966, event: "Australian Championships", round: "Final", opponent: "Roy Emerson", score: "4–6, 8–6, 2–6, 3–6", result: "L",
          why: "His first major final.", sources: [S.a1966],
        },
      ],
      tile: { label: "First Black player on the US Davis Cup team", value: "1963" },
      eraNote: {
        text: "In the summer Ashe joined the Davis Cup team, more than 250,000 people joined the March on Washington (28 August 1963). The Civil Rights Act was signed on 2 July 1964.",
        sources: [S.mow, S.cra],
      },
      sources: [S.ev, S.ithf, S.uclaHof, S.ncaa65, S.a1966],
    },
    {
      set: 5,
      title: "Lieutenant",
      years: [1966, 1968],
      headline: "An Army officer with a world-class serve",
      paragraphs: [
        "Ashe entered the US Army on 4 August 1966 and was commissioned a second lieutenant in the Adjutant General's Corps. He was posted to the US Military Academy at West Point, where he worked in data processing, and he was promoted to first lieutenant in February 1968.",
        "His brother Johnnie, a Marine, volunteered for a second tour of duty in Vietnam. According to UCLA's account of the brothers' story, that decision is what kept Arthur from being sent there himself. Ashe kept competing: in 1967 he was again runner-up to Emerson at the Australian Championships.",
      ],
      matches: [
        {
          year: 1967, event: "Australian Championships", round: "Final", opponent: "Roy Emerson", score: "4–6, 1–6, 4–6", result: "L",
          why: "A second straight Australian final.", sources: [S.a1967],
        },
      ],
      tile: { label: "His rank when he won the US Open", value: "1st Lt.", caption: "US Army" },
      eraNote: {
        text: "The war in Vietnam shaped the Ashe family directly. Johnnie Ashe's second tour there kept his younger brother at West Point and on the tennis court.",
        sources: [S.johnnie],
      },
      sources: [S.vt, S.johnnie, S.nypl, S.a1967],
    },
    {
      set: 6,
      title: "Forest Hills",
      years: [1968, 1968],
      headline: "The first Open US Championships, won by an amateur",
      paragraphs: [
        "In the summer of 1968 Ashe won the US Amateur Championships at the Longwood Cricket Club in Boston, beating his teammate Bob Lutz in five sets. Weeks later, at the first US Open, he beat Tom Okker of the Netherlands 14–12, 5–7, 6–3, 3–6, 6–3 in the final. He was the first Black man to win a men's singles major.",
        "Because Ashe was still registered as an amateur, and was serving as an Army lieutenant, he could not accept the $14,000 first prize. It went to Okker, the runner-up. Ashe received only an expense allowance. He came into the final on a long winning streak, put at 24 matches by one account and 26 by another.",
        "In December, as the top US singles player, he helped the United States win the Davis Cup, beating the holders Australia in the Challenge Round in Adelaide from 26 to 28 December 1968.",
      ],
      matches: [
        {
          year: 1968, event: "US Amateur Championships", round: "Final", opponent: "Bob Lutz", score: "4–6, 6–3, 8–10, 6–0, 6–4", result: "W",
          why: "Earned him a seeding at the first US Open.", sources: [S.ithf],
        },
        {
          year: 1968, event: "US Open", round: "Final", opponent: "Tom Okker", score: "14–12, 5–7, 6–3, 3–6, 6–3", result: "W",
          why: "The first major singles title won by a Black man.", sources: [S.u1968, S.npr, S.uso50],
        },
      ],
      tile: { label: "Prize money Ashe could take for winning the US Open", value: "$0", caption: "The $14,000 first prize went to Okker" },
      eraNote: {
        text: "Five months before Ashe's US Open win, Martin Luther King Jr. was assassinated in Memphis on 4 April 1968.",
        sources: [S.mlk],
      },
      sources: [S.u1968, S.npr, S.uso50, S.ithf, S.mt, S.wtm68, S.dc68],
    },
    {
      set: 7,
      title: "The Wall",
      years: [1969, 1973],
      headline: "Refused by apartheid South Africa, then let in on his terms",
      paragraphs: [
        "In 1969, his Army service over, Ashe turned professional and applied for a visa to play in South Africa. He was refused. He applied again in 1970 and was refused again. South Africa's sport minister, Frank Waring, cited Ashe's “general antagonism” towards the country. Ashe had said his trip would be an attempt “to put a crack in the racist wall down there.”",
        "In March 1970, amid protests over the government's apartheid policies, South Africa was expelled from the Davis Cup.",
        "In 1973 he was finally granted a visa and, after consulting political and cultural leaders, he went. He became the first Black professional to play the South African national championships at Ellis Park in Johannesburg, and he insisted that the seating be unsegregated. He reached the singles final and won the doubles with Tom Okker.",
      ],
      matches: [
        {
          year: 1973, event: "South African Open", round: "Final", opponent: "Jimmy Connors", score: "4–6, 6–7, 3–6", result: "L",
          why: "A final played in front of unsegregated seating, at Ashe's insistence.", sources: [S.sa73, S.nypl],
        },
        {
          year: 1973, event: "South African Open", round: "Doubles final (with Tom Okker)", opponent: "Lew Hoad / Robert Maud", score: "6–2, 4–6, 6–2, 6–4", result: "W",
          why: "A title in Johannesburg for the player the government had twice refused.", sources: [S.sa73],
        },
      ],
      tile: { label: "Visa refusals before South Africa let him in", value: "2", caption: "1969 and 1970" },
      eraNote: {
        text: "Nelson Mandela was in prison throughout. He was not released until 11 February 1990.",
        sources: [S.mandela],
      },
      sources: [S.nypl, S.ucla, S.saho, S.sa73],
    },
    {
      set: 8,
      title: "Melbourne & the Union",
      years: [1970, 1974],
      headline: "A second major, and a founding role in the players' union",
      paragraphs: [
        "In 1970 Ashe won the Australian Open, beating Dick Crealy 6–4, 9–7, 6–2 for his second major title. He returned to the final the next year and lost to Ken Rosewall, and he lost a five-set US Open final to Ilie Năstase in 1972.",
        "In September 1972 Ashe was among the players who formed the Association of Tennis Professionals, a union to represent their interests. Cliff Drysdale was its first president; Ashe served as president in 1974.",
      ],
      matches: [
        {
          year: 1970, event: "Australian Open", round: "Final", opponent: "Dick Crealy", score: "6–4, 9–7, 6–2", result: "W",
          why: "Major No. 2, the first Australian title won by a Black man.", sources: [S.a1970, S.ithf],
        },
        {
          year: 1971, event: "Australian Open", round: "Final", opponent: "Ken Rosewall", score: "1–6, 5–7, 3–6", result: "L",
          why: "Lost his title defence in the final.", sources: [S.a1971],
        },
        {
          year: 1972, event: "US Open", round: "Final", opponent: "Ilie Năstase", score: "6–3, 3–6, 7–6, 4–6, 3–6", result: "L",
          why: "Came within a set of a second US title.", sources: [S.u1972],
        },
      ],
      tile: { label: "ATP founded", value: "1972", caption: "Ashe was president in 1974" },
      eraNote: {
        text: "The new union flexed its muscle fast. In 1973, 81 ATP players, including 13 of the 16 seeds, boycotted Wimbledon over the suspension of Nikola Pilić.",
        sources: [S.boycott, S.tmBoycott],
      },
      sources: [S.a1970, S.a1971, S.u1972, S.ev, S.atpOrg],
    },
    {
      set: 9,
      title: "The Masterpiece",
      years: [1975, 1975],
      headline: "Taking the pace off, and taking Wimbledon",
      paragraphs: [
        "1975 was Ashe's finest year. He won the WCT Finals in Dallas, beating Björn Borg 3–6, 6–4, 6–4, 6–0. Then, on 5 July on Wimbledon's Centre Court, he beat Jimmy Connors, the world No. 1, 6–1, 6–1, 5–7, 6–4.",
        "Ashe was usually a big server and hitter, but against Connors he did the opposite. He took the pace off the ball, sliced low and short, and denied Connors the speed he liked to feed on. The final is widely regarded as a tactical masterpiece and one of Wimbledon's great upsets.",
        "He was the first Black man to win the Wimbledon singles title. Althea Gibson, in 1957, had been the first Black champion there.",
      ],
      matches: [
        {
          year: 1975, event: "WCT Finals, Dallas", round: "Final", opponent: "Björn Borg", score: "3–6, 6–4, 6–4, 6–0", result: "W",
          why: "Beat the rising Swede for the tour's showpiece title.", sources: [S.wct75],
        },
        {
          year: 1975, event: "Wimbledon", round: "Final", opponent: "Jimmy Connors", score: "6–1, 6–1, 5–7, 6–4", result: "W",
          why: "Major No. 3, the first Wimbledon title won by a Black man.", sources: [S.atp1975, S.espn1975, S.wapo1975, S.tc1975],
        },
      ],
      tile: { label: "Wimbledon final, 5 July 1975", value: "6–1 6–1 5–7 6–4", caption: "d. Jimmy Connors" },
      eraNote: {
        text: "By 1975 the ATP's computer rankings, which began in August 1973, were the official measure. Ashe's ATP peak was No. 2, in May 1976, though some year-end rankings for 1975 put him No. 1.",
        sources: [S.rankings, S.atp, S.nwe],
      },
      sources: [S.atp1975, S.espn1975, S.wapo1975, S.tc1975, S.wct75],
    },
    {
      set: 10,
      title: "Heart",
      years: [1977, 1985],
      headline: "A heart attack, a captaincy, and an arrest",
      paragraphs: [
        "On 20 February 1977 Ashe married the photographer Jeanne Moutoussamy in a chapel at the United Nations in New York. Andrew Young, the US Ambassador to the UN, officiated. On 31 July 1979 Ashe had a heart attack, and in December 1979 he underwent quadruple bypass surgery. Chest pains returned when he tried to train again, and on 16 April 1980 he retired from competition.",
        "From 1981 to 1985 he captained the US Davis Cup team, which won the Cup in 1981 and 1982. He resigned on 22 October 1985. In 1983 he needed a second bypass operation. Doctors later concluded it was most likely then that he was infected with HIV, through a blood transfusion.",
        "On 11 January 1985 Ashe was arrested at an anti-apartheid protest outside the South African embassy in Washington. That year he was elected to the International Tennis Hall of Fame.",
      ],
      tile: { label: "Davis Cups won as US captain", value: "2", caption: "1981 and 1982" },
      eraNote: {
        text: "Ashe's 1985 arrest came as protests outside South Africa's embassy in Washington were building international attention on apartheid.",
        sources: [S.wapo1985],
      },
      sources: [S.jeanne, S.hm, S.ithf, S.lva, S.cnn, S.tmDc, S.atp, S.mayo, S.wapo1985],
    },
    {
      set: 11,
      title: "Days of Grace",
      years: [1986, 1993],
      headline: "Historian, patient, activist",
      paragraphs: [
        "In December 1986 Arthur and Jeanne adopted a daughter, Camera. In 1988 Ashe published A Hard Road to Glory, a three-volume history of the African-American athlete. The same year, tests before brain surgery revealed toxoplasmosis, a marker for AIDS, and he learned he was HIV-positive.",
        "He kept his diagnosis private until USA Today prepared to report it. On 8 April 1992 he announced it himself, on his own terms. He founded the Arthur Ashe Foundation for the Defeat of AIDS, and on 1 December 1992, World AIDS Day, he addressed the UN General Assembly. He also founded the Arthur Ashe Institute for Urban Health. On 9 September 1992 he was one of about 95 people arrested outside the White House protesting US treatment of Haitian refugees. Sports Illustrated named him its 1992 Sportsman of the Year.",
        "Arthur Ashe died of AIDS-related pneumonia at New York Hospital on 6 February 1993, aged 49. His memoir, Days of Grace, was published that year. In Richmond he lay in state at the Governor's Mansion, and thousands filed past.",
      ],
      tile: { label: "Addressed the UN General Assembly", value: "1·12·92", caption: "World AIDS Day" },
      eraNote: {
        text: "Five months before Ashe's announcement, basketball star Magic Johnson announced on 7 November 1991 that he was HIV-positive, one of the first sports stars to go public.",
        sources: [S.magic],
      },
      sources: [S.jeanne, S.hrtg, S.mayo, S.upiAids, S.stat, S.unPhoto, S.unPv, S.upiHaiti, S.desHaiti, S.ithf, S.bp, S.smith, S.prh, S.legacy],
    },
    {
      set: 12,
      title: "Legacy",
      years: [1993, "present"],
      headline: "A medal, a monument, a stadium, an award",
      paragraphs: [
        "On 20 June 1993 President Bill Clinton awarded Ashe the Presidential Medal of Freedom, posthumously. The same year ESPN's ESPYS began presenting the Arthur Ashe Courage Award, given to people whose contributions transcend sport.",
        "On 10 July 1996, what would have been his 53rd birthday, a statue of Ashe by the sculptor Paul DiPasquale was unveiled on Richmond's Monument Avenue. On 25 August 1997 the new main court of the US Open, Arthur Ashe Stadium, opened its gates.",
      ],
      tile: { label: "Arthur Ashe Stadium opens", value: "1997", caption: "Main court of the US Open" },
      eraNote: {
        text: "Ashe lived to see apartheid sport begin to end: in 1992 South Africa returned to the Davis Cup for the first time since 1978.",
        sources: [S.dc1992],
      },
      sources: [S.medal, S.espyAward, S.hmdb, S.valentine, S.usoAshe, S.tmAshe],
    },
  ],

  records: [
    {
      label: "Major singles titles",
      value: "3",
      countTo: 3,
      detail: "US Open 1968, Australian Open 1970, Wimbledon 1975.",
      status: "stat",
      sources: [S.ithf, S.atp],
    },
    {
      label: "First Black man to win the US Open",
      value: "1968",
      detail: "d. Tom Okker 14–12, 5–7, 6–3, 3–6, 6–3, as an amateur.",
      status: "first",
      sources: [S.u1968, S.npr],
    },
    {
      label: "First Black man to win the Australian Open",
      value: "1970",
      detail: "d. Dick Crealy 6–4, 9–7, 6–2.",
      status: "first",
      sources: [S.a1970, S.ithf],
    },
    {
      label: "First Black man to win Wimbledon",
      value: "1975",
      detail: "d. Jimmy Connors 6–1, 6–1, 5–7, 6–4.",
      status: "first",
      sources: [S.atp1975, S.espn1975],
    },
    {
      label: "First Black player on the US Davis Cup team",
      value: "1963",
      detail: "He went on to captain the team to the Cup in 1981 and 1982.",
      status: "first",
      sources: [S.ithf, S.ev, S.tmDc],
    },
    {
      label: "US Open first prize, 1968 (went to Okker)",
      value: "$14,000",
      countTo: 14000,
      detail: "A record prize at the time, which Ashe could not accept as an amateur. It has been surpassed many times since.",
      status: "since-broken",
      sources: [S.u1968, S.npr],
    },
    {
      label: "Davis Cup singles record as a player",
      value: "27–5",
      detail: "Plus 1–1 in doubles.",
      status: "stat",
      sources: [S.atp],
    },
    {
      label: "Career singles titles",
      value: "33 / 44 / ~87",
      detail: "ATP's bio lists 33 Open Era titles; Wikipedia's reading of ATP data gives 44; totals including amateur events reach about 87.",
      status: "sources-differ",
      sources: [S.atp, S.atpTitles, S.wp, S.ta128],
    },
  ],

  era: {
    onCourt: [
      {
        title: "Amateur in an open year",
        body: "When the US Championships went open in 1968, Ashe was still an amateur and an Army officer. He won the tournament but not the money: the $14,000 first prize went to runner-up Tom Okker.",
        sources: [S.u1968, S.npr],
      },
      {
        title: "Players organise",
        body: "The ATP was formed in September 1972. The next year 81 of its members boycotted Wimbledon over the suspension of Nikola Pilić, and in August 1973 the ATP published its first computer rankings.",
        sources: [S.atpOrg, S.boycott, S.rankings],
      },
      {
        title: "Sport and apartheid",
        body: "South Africa refused Ashe a visa in 1969 and 1970 and was expelled from the Davis Cup in March 1970. It returned to the competition in 1992.",
        sources: [S.nypl, S.saho, S.dc1992],
      },
      {
        title: "Style",
        body: "Ashe was known as a big server and hitter, which made his soft, sliced plan against Connors in the 1975 Wimbledon final all the more striking.",
        sources: [S.atp1975, S.tc1975],
      },
    ],
    inTheWorld: [
      {
        title: "1963–64",
        body: "The March on Washington (28 August 1963) and the Civil Rights Act (2 July 1964) bracketed Ashe's first Davis Cup selection.",
        sources: [S.mow, S.cra],
      },
      {
        title: "1968",
        body: "Martin Luther King Jr. was assassinated in Memphis on 4 April, five months before Ashe won the US Open.",
        sources: [S.mlk],
      },
      {
        title: "Vietnam",
        body: "His brother Johnnie served two tours in Vietnam; Arthur served at West Point.",
        sources: [S.johnnie],
      },
      {
        title: "1990–92",
        body: "Nelson Mandela was released on 11 February 1990. Magic Johnson announced he was HIV-positive on 7 November 1991.",
        sources: [S.mandela, S.magic],
      },
    ],
  },

  beyond: [
    {
      title: "1968 US Open, as an amateur",
      body: "Ashe won the first US Open while still an amateur and a serving Army lieutenant, so the winner's cheque went to the man he beat.",
      sources: [S.u1968, S.npr, S.uso50],
    },
    {
      title: "Against apartheid",
      body: "He sought to play in South Africa to “put a crack in the racist wall”, was refused twice, went in 1973 only with unsegregated seating, and was arrested outside South Africa's embassy in Washington in 1985.",
      sources: [S.nypl, S.ucla, S.sa73, S.wapo1985],
    },
    {
      title: "A voice for players",
      body: "A founding member of the ATP in 1972 and its president in 1974.",
      sources: [S.ev, S.atpOrg],
    },
    {
      title: "HIV/AIDS advocacy",
      body: "After making his diagnosis public in April 1992, he founded the Arthur Ashe Foundation for the Defeat of AIDS and the Arthur Ashe Institute for Urban Health, and addressed the UN General Assembly on World AIDS Day.",
      sources: [S.upiAids, S.mayo, S.unPhoto],
    },
    {
      title: "Historian",
      body: "His three-volume A Hard Road to Glory (1988) is a history of the African-American athlete.",
      sources: [S.hrtg],
    },
    {
      title: "Haitian refugees",
      body: "In September 1992, months before his death, he was arrested outside the White House protesting US policy towards Haitian refugees.",
      sources: [S.upiHaiti, S.desHaiti],
    },
    {
      title: "Arthur Ashe Stadium",
      body: "The US Open's main court has carried his name since it opened on 25 August 1997.",
      sources: [S.usoAshe, S.tmAshe],
    },
  ],

  legacy: [
    "Ashe's three majors were each a first for a Black man, and they came in three different kinds of tennis: as an amateur in the first open US Championships, as a new professional in Melbourne, and as a 31-year-old tactician at Wimbledon.",
    "His larger legacy was built off the court: pressing apartheid South Africa, helping found the players' union, writing the history of Black athletes in America, and, in the last year of his life, talking openly about AIDS. The Presidential Medal of Freedom, the Monument Avenue statue, the Courage Award and the stadium in Queens all honour that wider life.",
  ],

  sourcesDiffer: [
    {
      title: "How many titles?",
      body: "ATP's bio lists 33 Open Era singles titles. Wikipedia, reading ATP data, gives 44. Counts that include amateur events reach about 87.",
      sources: [S.atp, S.wp, S.ta128],
    },
    {
      title: "World No. 1?",
      body: "Some sources describe Ashe as world No. 1 in 1975. His highest ATP computer ranking was No. 2, on 10 May 1976.",
      sources: [S.atp, S.nwe],
    },
    {
      title: "The 1968 winning streak",
      body: "Accounts of his winning streak into the 1968 US Open final range from 24 to 26 matches.",
      sources: [S.mt, S.wtm68],
    },
    {
      title: "Why South Africa left the Davis Cup",
      body: "Some accounts tie the March 1970 expulsion directly to Ashe's visa refusal; others describe it as a response to apartheid sport policy more broadly. We present it as part of the wider protest.",
      sources: [S.saho, S.nypl],
    },
  ],

  references: [S.ev, S.lva, S.ithf, S.atp, S.nypl, S.bp, S.wp, S.mayo],
};

export default ashe;
