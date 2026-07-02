# HalfLife

An interactive, single-page explainer of **radiometric dating and the discovery
of deep time** — how we came to know the Earth is 4.55 billion years old by
counting atoms.

Every section has a persistent **"Explain Like I'm…" depth toggle** (Beginner /
Technical). Both modes share the same visuals; only the accompanying text,
labels, and equations change.

## The seven concepts

1. **Radioactive decay, animated** — a jar of ~200 atoms flips parent→daughter
   at a rate set by a half-life slider, with a live parent-fraction decay curve
   and the analytic `N(t) = N₀e^(−λt)` overlay in Technical mode.
2. **The discovery story** — a scrubbable 1896–1956 timeline (Becquerel → the
   Curies → Rutherford → Patterson), each stop with a small animated
   recreation of the key observation.
3. **Isotopes & decay chains** — the U-238, U-235 and Th-232 chains play out as
   a relay of ~11–14 isotopes, each node holding the atom for its (log-mapped)
   half-life, from microseconds to billions of years.
4. **Different clocks for different timescales** — three jars (Carbon-14,
   Potassium-Argon, Uranium-Lead) on a shared logarithmic time slider, showing
   why the clock you pick depends on what you're dating.
5. **How Patterson dated the Earth** — the Pb–Pb meteorite isochron rebuilt as a
   physical simulation: points grow out of primordial lead onto one line whose
   slope reads 4.55 Gyr. Drag a point to see open-system scatter break the fit.
6. **Helioseismology cross-check** — the Sun's acoustic oscillations as an
   independent clock landing on ~4.6 Gyr; two unrelated methods agree.
7. **Deep time, on one scale** — every marker anchored by a named dating method,
   with a linear/log axis toggle in Technical mode.

The **jar-of-particles + decay-curve** visual language from Section 1 is reused
in Sections 3 and 4 so the mental model compounds. Color is consistent
throughout: **amber/gold = ancient / parent isotope**, **cool blue = recent /
daughter isotope**. Chart series colors are validated colorblind-safe against
the near-black surface.

## Stack

React 18 · Vite · TypeScript · Tailwind CSS v4 · Framer Motion · Recharts.
Everything runs client-side with real published half-lives, isochron data, and
ages; the particle simulations are pedagogical (a few hundred atoms, seeded and
deterministic, not 10²³).

## Develop

```bash
npm install
npm run dev      # start the dev server
npm run build    # type-check + production build to dist/
npm run preview  # serve the production build
```
