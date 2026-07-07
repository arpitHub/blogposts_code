// 2D side-view ball-flight simulation used by the Spin & Ball Flight lab.
// Units: meters, seconds, radians internally. x=0 is the hitter's baseline,
// the net stands at 11.885m, the far baseline at 23.77m.

export const COURT = {
  length: 23.77,
  netX: 11.885,
  netHeight: 0.914,
  serviceLineX: 11.885 + 6.4, // far service line
}

const G = 9.81
const KD = 0.017 // drag: a = -KD * |v| * v
const KM = 0.00058 // Magnus: a = KM * omega * (vy, -vx); +omega = topspin

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

/**
 * Simulate one shot.
 * @param {number} speedKmh ball speed off the racket
 * @param {number} launchDeg launch angle above horizontal
 * @param {number} rpm ball spin; positive = topspin, negative = slice
 * @returns trajectory points (including one bounce) plus derived stats
 */
export function simulateShot({ speedKmh, launchDeg, rpm, contactHeight = 1.0 }) {
  const v0 = speedKmh / 3.6
  const a0 = (launchDeg * Math.PI) / 180
  let vx = v0 * Math.cos(a0)
  let vy = v0 * Math.sin(a0)
  let x = 0
  let y = contactHeight
  let omega = (rpm * 2 * Math.PI) / 60
  let t = 0
  const dt = 0.004

  const points = [{ x, y, t }]
  let netClearance = null
  let hitNet = false
  let landingX = null
  let bounceCount = 0
  let apex = contactHeight
  let bounceApex = 0
  let landingSpeed = null

  while (t < 5 && x < 30 && bounceCount < 2) {
    const sp = Math.hypot(vx, vy)
    const ax = -KD * sp * vx + KM * omega * vy
    const ay = -G - KD * sp * vy - KM * omega * vx
    const prevX = x
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    t += dt

    // Net crossing
    if (prevX < COURT.netX && x >= COURT.netX && bounceCount === 0) {
      netClearance = y - COURT.netHeight
      if (y <= COURT.netHeight) {
        hitNet = true
        points.push({ x: COURT.netX, y: Math.max(y, 0), t })
        // let it drop to the ground at the net for a natural-looking end
        let dy = Math.max(y, 0.05)
        let tt = t
        while (dy > 0) {
          dy -= 2.2 * dt * 4
          tt += dt * 4
          points.push({ x: COURT.netX - 0.05, y: Math.max(dy, 0), t: tt })
        }
        break
      }
    }

    // Ground contact
    if (y <= 0) {
      y = 0
      bounceCount += 1
      if (bounceCount === 1) {
        landingX = x
        landingSpeed = sp
        const spinF = clamp(rpm / 3000, -1, 1)
        // Topspin kicks up and forward; slice stays low and slows down.
        vy = -vy * clamp(0.62 + 0.18 * spinF, 0.35, 0.85)
        vx = vx * clamp(0.68 + 0.22 * spinF, 0.4, 0.95)
        omega *= 0.55
      }
      points.push({ x, y, t })
      if (bounceCount >= 2) break
      continue
    }

    if (bounceCount === 0 && y > apex) apex = y
    if (bounceCount === 1 && y > bounceApex) bounceApex = y
    points.push({ x, y, t })
  }

  let verdict
  if (hitNet) verdict = 'net'
  else if (landingX === null) verdict = 'long'
  else if (landingX > COURT.length) verdict = 'long'
  else if (landingX <= COURT.netX) verdict = 'net' // landed on own side (shouldn't happen)
  else verdict = 'in'

  return {
    points,
    verdict,
    netClearance,
    landingX,
    landingSpeed,
    apex,
    bounceApex,
    duration: points[points.length - 1].t,
  }
}

/** Classify a spin rate into a human-readable shot type. */
export function classifySpin(rpm) {
  if (rpm >= 2400) return { label: 'Heavy topspin', tone: 'topspin' }
  if (rpm >= 900) return { label: 'Topspin', tone: 'topspin' }
  if (rpm > -600) return { label: 'Flat', tone: 'flat' }
  return { label: 'Slice (backspin)', tone: 'slice' }
}
