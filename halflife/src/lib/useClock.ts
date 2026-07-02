import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * A scrubbable simulation clock driven by requestAnimationFrame.
 * `speed` is sim-time units per real second and may change while running.
 */
export function useClock(speed: number, maxTime?: number) {
  const [time, setTime] = useState(0)
  const [running, setRunning] = useState(false)
  const speedRef = useRef(speed)
  speedRef.current = speed
  const maxRef = useRef(maxTime)
  maxRef.current = maxTime

  useEffect(() => {
    if (!running) return
    let id: number
    let prev = performance.now()
    const step = (now: number) => {
      const dt = Math.min((now - prev) / 1000, 0.1)
      prev = now
      setTime((t) => {
        const max = maxRef.current
        const nt = t + dt * speedRef.current
        return max !== undefined && nt >= max ? max : nt
      })
      id = requestAnimationFrame(step)
    }
    id = requestAnimationFrame(step)
    return () => cancelAnimationFrame(id)
  }, [running])

  // auto-stop at the end of the run
  useEffect(() => {
    if (maxTime !== undefined && time >= maxTime && running) setRunning(false)
  }, [time, maxTime, running])

  const reset = useCallback(() => {
    setRunning(false)
    setTime(0)
  }, [])

  return { time, setTime, running, setRunning, reset }
}
