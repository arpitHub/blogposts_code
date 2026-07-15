import { useEffect, useState } from "react";

// Generic localStorage-backed state. Falls back to `initialValue`
// when the key is missing or unreadable (e.g. private browsing).
export default function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage unavailable — state still works for the session.
    }
  }, [key, value]);

  return [value, setValue];
}
