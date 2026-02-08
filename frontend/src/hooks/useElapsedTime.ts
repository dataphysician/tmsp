import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Hook to track elapsed time with active ticking.
 * Starts timer when `running` transitions false → true.
 * Stops and freezes when `running` transitions true → false.
 * Returns [elapsedTime, reset]:
 *   - elapsedTime: null when not yet started, elapsed ms otherwise
 *   - reset: clears elapsed time back to null
 */
export function useElapsedTime(running: boolean): [number | null, () => void] {
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const prevRunningRef = useRef(false);

  useEffect(() => {
    // Start: false → true
    if (!prevRunningRef.current && running) {
      startTimeRef.current = Date.now();
      setElapsedTime(0);
    }

    // Stop: true → false — freeze at final value
    if (prevRunningRef.current && !running) {
      if (startTimeRef.current) {
        setElapsedTime(Date.now() - startTimeRef.current);
      }
      startTimeRef.current = null;
    }

    prevRunningRef.current = running;
  }, [running]);

  // Active ticking while running
  useEffect(() => {
    if (!running || !startTimeRef.current) return;

    const interval = setInterval(() => {
      if (startTimeRef.current) {
        setElapsedTime(Date.now() - startTimeRef.current);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [running]);

  const reset = useCallback(() => {
    setElapsedTime(null);
    startTimeRef.current = null;
  }, []);

  return [elapsedTime, reset];
}
