import { useState, useEffect, useRef } from 'react';
import type { TraversalStatus } from '../lib/types';

/**
 * Hook to track elapsed time during traversal.
 * Starts timer when status transitions to 'traversing'.
 * Stops timer when status transitions to 'complete' or 'error'.
 * Resets when status returns to 'idle'.
 */
export function useElapsedTime(status: TraversalStatus): number | null {
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const prevStatusRef = useRef<TraversalStatus>('idle');

  useEffect(() => {
    // Started: any status -> traversing (handles re-runs from complete/error)
    if (prevStatusRef.current !== 'traversing' && status === 'traversing') {
      startTimeRef.current = Date.now();
      setElapsedTime(null);
    }

    // Finished: traversing -> complete or error
    if (prevStatusRef.current === 'traversing' && (status === 'complete' || status === 'error')) {
      if (startTimeRef.current) {
        setElapsedTime(Date.now() - startTimeRef.current);
      }
    }

    // Reset when going back to idle
    if (status === 'idle') {
      startTimeRef.current = null;
      setElapsedTime(null);
    }

    prevStatusRef.current = status;
  }, [status]);

  return elapsedTime;
}