import { useCallback, useEffect, useRef, useState } from "react";

export interface Fetched<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/** Fetch on mount (and whenever `fn` changes); optionally poll every `intervalMs`. */
export function useFetch<T>(fn: () => Promise<T>, intervalMs?: number): Fetched<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const load = useCallback(() => {
    fn()
      .then((value) => {
        if (!alive.current) return;
        setData(value);
        setError(null);
        setLoading(false);
      })
      .catch((err: Error) => {
        if (!alive.current) return;
        setError(err.message);
        setLoading(false);
      });
  }, [fn]);

  useEffect(() => {
    alive.current = true;
    setLoading(true);
    load();
    const timer = intervalMs ? window.setInterval(load, intervalMs) : undefined;
    return () => {
      alive.current = false;
      if (timer) window.clearInterval(timer);
    };
  }, [load, intervalMs]);

  return { data, loading, error, refresh: load };
}
