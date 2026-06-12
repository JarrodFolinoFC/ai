import { useEffect, useRef, useState } from 'react';

import { ApiError } from '../api';

export interface DebouncedApiResult<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/**
 * Calls `fetcher(input)` 200ms after `input` last changes.
 * Keeps the previous successful `data` while loading or while the latest call errors,
 * so the UI doesn't flicker between successful states.
 */
export function useDebouncedApi<TInput, TOutput>(
  input: TInput,
  fetcher: (input: TInput) => Promise<TOutput>,
  delayMs = 200
): DebouncedApiResult<TOutput> {
  const [data, setData] = useState<TOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const reqIdRef = useRef(0);

  useEffect(() => {
    const myId = ++reqIdRef.current;
    setLoading(true);
    const handle = setTimeout(() => {
      fetcher(input)
        .then((res) => {
          if (myId !== reqIdRef.current) return;
          setData(res);
          setError(null);
        })
        .catch((e: unknown) => {
          if (myId !== reqIdRef.current) return;
          if (e instanceof ApiError) setError(e.detail);
          else if (e instanceof Error) setError(e.message);
          else setError('unknown error');
        })
        .finally(() => {
          if (myId === reqIdRef.current) setLoading(false);
        });
    }, delayMs);
    return () => clearTimeout(handle);
  }, [input, fetcher, delayMs]);

  return { data, error, loading };
}
