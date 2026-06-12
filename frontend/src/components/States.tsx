/** Loading / empty / error states shared by every view. */

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex h-48 items-center justify-center text-sm text-neutral-500">
      <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
      {label}
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex h-48 flex-col items-center justify-center text-center">
      <p className="text-sm text-neutral-400">{title}</p>
      {hint && <p className="mt-1 max-w-md font-mono text-xs text-neutral-600">{hint}</p>}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex h-48 flex-col items-center justify-center text-center">
      <p className="text-sm text-rose-300">Request failed</p>
      <p className="mt-1 max-w-md font-mono text-xs text-neutral-500">{message}</p>
      <p className="mt-2 text-xs text-neutral-600">
        Is the API running? <span className="font-mono">uvicorn engram.api.app:app --port 8000</span>
      </p>
    </div>
  );
}
