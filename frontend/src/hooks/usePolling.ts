/**
 * hooks/usePolling.ts
 * 轻量轮询 hook：按需拉取真实接口（组件内以 useEffect 使用）。
 * 不依赖 react-query，只负责「定时拉取 + 状态管理」。
 */
import { useEffect, useRef, useState } from "react"

interface PollingOptions {
  intervalMs?: number
  enabled?: boolean
}

export function usePolling<T>(
  loader: () => Promise<T>,
  dependencies: readonly unknown[],
  options: PollingOptions = {},
) {
  const { intervalMs = 3000, enabled = true } = options
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [version, setVersion] = useState(0)

  const loaderRef = useRef(loader)
  loaderRef.current = loader
  const intervalRef = useRef<number | undefined>(undefined)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false

    const tick = async () => {
      try {
        const result = await loaderRef.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError(err as Error)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    void tick()
    intervalRef.current = window.setInterval(() => void tick(), intervalMs)
    return () => {
      cancelled = true
      if (intervalRef.current !== undefined) {
        window.clearInterval(intervalRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, intervalMs, enabled, version])

  return { data, loading, error, refresh: () => setVersion((v) => v + 1) }
}
