/**
 * components/lab/BenchmarkRadar.tsx
 * ECharts 紫色雷达图：真实「压缩前 vs 压缩后」多指标对比。
 * 配色随 useLabTheme 自动切换；echarts 通过 import() 按需拆包。
 */
import { useEffect, useRef, useState } from "react"
import { useLabTheme } from "./theme"

/** 仅依赖我们实际用到的最小接口 */
interface MinimalChart {
  setOption: (option: unknown, notMerge?: boolean) => void
  resize: () => void
  dispose: () => void
}

export interface RadarSeriesInput {
  name: string
  values: number[]
  color: string
}

const PALETTES: Record<
  "light" | "dark",
  {
    axis: string
    grid: string
    split: string
    legend: string
    tooltipBg: string
    tooltipText: string
  }
> = {
  light: {
    axis: "#64748b",
    grid: "#94a3b8",
    split: "#e2e8f0",
    legend: "#334155",
    tooltipBg: "#ffffff",
    tooltipText: "#1e293b",
  },
  dark: {
    axis: "#b6b6c8",
    grid: "rgba(240, 240, 245, 0.18)",
    split: "rgba(240, 240, 245, 0.12)",
    legend: "#b6b6c8",
    tooltipBg: "#1a1a24",
    tooltipText: "#f0f0f5",
  },
}

export function BenchmarkRadar({
  indicators,
  series,
  height = 320,
}: {
  indicators: string[]
  series: RadarSeriesInput[]
  height?: number
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [chart, setChart] = useState<MinimalChart | null>(null)
  const { theme } = useLabTheme()

  // 仅在组件挂载时动态加载 echarts 并按需 init（code-split）
  useEffect(() => {
    let disposed = false
    let instance: MinimalChart | null = null
    let observer: ResizeObserver | null = null

    void import("echarts")
      .then((echarts) => {
        if (!containerRef.current || disposed) return
        const created = echarts.init(containerRef.current, undefined, {
          renderer: "canvas",
        })
        instance = created
        observer = new ResizeObserver(() => created.resize())
        observer.observe(containerRef.current)
        setChart(created)
      })
      .catch(() => {
        /* 加载失败时保持空白 */
      })

    return () => {
      disposed = true
      observer?.disconnect()
      instance?.dispose()
      setChart(null)
    }
  }, [])

  useEffect(() => {
    if (!chart || indicators.length === 0) return
    const p = PALETTES[theme]
    const maxes = indicators.map((_, i) => {
      const peak = Math.max(
        1e-6,
        ...series.map((s) =>
          typeof s.values[i] === "number" ? s.values[i] : 0,
        ),
      )
      return Number((peak * 1.2).toFixed(1))
    })

    chart.setOption(
      {
        color: series.map((s) => s.color),
        legend: {
          bottom: 0,
          icon: "circle",
          itemWidth: 8,
          textStyle: { color: p.legend, fontSize: 12 },
        },
        tooltip: {
          trigger: "item",
          backgroundColor: p.tooltipBg,
          textStyle: { color: p.tooltipText },
        },
        radar: {
          indicator: indicators.map((name, i) => ({ name, max: maxes[i] })),
          radius: "64%",
          center: ["50%", "48%"],
          splitArea: { show: false },
          axisName: { color: p.axis, fontSize: 12 },
          axisLine: { lineStyle: { color: p.grid } },
          splitLine: { lineStyle: { color: p.split } },
        },
        series: [
          {
            type: "radar",
            data: series.map((s) => ({
              value: s.values,
              name: s.name,
              areaStyle: { opacity: 0.22 },
              lineStyle: { width: 2 },
              symbol: "circle",
              symbolSize: 5,
            })),
          },
        ],
      },
      true,
    )
  }, [chart, indicators, series, theme])

  return (
    <div ref={containerRef} style={{ width: "100%", height }} aria-label="性能对比雷达图" />
  )
}
