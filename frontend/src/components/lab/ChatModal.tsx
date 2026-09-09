/**
 * components/lab/ChatModal.tsx · LLM 压缩产物对话测试（非流式）
 * 暴露 max_new_tokens / temperature / top_p / min_p 采样参数。
 */
import { Bot, Loader2, Send, User } from "lucide-react"
import { useEffect, useRef, useState } from "react"

import { useI18n } from "./i18n"
import { Button, Modal, TextAreaField, TextField } from "./ui"
import { apiChatCompletion } from "@/services/api"
import { apiErrorMessage } from "@/services/http"

interface Exchange {
  role: "user" | "assistant"
  content: string
}

export function ChatModal({
  taskId,
  open,
  onClose,
}: {
  taskId: string
  open: boolean
  onClose: () => void
}) {
  const { t } = useI18n()
  const [history, setHistory] = useState<Exchange[]>([])
  const [input, setInput] = useState("")
  const [maxTokens, setMaxTokens] = useState("96")
  const [temperature, setTemperature] = useState("0.4")
  const [topP, setTopP] = useState("0.9")
  const [minP, setMinP] = useState("0.0")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const endRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (open) {
      setHistory([])
      setInput("")
      setError("")
    }
  }, [open, taskId])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [history, busy])

  const submit = async () => {
    const text = input.trim()
    if (!text || busy) return
    setBusy(true)
    setError("")
    const messages: Exchange[] = [...history, { role: "user", content: text }]
    setHistory(messages)
    setInput("")
    try {
      const reply = await apiChatCompletion(taskId, messages, {
        max_new_tokens: Math.max(Number.parseInt(maxTokens, 10) || 64, 1),
        temperature: Number.parseFloat(temperature) || 0,
        top_p: Number.parseFloat(topP) || 1,
        min_p: Number.parseFloat(minP) || 0,
      })
      setHistory([...messages, { role: "assistant", content: reply.content }])
    } catch (e) {
      setError(apiErrorMessage(e))
      setHistory(messages)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={t("LLM 对话测试", "LLM Chat Test")}
      footer={
        <>
          <Button variant="outline" onClick={onClose}>
            {t("关闭", "Close")}
          </Button>
          <Button onClick={() => void submit()} disabled={busy || !input.trim()}>
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
            {busy ? t("生成中…", "Generating…") : t("发送", "Send")}
          </Button>
        </>
      }
    >
      <div className="grid gap-4">
        <div className="max-h-[300px] space-y-3 overflow-y-auto rounded-lg border border-[var(--lb-border)] bg-[var(--lb-surface-2)] p-3">
          {history.length === 0 && (
            <p className="lb-muted py-6 text-center text-[13px]">
              {t(
                "输入问题开始对话（采样参数：top_p/min_p/max tokens）。",
                "Type a message to start (sampling: top_p / min_p / max tokens).",
              )}
            </p>
          )}
          {history.map((item, i) => (
            <div key={i} className={`flex gap-2 ${item.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[85%] rounded-lg px-3 py-2 text-[13px] leading-relaxed ${
                  item.role === "user"
                    ? "bg-[var(--lb-primary)] text-[var(--lb-primary-foreground)]"
                    : "bg-[var(--lb-surface)] text-[var(--lb-text)]"
                }`}
              >
                <div className="mb-1 flex items-center gap-1 text-[11px] font-semibold opacity-70">
                  {item.role === "user" ? <User size={11} /> : <Bot size={11} />}
                  {item.role === "user" ? t("你", "You") : "Qwen/LLM"}
                </div>
                <div className="whitespace-pre-wrap">{item.content}</div>
              </div>
            </div>
          ))}
          {busy && (
            <div className="flex justify-start">
              <Loader2 size={16} className="animate-spin text-[var(--lb-primary)]" />
            </div>
          )}
          <div ref={endRef} />
        </div>

        <TextAreaField
          label={t("输入内容", "Input")}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={2}
          placeholder={t("说点什么…", "Say something…")}
        />
        {error && <p className="text-sm text-[var(--lb-danger)]">{error}</p>}

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <TextField label={t("最大新 tokens", "Max new tokens")} value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)} />
          <TextField label={t("温度 temperature", "Temperature")} value={temperature} onChange={(e) => setTemperature(e.target.value)} />
          <TextField label={t("top_p", "top_p")} value={topP} onChange={(e) => setTopP(e.target.value)} />
          <TextField label={t("min_p", "min_p")} value={minP} onChange={(e) => setMinP(e.target.value)} />
        </div>
      </div>
    </Modal>
  )
}
