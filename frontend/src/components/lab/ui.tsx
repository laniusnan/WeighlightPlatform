/**
 * components/lab/ui.tsx
 * /lab 模块自带的浅色原语（不使用依赖全局暗色令牌的 shadcn 组件）。
 */
import { useEffect, type ReactNode } from "react"
import { X } from "lucide-react"

export const cx = (...parts: Array<string | false | null | undefined>) =>
  parts.filter(Boolean).join(" ")

/* ------------------------------ Button ------------------------------ */
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "outline" | "ghost" | "danger"
  size?: "sm" | "md"
}

export function Button({
  variant = "primary",
  size = "md",
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cx(
        "lb-btn",
        `lb-btn--${variant}`,
        size === "sm" && "px-2.5 py-1.5 text-[13px]",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

/* ------------------------------ Card -------------------------------- */
export function Card({
  children,
  className,
  white = true,
}: {
  children: ReactNode
  className?: string
  white?: boolean
}) {
  return (
    <section
      className={cx("lb-card", white && "lb-card-white", className)}
    >
      {children}
    </section>
  )
}

/* --------------------------- Form fields ---------------------------- */
export function FieldLabel({ children }: { children: ReactNode }) {
  return <label className="lb-label">{children}</label>
}

export function TextField({
  label,
  hint,
  className,
  ...rest
}: React.InputHTMLAttributes<HTMLInputElement> & {
  label?: string
  hint?: string
}) {
  return (
    <div className={className}>
      {label && <FieldLabel>{label}</FieldLabel>}
      <input className="lb-input" {...rest} />
      {hint && <p className="lb-hint">{hint}</p>}
    </div>
  )
}

export function SelectField({
  label,
  hint,
  children,
  className,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement> & {
  label?: string
  hint?: string
}) {
  return (
    <div className={className}>
      {label && <FieldLabel>{label}</FieldLabel>}
      <select className="lb-select" {...rest}>
        {children}
      </select>
      {hint && <p className="lb-hint">{hint}</p>}
    </div>
  )
}

export function TextAreaField({
  label,
  hint,
  className,
  ...rest
}: React.TextareaHTMLAttributes<HTMLTextAreaElement> & {
  label?: string
  hint?: string
}) {
  return (
    <div className={className}>
      {label && <FieldLabel>{label}</FieldLabel>}
      <textarea className="lb-textarea" {...rest} />
      {hint && <p className="lb-hint">{hint}</p>}
    </div>
  )
}

/* ------------------------------ Modal ------------------------------- */
export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
  wide = false,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  footer?: ReactNode
  wide?: boolean
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null
  return (
    <div
      className="lb-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div
        className="lb-modal"
        style={wide ? { width: "min(920px, 100%)" } : undefined}
        role="dialog"
        aria-modal="true"
      >
        <div className="lb-modal__head">
          <h2 className="lb-modal__title">{title}</h2>
          <button
            type="button"
            aria-label="关闭"
            onClick={onClose}
            className="lb-btn lb-btn--ghost p-1.5"
          >
            <X size={18} />
          </button>
        </div>
        <div className="lb-modal__body">{children}</div>
        {footer && <div className="lb-modal__foot">{footer}</div>}
      </div>
    </div>
  )
}

/* --------------------------- Progress bar --------------------------- */
export function ProgressBar({
  value,
  className,
}: {
  value: number
  className?: string
}) {
  const pct = Math.min(100, Math.max(0, value))
  return (
    <div className={cx("lb-progress", className)} role="progressbar">
      <div className="lb-progress__bar" style={{ width: `${pct}%` }} />
    </div>
  )
}

/* --------------------------- Status pill ---------------------------- */
export type PillTone =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "canceled"
  | "idle"
  | "info"
  | "ready"

export function Pill({
  tone,
  children,
}: {
  tone: PillTone
  children: ReactNode
}) {
  const resolved = tone === "ready" ? "succeeded" : tone
  return <span className={cx("lb-pill", `lb-pill--${resolved}`)}>{children}</span>
}

/* ---------------------------- Empty state --------------------------- */
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-16 text-center">
      <p className="text-[15px] font-semibold text-[var(--lb-text-2)]">
        {title}
      </p>
      {description && <p className="lb-muted max-w-md text-sm">{description}</p>}
      {action}
    </div>
  )
}

/* --------------------------- Stat card ------------------------------ */
export function StatCard({
  label,
  value,
  hint,
  icon,
}: {
  label: string
  value: string
  hint?: string
  icon?: ReactNode
}) {
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <p className="lb-muted text-[13px] font-medium">{label}</p>
        {icon && <span className="text-[var(--lb-primary)]">{icon}</span>}
      </div>
      <p className="lb-accent-text mt-2 text-[26px] font-bold leading-none tracking-tight">
        {value}
      </p>
      {hint && <p className="lb-muted mt-2 text-xs">{hint}</p>}
    </Card>
  )
}
