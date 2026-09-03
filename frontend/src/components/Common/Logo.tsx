import { Link } from "@tanstack/react-router"
import { Cpu } from "lucide-react"

import { cn } from "@/lib/utils"

interface LogoProps {
  variant?: "full" | "icon" | "responsive"
  className?: string
  asLink?: boolean
}

const BRAND_NAME = "AI Model Lightweight Platform"

export function Logo({
  variant = "full",
  className,
  asLink = true,
}: LogoProps) {
  const icon = <Cpu className={cn("size-5 shrink-0", className)} />

  const content =
    variant === "responsive" ? (
      <>
        <span className="group-data-[collapsible=icon]:hidden flex items-center gap-2">
          <Cpu className="size-5 shrink-0" />
          <span className="truncate font-semibold">{BRAND_NAME}</span>
        </span>
        <Cpu className="group-data-[collapsible=icon]:block hidden size-5 shrink-0" />
      </>
    ) : variant === "icon" ? (
      icon
    ) : (
      <span className={cn("flex items-center gap-2", className)}>
        <Cpu className="size-6 shrink-0" />
        <span className="whitespace-nowrap font-semibold">{BRAND_NAME}</span>
      </span>
    )

  if (!asLink) {
    return content
  }

  return <Link to="/lab/dashboard">{content}</Link>
}
