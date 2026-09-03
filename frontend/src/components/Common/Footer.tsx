import { Link as RouterLink } from "@tanstack/react-router"
import { FaGithub, FaLinkedinIn } from "react-icons/fa"
import { FaXTwitter } from "react-icons/fa6"

const socialLinks = [
  {
    icon: FaGithub,
    href: "https://github.com/fastapi/fastapi",
    label: "GitHub",
  },
  { icon: FaXTwitter, href: "https://x.com/fastapi", label: "X" },
  {
    icon: FaLinkedinIn,
    href: "https://linkedin.com/company/fastapi",
    label: "LinkedIn",
  },
]

export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t py-4 px-6">
      <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
        <p className="text-muted-foreground text-sm">
          AI Model Lightweight Platform - {currentYear}
        </p>
        <div className="flex items-center gap-4">
          <RouterLink
            to="/lab/dashboard"
            className="text-muted-foreground hover:text-foreground text-sm transition-colors"
          >
            平台使用报告
          </RouterLink>
          {socialLinks.map(({ icon: Icon, href, label }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={label}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <Icon className="h-5 w-5" />
            </a>
          ))}
        </div>
      </div>
    </footer>
  )
}
