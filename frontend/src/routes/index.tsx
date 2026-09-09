import { createFileRoute, redirect } from "@tanstack/react-router"

// Landing route: the app lives under /lab/*; send bare "/" to the login page
// so the platform never shows the router's "404 Oops" on the root path.
// After a successful login the app itself navigates to /lab/dashboard.
export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({ to: "/lab/login" })
  },
})
