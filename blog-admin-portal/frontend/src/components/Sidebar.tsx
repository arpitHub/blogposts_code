import { NavLink } from "react-router-dom";
import { FileText, LayoutDashboard, PenSquare } from "lucide-react";

import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/posts", label: "Posts", icon: FileText, end: false },
  { to: "/posts/new", label: "New Post", icon: PenSquare, end: false },
];

export function Sidebar() {
  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-5 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-sky-500/20 text-sky-300">
          <PenSquare className="h-4 w-4" />
        </div>
        <div>
          <div className="text-sm font-semibold">Blog Admin</div>
          <div className="text-xs text-sidebar-foreground/60">
            Writer's portal
          </div>
        </div>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-sidebar-accent text-white"
                  : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-white",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-sidebar-border px-5 py-4 text-xs text-sidebar-foreground/60">
        v0.1.0 · local
      </div>
    </aside>
  );
}
