import { Outlet } from "react-router-dom";

import { Sidebar } from "./Sidebar";

export function Layout() {
  return (
    <div className="flex h-full min-h-screen w-full">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-slate-50">
        <div className="mx-auto w-full max-w-6xl px-6 py-8">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
