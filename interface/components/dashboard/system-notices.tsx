"use client";

import { HiCheckBadge, HiExclamationTriangle } from "react-icons/hi2";

import { useLocalAgent } from "@/components/localagent-provider";
import { useThemeMode } from "@/components/theme-provider";

export function SystemNotices() {
  const { isDark } = useThemeMode();
  const { connectionError, message } = useLocalAgent();

  if (!connectionError && !message) {
    return null;
  }

  return (
    <div className="grid gap-3">
      {connectionError ? (
        <article
          className={[
            "flex items-start gap-3 rounded-[1.5rem] border px-5 py-4",
            isDark
              ? "border-rose-900 bg-rose-950/70 text-rose-200"
              : "border-rose-200 bg-rose-50 text-rose-700",
          ].join(" ")}
        >
          <HiExclamationTriangle className="mt-0.5 text-xl" />
          <div>
            <div className="text-sm font-bold uppercase tracking-[0.22em]">
              Connection error
            </div>
            <p className="mt-1 text-sm leading-6">{connectionError}</p>
          </div>
        </article>
      ) : null}

      {message ? (
        <article
          className={[
            "flex items-start gap-3 rounded-[1.5rem] border px-5 py-4",
            isDark
              ? "border-emerald-900 bg-emerald-950/70 text-emerald-200"
              : "border-emerald-200 bg-emerald-50 text-emerald-700",
          ].join(" ")}
        >
          <HiCheckBadge className="mt-0.5 text-xl" />
          <div>
            <div className="text-sm font-bold uppercase tracking-[0.22em]">
              Latest action
            </div>
            <p className="mt-1 text-sm leading-6">{message}</p>
          </div>
        </article>
      ) : null}
    </div>
  );
}
