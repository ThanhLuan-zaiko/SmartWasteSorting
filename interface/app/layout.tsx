import type { Metadata } from "next";

import { AppFrame } from "@/components/app-frame";
import { LocalAgentProvider } from "@/components/localagent-provider";
import { ThemeProvider } from "@/components/theme-provider";

import "./globals.css";

export const metadata: Metadata = {
  title: "Smart Waste Control Panel",
  description:
    "GUI for localagent pipelines, jobs, benchmarks, and experiment dashboards.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <LocalAgentProvider>
            <AppFrame>{children}</AppFrame>
          </LocalAgentProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
