"use client";

import { LocalAgentContext, useLocalAgent } from "@/components/localagent-context";
import { useLocalAgentController } from "@/components/use-localagent-controller";

export function LocalAgentProvider({ children }: { children: React.ReactNode }) {
  const value = useLocalAgentController();

  return (
    <LocalAgentContext.Provider value={value}>
      {children}
    </LocalAgentContext.Provider>
  );
}

export { useLocalAgent };
