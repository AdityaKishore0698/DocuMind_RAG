"use client";

import { useAuth } from "@/lib/auth-context";
import LoginView from "@/components/LoginView";
import AppShell from "@/components/AppShell";
import { Spinner } from "@/components/ui/Spinner";

export default function Home() {
  const { session, ready } = useAuth();

  if (!ready) {
    return (
      <div className="flex h-dvh items-center justify-center bg-md-surface text-md-on-surface-variant">
        <Spinner size={28} label="Loading DocuMind" />
      </div>
    );
  }

  return session ? <AppShell /> : <LoginView />;
}
