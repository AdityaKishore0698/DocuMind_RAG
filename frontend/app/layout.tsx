import type { Metadata } from "next";
import { Roboto_Flex, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-context";
import { ThemeProvider } from "@/lib/theme-context";
import { ToastProvider } from "@/components/ui/Toast";

/**
 * Runs before first paint: applies the stored theme preference to <html> so
 * there is no flash of the wrong colour scheme. Kept in sync with
 * `lib/theme-context.tsx` (same storage key, same `data-theme` contract).
 */
const themeInitScript = `(function(){try{var t=localStorage.getItem("documind-theme");var d=document.documentElement;if(t==="dark"||t==="light"){d.setAttribute("data-theme",t);}else{d.removeAttribute("data-theme");}}catch(e){}})();`;

const robotoFlex = Roboto_Flex({
  variable: "--font-roboto-flex",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DocuMind",
  description: "Chat with your documents — a multi-tenant RAG workspace.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${robotoFlex.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full bg-md-surface text-md-on-surface">
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
        <ThemeProvider>
          <AuthProvider>
            <ToastProvider>{children}</ToastProvider>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
