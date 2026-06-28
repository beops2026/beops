import "./globals.css";
import { ClerkProvider } from "@clerk/nextjs";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Grid Substation Forecasting System",
  description:
    "AI-powered electricity demand forecasting and operational reporting for grid substation operators",
};

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body className={inter.className}>
          <main className="min-h-screen bg-neutral-950">
            {children}
          </main>
        </body>
      </html>
    </ClerkProvider>
  );
}