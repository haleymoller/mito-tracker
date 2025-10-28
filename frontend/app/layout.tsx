import type { Metadata } from "next";
import { Geist, Geist_Mono, McLaren } from "next/font/google";
import Image from "next/image";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const mclaren = McLaren({ subsets: ["latin"], weight: "400" });

export const metadata: Metadata = {
  title: "Mito Tracker",
  description: "Upload EMs to segment mitochondria, visualize overlays, and get metrics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} ${mclaren.className} antialiased`}>
        <div className="w-full border-0 bg-[rgba(152,178,176,0.58)]">
          <div className="max-w-6xl mx-auto flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2">
              <Image src="/mito.png" alt="Mito" width={28} height={28} />
              <span className="font-medium">Mito Tracker</span>
            </div>
          </div>
        </div>
        {children}
      </body>
    </html>
  );
}
