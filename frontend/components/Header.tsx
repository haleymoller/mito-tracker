import { Activity } from "lucide-react";
import { Button } from "./ui/button";

export function Header() {
  return (
    <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-6 h-6 text-blue-600" />
          <span className="text-xl">Mito Tracker</span>
        </div>
        <Button variant="ghost">Sign in</Button>
      </div>
    </header>
  );
}


