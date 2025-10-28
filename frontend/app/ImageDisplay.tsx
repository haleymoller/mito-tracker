interface ImageDisplayProps {
    image: string | null;
    placeholder: string;
    showOverlay?: boolean;
}

export function ImageDisplay({ image, placeholder, showOverlay }: ImageDisplayProps) {
    if (!image) {
        return (
            <div className="w-full aspect-square bg-slate-100 rounded-lg flex items-center justify-center border border-slate-200">
                <p className="text-slate-400">{placeholder}</p>
            </div>
        );
    }

    return (
        <div className="w-full aspect-square bg-slate-100 rounded-lg overflow-hidden border border-slate-200 relative">
            <img
                src={image}
                alt="Micrograph"
                className="w-full h-full object-contain"
            />
            {showOverlay && (
                <div className="absolute inset-0 pointer-events-none">
                    {/* Mock segmentation overlay - random colored regions */}
                    <svg className="w-full h-full" viewBox="0 0 100 100">
                        <circle cx="30" cy="35" r="8" fill="rgba(59, 130, 246, 0.3)" stroke="#3b82f6" strokeWidth="0.5" />
                        <ellipse cx="65" cy="30" rx="10" ry="7" fill="rgba(239, 68, 68, 0.3)" stroke="#ef4444" strokeWidth="0.5" />
                        <circle cx="45" cy="55" r="6" fill="rgba(34, 197, 94, 0.3)" stroke="#22c55e" strokeWidth="0.5" />
                        <ellipse cx="25" cy="70" rx="7" ry="9" fill="rgba(168, 85, 247, 0.3)" stroke="#a855f7" strokeWidth="0.5" />
                        <circle cx="70" cy="65" r="9" fill="rgba(234, 179, 8, 0.3)" stroke="#eab308" strokeWidth="0.5" />
                    </svg>
                </div>
            )}
        </div>
    );
}

