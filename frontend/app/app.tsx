import { useState } from 'react';
import { Upload, Activity } from 'lucide-react';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Slider } from './components/ui/slider';
import { Checkbox } from './components/ui/checkbox';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { ImageDisplay } from './components/ImageDisplay';
import { MetricsTable } from './components/MetricsTable';
import { Header } from './components/Header';

export default function App() {
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [confidence, setConfidence] = useState([0.8]);
    const [useLLMLabels, setUseLLMLabels] = useState(false);
    const [pixelSize, setPixelSize] = useState('');
    const [isSegmented, setIsSegmented] = useState(false);
    const [metrics, setMetrics] = useState<any[]>([]);

    const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                setUploadedImage(e.target?.result as string);
                setIsSegmented(false);
                setMetrics([]);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleSegment = () => {
        // Mock segmentation - in real app, this would call your backend
        setIsSegmented(true);

        // Generate mock metrics
        const mockMetrics = Array.from({ length: 5 }, (_, i) => ({
            id: i + 1,
            area_px: Math.floor(Math.random() * 5000 + 1000),
            perimeter_px: Math.floor(Math.random() * 300 + 100),
            circularity: (Math.random() * 0.3 + 0.7).toFixed(2),
            area_nm: Math.floor(Math.random() * 50000 + 10000),
            perimeter_nm: Math.floor(Math.random() * 3000 + 1000),
            length_nm: Math.floor(Math.random() * 1500 + 500),
            width_nm: Math.floor(Math.random() * 800 + 200),
        }));

        setMetrics(mockMetrics);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
            <Header />

            <main className="container mx-auto px-4 py-8">
                {/* Hero Section */}
                <div className="text-center mb-12 mt-8">
                    <div className="flex items-center justify-center gap-3 mb-4">
                        <Activity className="w-12 h-12 text-blue-600" />
                        <h1 className="text-5xl">Mito Tracker</h1>
                    </div>
                    <p className="text-slate-600 text-lg max-w-2xl mx-auto">
                        Upload electron micrographs of mitochondria and get automatic segmentation,
                        labeling, and comprehensive metrics analysis powered by advanced AI.
                    </p>
                </div>

                {/* Main Interface */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
                    {/* Upload Panel */}
                    <Card className="p-6">
                        <h2 className="mb-6">Upload</h2>

                        <div className="space-y-6">
                            {/* File Upload */}
                            <div>
                                <Label htmlFor="file-upload" className="block mb-2">
                                    Choose file to upload
                                </Label>
                                <div className="relative">
                                    <input
                                        id="file-upload"
                                        type="file"
                                        accept="image/*"
                                        onChange={handleImageUpload}
                                        className="hidden"
                                    />
                                    <label
                                        htmlFor="file-upload"
                                        className="flex items-center justify-center gap-2 w-full p-4 border-2 border-dashed border-slate-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors cursor-pointer"
                                    >
                                        <Upload className="w-5 h-5 text-slate-500" />
                                        <span className="text-slate-600">
                                            {uploadedImage ? 'Change image' : 'Select image'}
                                        </span>
                                    </label>
                                </div>
                            </div>

                            {/* Confidence Threshold */}
                            <div>
                                <Label className="block mb-3">
                                    Confidence threshold: {confidence[0].toFixed(1)}
                                </Label>
                                <Slider
                                    value={confidence}
                                    onValueChange={setConfidence}
                                    min={0}
                                    max={1}
                                    step={0.1}
                                    className="w-full"
                                />
                            </div>

                            {/* LLM Labels Checkbox */}
                            <div className="flex items-center space-x-2">
                                <Checkbox
                                    id="llm-labels"
                                    checked={useLLMLabels}
                                    onCheckedChange={(checked) => setUseLLMLabels(checked as boolean)}
                                />
                                <Label
                                    htmlFor="llm-labels"
                                    className="cursor-pointer"
                                >
                                    Use LLM labels (sometimes more accurate)
                                </Label>
                            </div>

                            {/* Pixel Size Input */}
                            <div>
                                <Label htmlFor="pixel-size" className="block mb-2">
                                    Real one nm / pixel (optional)
                                </Label>
                                <Input
                                    id="pixel-size"
                                    type="number"
                                    placeholder="Enter pixel size"
                                    value={pixelSize}
                                    onChange={(e) => setPixelSize(e.target.value)}
                                />
                            </div>

                            {/* Segment Button */}
                            <Button
                                onClick={handleSegment}
                                disabled={!uploadedImage}
                                className="w-full"
                                size="lg"
                            >
                                Segment
                            </Button>
                        </div>
                    </Card>

                    {/* Original Image Panel */}
                    <Card className="p-6">
                        <h2 className="mb-6">Original</h2>
                        <ImageDisplay
                            image={uploadedImage}
                            placeholder="No image yet."
                        />
                    </Card>

                    {/* Overlay Panel */}
                    <Card className="p-6">
                        <h2 className="mb-6">Overlay</h2>
                        <ImageDisplay
                            image={isSegmented ? uploadedImage : null}
                            placeholder="Run segmentation to see overlay."
                            showOverlay={isSegmented}
                        />
                    </Card>
                </div>

                {/* Metrics Table */}
                <Card className="p-6">
                    <h2 className="mb-6">Mask & Metrics</h2>
                    <MetricsTable metrics={metrics} />
                </Card>
            </main>
        </div>
    );
}

