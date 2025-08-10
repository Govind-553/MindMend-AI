import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Camera, Mic, Keyboard, Activity } from "lucide-react";

interface DetectionData {
  facial?: {
    emotion: string;
    confidence: number;
  };
  speech?: {
    emotion: string;
    confidence: number;
  };
  keystroke?: {
    pattern: string;
    stress: number;
  };
  isActive: boolean;
}

interface LiveDetectionCardProps {
  data: DetectionData;
  className?: string;
}

export const LiveDetectionCard = ({ data, className = "" }: LiveDetectionCardProps) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "wellness-excellent";
    if (confidence >= 0.6) return "wellness-good";
    if (confidence >= 0.4) return "wellness-moderate";
    return "wellness-poor";
  };

  const getStressColor = (stress: number) => {
    if (stress <= 0.3) return "wellness-excellent";
    if (stress <= 0.5) return "wellness-good";
    if (stress <= 0.7) return "wellness-moderate";
    return "wellness-poor";
  };

  return (
    <Card className={`p-6 bg-gradient-glow border-0 shadow-card ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-foreground">
          Live Detection
        </h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${data.isActive ? 'bg-wellness-excellent animate-pulse-glow' : 'bg-muted'}`} />
          <span className="text-sm text-muted-foreground">
            {data.isActive ? 'Active' : 'Inactive'}
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {/* Facial Detection */}
        <div className="flex items-center justify-between p-4 bg-card/50 rounded-lg backdrop-blur-sm">
          <div className="flex items-center space-x-3">
            <Camera className="w-5 h-5 text-primary" />
            <div>
              <div className="font-medium text-foreground">Facial Analysis</div>
              <div className="text-sm text-muted-foreground">
                {data.facial?.emotion || 'No data'}
              </div>
            </div>
          </div>
          {data.facial && (
            <Badge 
              variant="outline" 
              className={`bg-${getConfidenceColor(data.facial.confidence)}/10 border-${getConfidenceColor(data.facial.confidence)}/30`}
            >
              {Math.round(data.facial.confidence * 100)}%
            </Badge>
          )}
        </div>

        {/* Speech Detection */}
        <div className="flex items-center justify-between p-4 bg-card/50 rounded-lg backdrop-blur-sm">
          <div className="flex items-center space-x-3">
            <Mic className="w-5 h-5 text-accent" />
            <div>
              <div className="font-medium text-foreground">Speech Analysis</div>
              <div className="text-sm text-muted-foreground">
                {data.speech?.emotion || 'No data'}
              </div>
            </div>
          </div>
          {data.speech && (
            <Badge 
              variant="outline" 
              className={`bg-${getConfidenceColor(data.speech.confidence)}/10 border-${getConfidenceColor(data.speech.confidence)}/30`}
            >
              {Math.round(data.speech.confidence * 100)}%
            </Badge>
          )}
        </div>

        {/* Keystroke Detection */}
        <div className="flex items-center justify-between p-4 bg-card/50 rounded-lg backdrop-blur-sm">
          <div className="flex items-center space-x-3">
            <Keyboard className="w-5 h-5 text-accent-light" />
            <div>
              <div className="font-medium text-foreground">Typing Pattern</div>
              <div className="text-sm text-muted-foreground">
                {data.keystroke?.pattern || 'No data'}
              </div>
            </div>
          </div>
          {data.keystroke && (
            <Badge 
              variant="outline" 
              className={`bg-${getStressColor(data.keystroke.stress)}/10 border-${getStressColor(data.keystroke.stress)}/30`}
            >
              Stress: {Math.round(data.keystroke.stress * 100)}%
            </Badge>
          )}
        </div>

        {/* Overall Status */}
        <div className="flex items-center justify-center pt-4 border-t border-border/30">
          <div className="flex items-center space-x-2 text-primary">
            <Activity className="w-4 h-4" />
            <span className="text-sm font-medium">
              Multi-modal analysis active
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};