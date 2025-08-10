import { useEffect, useState } from "react";

interface WellnessGaugeProps {
  score: number; // 0-100
  className?: string;
}

export const WellnessGauge = ({ score, className = "" }: WellnessGaugeProps) => {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedScore(score);
    }, 500);
    return () => clearTimeout(timer);
  }, [score]);

  const getWellnessLevel = (score: number) => {
    if (score >= 85) return { level: "Excellent", color: "wellness-excellent" };
    if (score >= 70) return { level: "Good", color: "wellness-good" };
    if (score >= 50) return { level: "Moderate", color: "wellness-moderate" };
    return { level: "Needs Attention", color: "wellness-poor" };
  };

  const { level, color } = getWellnessLevel(animatedScore);
  const rotation = (animatedScore / 100) * 180 - 90; // -90 to 90 degrees

  return (
    <div className={`relative flex flex-col items-center ${className}`}>
      {/* Gauge Container */}
      <div className="relative w-48 h-24 mb-4">
        {/* Background Arc */}
        <svg
          viewBox="0 0 200 100"
          className="w-full h-full absolute inset-0"
        >
          <path
            d="M 20 80 A 80 80 0 0 1 180 80"
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth="8"
            className="opacity-30"
          />
          
          {/* Animated Progress Arc */}
          <path
            d="M 20 80 A 80 80 0 0 1 180 80"
            fill="none"
            stroke={`hsl(var(--${color}))`}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray="251.3"
            strokeDashoffset={251.3 - (animatedScore / 100) * 251.3}
            className="transition-all duration-2000 ease-out"
          />
        </svg>

        {/* Center Indicator */}
        <div 
          className="absolute top-1/2 left-1/2 w-3 h-3 bg-primary rounded-full transform -translate-x-1/2 -translate-y-1/2 transition-transform duration-2000 ease-out shadow-glow"
          style={{ 
            transform: `translate(-50%, -50%) rotate(${rotation}deg) translateY(-35px)` 
          }}
        />

        {/* Glow Effect */}
        <div className="absolute inset-0 bg-gradient-glow rounded-full opacity-20 animate-pulse-glow" />
      </div>

      {/* Score Display */}
      <div className="text-center">
        <div className="text-4xl font-bold text-foreground mb-1">
          {animatedScore}
        </div>
        <div className={`text-lg font-medium text-${color} mb-1`}>
          {level}
        </div>
        <div className="text-sm text-muted-foreground">
          Wellness Index
        </div>
      </div>
    </div>
  );
};