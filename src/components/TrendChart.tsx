import { Card } from "@/components/ui/card";

interface TrendDataPoint {
  date: string;
  score: number;
  mood: string;
}

interface TrendChartProps {
  data: TrendDataPoint[];
  className?: string;
}

export const TrendChart = ({ data, className = "" }: TrendChartProps) => {
  const maxScore = Math.max(...data.map(d => d.score));
  
  return (
    <Card className={`p-6 bg-gradient-glow border-0 shadow-card ${className}`}>
      <h3 className="text-xl font-semibold mb-4 text-foreground">
        7-Day Wellness Trend
      </h3>
      
      <div className="relative h-48 w-full">
        <svg className="w-full h-full" viewBox="0 0 400 150">
          {/* Grid Lines */}
          {[0, 25, 50, 75, 100].map((value) => (
            <line
              key={value}
              x1="40"
              y1={130 - (value / 100) * 120}
              x2="380"
              y2={130 - (value / 100) * 120}
              stroke="hsl(var(--border))"
              strokeWidth="1"
              opacity="0.3"
            />
          ))}
          
          {/* Trend Line */}
          <polyline
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="3"
            points={data.map((point, index) => 
              `${60 + (index * 50)},${130 - (point.score / 100) * 120}`
            ).join(' ')}
            className="animate-fade-in"
          />
          
          {/* Data Points */}
          {data.map((point, index) => (
            <g key={index} className="animate-slide-up" style={{ animationDelay: `${index * 0.1}s` }}>
              <circle
                cx={60 + (index * 50)}
                cy={130 - (point.score / 100) * 120}
                r="4"
                fill="hsl(var(--primary))"
                className="hover:r-6 transition-all duration-200"
              />
              
              {/* Hover tooltip area */}
              <circle
                cx={60 + (index * 50)}
                cy={130 - (point.score / 100) * 120}
                r="12"
                fill="transparent"
                className="hover:fill-primary/10 cursor-pointer"
              />
            </g>
          ))}
          
          {/* Y-axis labels */}
          {[0, 25, 50, 75, 100].map((value) => (
            <text
              key={value}
              x="35"
              y={135 - (value / 100) * 120}
              fontSize="12"
              fill="hsl(var(--muted-foreground))"
              textAnchor="end"
            >
              {value}
            </text>
          ))}
        </svg>
        
        {/* X-axis labels */}
        <div className="flex justify-between mt-2 px-12 text-xs text-muted-foreground">
          {data.map((point, index) => (
            <span key={index} className="transform -rotate-45 origin-center">
              {point.date}
            </span>
          ))}
        </div>
      </div>
      
      {/* Legend */}
      <div className="flex items-center justify-center mt-4 text-sm text-muted-foreground">
        <div className="w-3 h-3 bg-primary rounded-full mr-2" />
        Daily Wellness Score
      </div>
    </Card>
  );
};