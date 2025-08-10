import { Card } from "@/components/ui/card";
import { TrendingUp, Heart, Zap, Target } from "lucide-react";

interface StatItem {
  label: string;
  value: string;
  change: string;
  trend: "up" | "down" | "stable";
  icon: any;
}

interface QuickStatsProps {
  className?: string;
}

export const QuickStats = ({ className = "" }: QuickStatsProps) => {
  const stats: StatItem[] = [
    {
      label: "Weekly Average",
      value: "78",
      change: "+5%",
      trend: "up",
      icon: TrendingUp
    },
    {
      label: "Stress Level",
      value: "Low",
      change: "-12%",
      trend: "down",
      icon: Heart
    },
    {
      label: "Energy Level",
      value: "High",
      change: "+8%",
      trend: "up",
      icon: Zap
    },
    {
      label: "Daily Goal",
      value: "85%",
      change: "On track",
      trend: "stable",
      icon: Target
    }
  ];

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case "up": return "wellness-excellent";
      case "down": return "wellness-good";
      default: return "muted-foreground";
    }
  };

  return (
    <div className={`grid grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}>
      {stats.map((stat, index) => (
        <Card 
          key={stat.label} 
          className="p-4 bg-gradient-glow border-0 shadow-card animate-slide-up hover:shadow-glow transition-all duration-300"
          style={{ animationDelay: `${index * 0.1}s` }}
        >
          <div className="flex items-center justify-between mb-2">
            <stat.icon className="w-5 h-5 text-primary" />
            <span className={`text-xs font-medium text-${getTrendColor(stat.trend)}`}>
              {stat.change}
            </span>
          </div>
          
          <div className="space-y-1">
            <div className="text-2xl font-bold text-foreground">
              {stat.value}
            </div>
            <div className="text-sm text-muted-foreground">
              {stat.label}
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};