import { WellnessGauge } from "@/components/WellnessGauge";
import { TrendChart } from "@/components/TrendChart";
import { LiveDetectionCard } from "@/components/LiveDetectionCard";
import { QuickStats } from "@/components/QuickStats";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bell, Settings, User, Play } from "lucide-react";
import MindMendLogo from "/img/MindMendLogo.png";

const Index = () => {
  // Mock data for demo
  const trendData = [
    { date: "Mon", score: 72, mood: "Calm" },
    { date: "Tue", score: 68, mood: "Focused" },
    { date: "Wed", score: 85, mood: "Happy" },
    { date: "Thu", score: 79, mood: "Relaxed" },
    { date: "Fri", score: 88, mood: "Energetic" },
    { date: "Sat", score: 92, mood: "Joyful" },
    { date: "Sun", score: 78, mood: "Content" }
  ];

  const liveDetectionData = {
    facial: {
      emotion: "Calm",
      confidence: 0.87
    },
    speech: {
      emotion: "Positive",
      confidence: 0.92
    },
    keystroke: {
      pattern: "Relaxed",
      stress: 0.25
    },
    isActive: true
  };

  return (
    <div className="min-h-screen bg-gradient-calm">
      {/* Header */}
      <header className="bg-card/80 backdrop-blur-sm border-b border-border/30 sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <img src={MindMendLogo} alt="MindMend Logo" className="w-20 h-12 rounded-full" />
              <h1 className="text-2xl font-bold text-foreground">MindMend AI</h1>
            </div>
            
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="icon">
                <Bell className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon">
                <Settings className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="icon">
                <User className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="mb-8 animate-fade-in">
          <h2 className="text-3xl font-bold text-foreground mb-2">
            Good morning, Alex
          </h2>
          <p className="text-lg text-muted-foreground">
            Your wellness journey continues. Here's your current status.
          </p>
        </div>

        {/* Quick Stats */}
        <QuickStats className="mb-8" />

        {/* Main Dashboard Grid */}
        <div className="grid lg:grid-cols-3 gap-8 mb-8">
          {/* Wellness Gauge */}
          <Card className="lg:col-span-1 p-8 bg-gradient-glow border-0 shadow-card text-center animate-slide-up">
            <WellnessGauge score={78} />
          </Card>

          {/* Trend Chart */}
          <div className="lg:col-span-2 animate-slide-up" style={{ animationDelay: "0.1s" }}>
            <TrendChart data={trendData} />
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Live Detection */}
          <div className="animate-slide-up" style={{ animationDelay: "0.2s" }}>
            <LiveDetectionCard data={liveDetectionData} />
          </div>

          {/* Quick Actions */}
          <Card className="p-6 bg-gradient-glow border-0 shadow-card animate-slide-up" style={{ animationDelay: "0.3s" }}>
            <h3 className="text-xl font-semibold mb-4 text-foreground">
              Quick Actions
            </h3>
            
            <div className="space-y-3">
              <Button 
                className="w-full justify-start bg-primary hover:bg-primary-light shadow-soft"
                size="lg"
              >
                <Play className="w-5 h-5 mr-3" />
                Start Live Analysis
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full justify-start border-accent/30 hover:bg-accent/10"
                size="lg"
              >
                <User className="w-5 h-5 mr-3" />
                View Detailed Report
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full justify-start border-accent/30 hover:bg-accent/10"
                size="lg"
              >
                <Settings className="w-5 h-5 mr-3" />
                Customize Settings
              </Button>
            </div>

            {/* Recent Activity */}
            <div className="mt-6 pt-6 border-t border-border/30">
              <h4 className="font-medium text-foreground mb-3">Recent Activity</h4>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex justify-between">
                  <span>Last session</span>
                  <span>2 hours ago</span>
                </div>
                <div className="flex justify-between">
                  <span>Weekly goal</span>
                  <span className="text-wellness-excellent">85% complete</span>
                </div>
                <div className="flex justify-between">
                  <span>Best day</span>
                  <span>Saturday (92)</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Index;
