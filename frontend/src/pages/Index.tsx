import { useEffect, useState } from "react";
import { WellnessGauge } from "@/components/WellnessGauge";
import { TrendChart } from "@/components/TrendChart";
import { LiveDetectionCard } from "@/components/LiveDetectionCard";
import { QuickStats } from "@/components/QuickStats";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bell, Settings, User, Play, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { getWellnessHistory, getWellnessStats, healthCheck } from "@/services/api";
import MindMendLogo from "/img/MindMendLogo.png";

const Index = () => {
  const { toast } = useToast();
  const [wellnessScore, setWellnessScore] = useState(75);
  const [trendData, setTrendData] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [backendConnected, setBackendConnected] = useState(false);
  const [liveDetectionData, setLiveDetectionData] = useState({
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
    isActive: false
  });

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
    loadWellnessData();
  }, []);

  const checkBackendConnection = async () => {
    try {
      const response = await healthCheck();
      if (response.status === 'healthy') {
        setBackendConnected(true);
        toast({
          title: "✅ Connected to Backend",
          description: "MindMend AI services are online",
        });
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setBackendConnected(false);
      toast({
        title: "⚠️ Backend Offline",
        description: "Using demo data. Please start the backend server.",
        variant: "destructive"
      });
      loadMockData();
    }
  };

  const loadWellnessData = async () => {
    setIsLoading(true);
    try {
      // Fetch wellness history (last 7 days)
      const historyResponse = await getWellnessHistory(7);
      
      if (historyResponse.success && historyResponse.data.length > 0) {
        // Transform history data for trend chart
        const chartData = historyResponse.data
          .reverse()
          .map((entry: any, index: number) => {
            const date = new Date(entry.timestamp);
            const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
            
            // Get emotion label from speech or facial data
            let mood = 'Neutral';
            if (entry.speechEmotion?.label) {
              mood = entry.speechEmotion.label;
            } else if (entry.facialEmotion?.label) {
              mood = entry.facialEmotion.label;
            }

            return {
              date: days[date.getDay()],
              score: Math.round(entry.wellnessIndex),
              mood: mood.charAt(0).toUpperCase() + mood.slice(1)
            };
          });

        setTrendData(chartData);

        // Set latest wellness score
        const latestScore = historyResponse.data[0]?.wellnessIndex || 75;
        setWellnessScore(Math.round(latestScore));
      } else {
        loadMockData();
      }

      // Fetch wellness statistics
      const statsResponse = await getWellnessStats(30);
      if (statsResponse.success && statsResponse.data.stats) {
        setStats(statsResponse.data.stats);
      }

    } catch (error) {
      console.error('Error loading wellness data:', error);
      loadMockData();
    } finally {
      setIsLoading(false);
    }
  };

  const loadMockData = () => {
    // Fallback to mock data if backend is unavailable
    const mockTrendData = [
      { date: "Mon", score: 72, mood: "Calm" },
      { date: "Tue", score: 68, mood: "Focused" },
      { date: "Wed", score: 85, mood: "Happy" },
      { date: "Thu", score: 79, mood: "Relaxed" },
      { date: "Fri", score: 88, mood: "Energetic" },
      { date: "Sat", score: 92, mood: "Joyful" },
      { date: "Sun", score: 78, mood: "Content" }
    ];
    setTrendData(mockTrendData);
    setWellnessScore(78);
  };

  const startLiveAnalysis = () => {
    if (!backendConnected) {
      toast({
        title: "Backend Required",
        description: "Please start the backend server to use live analysis.",
        variant: "destructive"
      });
      return;
    }

    toast({
      title: "Starting Live Analysis",
      description: "Initializing camera, microphone, and keystroke tracking...",
    });

    // TODO: Implement actual live analysis
    setLiveDetectionData(prev => ({ ...prev, isActive: true }));
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
              
              {/* Backend Status Indicator */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${backendConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                <span className="text-xs text-muted-foreground">
                  {backendConnected ? 'Online' : 'Offline'}
                </span>
              </div>
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

      {/* Backend Warning Banner */}
      {!backendConnected && (
        <div className="bg-yellow-500/10 border-b border-yellow-500/30">
          <div className="container mx-auto px-6 py-3">
            <div className="flex items-center space-x-3 text-yellow-600">
              <AlertCircle className="w-5 h-5" />
              <p className="text-sm">
                Backend server is offline. Using demo data. Start the backend with: <code className="bg-black/20 px-2 py-1 rounded">python backend/app.py</code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Welcome Section */}
        <div className="mb-8 animate-fade-in">
          <h2 className="text-3xl font-bold text-foreground mb-2">
            Good morning, Alex
          </h2>
          <p className="text-lg text-muted-foreground">
            {backendConnected 
              ? "Your wellness journey continues. Here's your current status."
              : "Demo mode active. Connect backend for real-time analysis."}
          </p>
        </div>

        {/* Quick Stats */}
        <QuickStats className="mb-8" />

        {/* Main Dashboard Grid */}
        <div className="grid lg:grid-cols-3 gap-8 mb-8">
          {/* Wellness Gauge */}
          <Card className="lg:col-span-1 p-8 bg-gradient-glow border-0 shadow-card text-center animate-slide-up">       
            <WellnessGauge score={wellnessScore} />
            {stats && (
              <div className="mt-4 text-sm text-muted-foreground">
                <p>7-day average: {Math.round(stats.average)}</p>
                <p>Trend: <span className={`font-semibold ${
                  stats.trend.includes('improving') ? 'text-green-500' : 
                  stats.trend.includes('declining') ? 'text-red-500' : 
                  'text-yellow-500'
                }`}>{stats.trend.replace('_', ' ')}</span></p>
              </div>
            )}
          </Card>

          {/* Trend Chart */}
          <div className="lg:col-span-2 animate-slide-up" style={{ animationDelay: "0.1s" }}>
            {isLoading ? (
              <Card className="p-8 bg-gradient-glow border-0 shadow-card h-full flex items-center justify-center">
                <p className="text-muted-foreground">Loading wellness data...</p>
              </Card>
            ) : (
              <TrendChart data={trendData} />
            )}
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
                onClick={startLiveAnalysis}
                className="w-full justify-start bg-primary hover:bg-primary-light shadow-soft"
                size="lg"
                disabled={!backendConnected}
              >
                <Play className="w-5 h-5 mr-3" />
                Start Live Analysis
              </Button>

              <Button
                variant="outline"
                className="w-full justify-start border-accent/30 hover:bg-accent/10"
                size="lg"
                onClick={() => loadWellnessData()}
              >
                <User className="w-5 h-5 mr-3" />
                Refresh Data
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
                {stats ? (
                  <>
                    <div className="flex justify-between">
                      <span>Data points</span>
                      <span>{stats.data_points} sessions</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Best score</span>
                      <span className="text-wellness-excellent">{Math.round(stats.maximum)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Average score</span>
                      <span>{Math.round(stats.average)}</span>
                    </div>
                  </>
                ) : (
                  <>
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
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default Index;