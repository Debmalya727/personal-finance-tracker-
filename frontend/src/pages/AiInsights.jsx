import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Brain, TrendingUp, AlertTriangle, Target, Heart, RefreshCw } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const InsightCard = ({ text, index }) => {
  const getIcon = (t) => {
    if (t.toLowerCase().includes('alert') || t.toLowerCase().includes('unusual')) return { Icon: AlertTriangle, color: 'text-rose-400', bg: 'bg-rose-500/10 border-rose-500/20' };
    if (t.toLowerCase().includes('savings') || t.toLowerCase().includes('reach')) return { Icon: Target, color: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/20' };
    if (t.toLowerCase().includes('health')) return { Icon: Heart, color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/20' };
    return { Icon: TrendingUp, color: 'text-brand-purple', bg: 'bg-brand-purple/10 border-brand-purple/20' };
  };
  const { Icon, color, bg } = getIcon(text);

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className={`flex items-start gap-4 p-4 rounded-xl border ${bg} transition-all hover:scale-[1.01]`}
    >
      <div className={`mt-0.5 shrink-0 ${color}`}>
        <Icon size={18} />
      </div>
      <p className="text-sm text-white/80 font-medium leading-relaxed font-sans">{text}</p>
    </motion.div>
  );
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d0b25]/95 border border-white/10 rounded-xl p-3 shadow-2xl">
      <p className="text-white/50 text-xs mb-1">{label}</p>
      <p className="text-brand-cyan font-bold text-sm">₹{payload[0]?.value?.toFixed(2)}</p>
    </div>
  );
};

export const AiInsights = () => {
  const [insights, setInsights] = useState([]);
  const [prediction, setPrediction] = useState([]);
  const [loadingInsights, setLoadingInsights] = useState(true);
  const [loadingPrediction, setLoadingPrediction] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchInsights = async () => {
    setLoadingInsights(true);
    try {
      const res = await fetch('/api/ai_insights');
      const data = await res.json();
      setInsights(data.insights || []);
    } catch {
      setInsights(['Could not load AI insights.']);
    } finally {
      setLoadingInsights(false);
    }
  };

  const fetchPrediction = async () => {
    setLoadingPrediction(true);
    try {
      const res = await fetch('/api/predict_balance');
      const data = await res.json();
      setPrediction(data.prediction || []);
    } catch {
      setPrediction([]);
    } finally {
      setLoadingPrediction(false);
    }
  };

  useEffect(() => {
    fetchInsights();
    fetchPrediction();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await Promise.all([fetchInsights(), fetchPrediction()]);
    setRefreshing(false);
  };

  const predictionChartData = prediction.map((val, i) => ({ day: `Day ${i + 1}`, balance: parseFloat(val.toFixed(2)) }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white pb-16"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Brain size={16} className="text-brand-purple" />
            <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">AI Engine</span>
          </div>
          <h1 className="text-4xl font-black tracking-tight font-sans">AI Insights</h1>
          <p className="text-white/40 text-sm mt-1 font-sans">ML-powered financial intelligence and 30-day balance forecast</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-brand-purple/10 border border-brand-purple/25 text-brand-purple text-sm font-bold hover:bg-brand-purple/20 transition-all font-sans"
        >
          <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
          Refresh
        </motion.button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* AI Insights Panel */}
        <div className="glass-panel rounded-2xl p-6 border border-white/5 flex flex-col gap-5">
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-brand-purple animate-pulse" />
            <h2 className="text-base font-bold text-white font-sans">Smart Analysis</h2>
          </div>

          {loadingInsights ? (
            <div className="flex flex-col gap-3">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-14 rounded-xl bg-white/5 animate-pulse" />
              ))}
            </div>
          ) : insights.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <Brain size={36} className="text-white/20 mb-3" />
              <p className="text-white/30 text-sm font-sans">Add more transactions to generate AI insights</p>
            </div>
          ) : (
            <div className="flex flex-col gap-3">
              {insights.map((insight, i) => (
                <InsightCard key={i} text={insight} index={i} />
              ))}
            </div>
          )}
        </div>

        {/* Balance Prediction Panel */}
        <div className="glass-panel rounded-2xl p-6 border border-white/5 flex flex-col gap-5">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <TrendingUp size={16} className="text-brand-cyan" />
              <h2 className="text-base font-bold text-white font-sans">30-Day Balance Forecast</h2>
            </div>
            <p className="text-white/40 text-xs font-sans">RandomForest ML model trained on your transaction history</p>
          </div>

          {loadingPrediction ? (
            <div className="h-60 rounded-xl bg-white/5 animate-pulse" />
          ) : predictionChartData.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-60 text-center">
              <TrendingUp size={36} className="text-white/20 mb-3" />
              <p className="text-white/30 text-sm font-sans">Not enough transaction data for prediction</p>
            </div>
          ) : (
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={predictionChartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="predGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="day" stroke="rgba(255,255,255,0.2)" fontSize={10} tickLine={false} interval={4} />
                  <YAxis stroke="rgba(255,255,255,0.2)" fontSize={10} tickLine={false} axisLine={false} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="balance" stroke="#06b6d4" strokeWidth={2.5} fillOpacity={1} fill="url(#predGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default AiInsights;
