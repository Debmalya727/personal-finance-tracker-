import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, TrendingDown, DollarSign, Users, Briefcase, 
  Percent, ArrowUpRight, BarChart3, Activity 
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export const BusinessFinancials = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/business/financials')
      .then(res => res.json())
      .then(resData => {
        setData(resData);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load business financials:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-12 h-12 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
      </div>
    );
  }

  const financials = data || {
    monthly_revenue: 0,
    monthly_expenses: 0,
    net_profit: 0,
    profit_margin: 0,
    active_clients: 0,
    total_investment_value: 0,
    total_outstanding_loans: 0
  };

  const chartData = [
    { name: 'Expenses', amount: financials.monthly_expenses, color: '#db2777' },
    { name: 'Net Profit', amount: financials.net_profit, color: '#06b6d4' },
    { name: 'Revenue', amount: financials.monthly_revenue, color: '#7c3aed' }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white relative z-10 font-sans"
    >
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight">Business Financials</h1>
        <p className="text-white/40 text-sm mt-1">
          Detailed metrics, monthly summaries, and profit margin analysis.
        </p>
      </div>

      {/* Grid of Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div 
          whileHover={{ y: -4, scale: 1.01 }}
          className="glass-panel p-6 rounded-2xl border border-white/5 bg-gradient-to-tr from-emerald-500/10 to-transparent flex items-center justify-between"
        >
          <div className="flex flex-col gap-1">
            <span className="text-xs text-white/50 font-bold uppercase tracking-wider">Monthly Revenue</span>
            <span className="text-2xl font-black text-emerald-400">₹{financials.monthly_revenue.toLocaleString('en-IN')}</span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <TrendingUp size={22} className="text-emerald-400" />
          </div>
        </motion.div>

        <motion.div 
          whileHover={{ y: -4, scale: 1.01 }}
          className="glass-panel p-6 rounded-2xl border border-white/5 bg-gradient-to-tr from-brand-pink/10 to-transparent flex items-center justify-between"
        >
          <div className="flex flex-col gap-1">
            <span className="text-xs text-white/50 font-bold uppercase tracking-wider">Monthly Expenses</span>
            <span className="text-2xl font-black text-brand-pink">₹{financials.monthly_expenses.toLocaleString('en-IN')}</span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-brand-pink/10 border border-brand-pink/20 flex items-center justify-center">
            <TrendingDown size={22} className="text-brand-pink" />
          </div>
        </motion.div>

        <motion.div 
          whileHover={{ y: -4, scale: 1.01 }}
          className="glass-panel p-6 rounded-2xl border border-white/5 bg-gradient-to-tr from-brand-purple/10 to-transparent flex items-center justify-between"
        >
          <div className="flex flex-col gap-1">
            <span className="text-xs text-white/50 font-bold uppercase tracking-wider">Net Profit</span>
            <span className={`text-2xl font-black ${financials.net_profit >= 0 ? 'text-brand-cyan' : 'text-rose-400'}`}>
              ₹{financials.net_profit.toLocaleString('en-IN')}
            </span>
          </div>
          <div className="w-12 h-12 rounded-xl bg-brand-purple/10 border border-brand-purple/20 flex items-center justify-center">
            <DollarSign size={22} className="text-brand-purple" />
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left Chart Panel (3 Columns) */}
        <div className="lg:col-span-3 glass-panel p-6 rounded-2xl border border-white/5 flex flex-col gap-4">
          <h2 className="text-base font-bold flex items-center gap-2">
            <BarChart3 size={16} className="text-brand-purple" />
            Cashflow Distribution
          </h2>
          <div className="h-[250px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="chartColor" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#7c3aed" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.4)" fontSize={10} />
                <YAxis stroke="rgba(255,255,255,0.4)" fontSize={10} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(10, 8, 28, 0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px' }}
                  labelStyle={{ color: 'rgba(255,255,255,0.6)' }}
                />
                <Area type="monotone" dataKey="amount" stroke="#7c3aed" strokeWidth={2} fillOpacity={1} fill="url(#chartColor)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right Info Cards (2 Columns) */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-3">
            <span className="text-xs text-white/50 font-bold uppercase tracking-wider">Profit Margin</span>
            <div className="flex items-center justify-between">
              <span className="text-3xl font-black text-brand-purple">{financials.profit_margin.toFixed(1)}%</span>
              <div className="w-10 h-10 rounded-xl bg-brand-purple/10 border border-brand-purple/20 flex items-center justify-center">
                <Percent size={18} className="text-brand-purple" />
              </div>
            </div>
            {/* Progress bar */}
            <div className="w-full bg-white/5 h-2 rounded-full overflow-hidden border border-white/5">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, Math.max(0, financials.profit_margin))}%` }}
                transition={{ duration: 1, ease: 'easeOut' }}
                className="h-full bg-gradient-to-r from-brand-purple to-brand-pink"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="glass-panel p-4 rounded-xl border border-white/5 flex flex-col gap-1">
              <span className="text-[10px] text-white/40 font-bold uppercase tracking-wider block">Active Clients</span>
              <div className="flex items-center justify-between mt-1">
                <span className="text-xl font-bold">{financials.active_clients}</span>
                <Users size={14} className="text-brand-cyan" />
              </div>
            </div>

            <div className="glass-panel p-4 rounded-xl border border-white/5 flex flex-col gap-1">
              <span className="text-[10px] text-white/40 font-bold uppercase tracking-wider block">Investments Value</span>
              <div className="flex items-center justify-between mt-1">
                <span className="text-lg font-bold">₹{financials.total_investment_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                <Briefcase size={14} className="text-brand-pink" />
              </div>
            </div>
          </div>

          <div className="glass-panel p-4 rounded-xl border border-white/5 flex items-center justify-between bg-brand-pink/5">
            <div className="flex flex-col gap-0.5">
              <span className="text-[10px] text-white/40 font-bold uppercase tracking-wider block">Outstanding Loans</span>
              <span className="text-lg font-bold text-white/95">₹{financials.total_outstanding_loans.toLocaleString('en-IN')}</span>
            </div>
            <div className="w-8 h-8 rounded-lg bg-brand-pink/15 flex items-center justify-center border border-brand-pink/20">
              <Activity size={14} className="text-brand-pink" />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default BusinessFinancials;
