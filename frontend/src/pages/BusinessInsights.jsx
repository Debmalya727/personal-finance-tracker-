import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Calendar, ShieldAlert, Sparkles, TrendingUp, HelpCircle } from 'lucide-react';

export const BusinessInsights = () => {
  const [dateForm, setDateForm] = useState({
    month: new Date().getMonth() + 1,
    year: new Date().getFullYear()
  });
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);

  const months = [
    { label: 'January', val: 1 }, { label: 'February', val: 2 }, { label: 'March', val: 3 },
    { label: 'April', val: 4 }, { label: 'May', val: 5 }, { label: 'June', val: 6 },
    { label: 'July', val: 7 }, { label: 'August', val: 8 }, { label: 'September', val: 9 },
    { label: 'October', val: 10 }, { label: 'November', val: 11 }, { label: 'December', val: 12 }
  ];

  const years = Array.from({ length: 5 }, (_, i) => new Date().getFullYear() - i);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setInsights(null);
    setError(null);

    fetch('/api/business/insights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dateForm)
    })
      .then(res => res.json())
      .then(data => {
        setLoading(false);
        if (data.error) {
          setError(data.error);
        } else {
          setInsights(data.insights);
        }
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
        setError('Failed to reach AI server. Please check your network connection.');
      });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-4xl flex flex-col gap-8 text-white relative z-10 font-sans"
    >
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight flex items-center gap-3">
          <Brain className="text-brand-purple" />
          Business AI Analyst
        </h1>
        <p className="text-white/40 text-sm mt-1">
          Generate small business financial evaluations, profit diagnostics, and custom cash flow forecasts using Google Gemini.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* Left: Input Selection Card (1 Column) */}
        <div className="lg:col-span-1 glass-panel p-6 rounded-2xl border border-white/5 bg-white/2 flex flex-col gap-4">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white/50 flex items-center gap-2">
            <Calendar size={14} />
            Period Selector
          </h3>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Month</label>
              <select
                value={dateForm.month}
                onChange={(e) => setDateForm({ ...dateForm, month: parseInt(e.target.value) })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              >
                {months.map(m => (
                  <option key={m.val} value={m.val}>{m.label}</option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Year</label>
              <select
                value={dateForm.year}
                onChange={(e) => setDateForm({ ...dateForm, year: parseInt(e.target.value) })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              >
                {years.map(y => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="mt-2 w-full py-3 rounded-xl font-bold bg-gradient-to-r from-brand-purple to-brand-pink text-white flex items-center justify-center gap-2 shadow-lg shadow-brand-purple/20 hover:shadow-brand-purple/35 transition-all text-sm disabled:opacity-55 disabled:cursor-not-allowed"
            >
              <Sparkles size={14} className="animate-pulse" />
              <span>{loading ? 'Consulting AI...' : 'Generate Insights'}</span>
            </button>
          </form>
        </div>

        {/* Right: Results Card (2 Columns) */}
        <div className="lg:col-span-2 flex flex-col gap-4 min-h-[300px]">
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="glass-panel p-8 rounded-2xl border border-white/5 bg-white/2 flex flex-col items-center justify-center gap-4 h-[300px]"
              >
                <div className="relative w-16 h-16">
                  <div className="absolute inset-0 rounded-full border-2 border-brand-purple/20 animate-ping" />
                  <div className="absolute inset-2 rounded-full border-t-2 border-brand-purple animate-spin" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold">Gemini is analyzing ledger tables...</p>
                  <p className="text-[10px] text-white/30 mt-1">Aggregating invoices, calculating growth rates and expenses.</p>
                </div>
              </motion.div>
            ) : error ? (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="glass-panel p-6 rounded-2xl border border-rose-500/10 bg-rose-500/5 text-rose-400 flex items-start gap-3 h-[300px] justify-center flex-col text-center"
              >
                <ShieldAlert size={36} className="mx-auto text-rose-500" />
                <h4 className="font-bold text-base mt-2 mx-auto">AI Analysis Halted</h4>
                <p className="text-xs text-white/50 mt-1 max-w-md mx-auto">{error}</p>
              </motion.div>
            ) : insights ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="glass-panel p-6 rounded-2xl border border-white/5 bg-white/2 flex flex-col gap-4 min-h-[300px]"
              >
                <div className="flex items-center gap-2 border-b border-white/5 pb-3">
                  <div className="w-8 h-8 rounded-lg bg-brand-purple/10 flex items-center justify-center border border-brand-purple/20">
                    <Sparkles size={14} className="text-brand-purple" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold">Financial Analysis Summary</h4>
                    <span className="text-[10px] text-white/40">Report compiled for {months.find(m => m.val === dateForm.month)?.label} {dateForm.year}</span>
                  </div>
                </div>

                <div className="text-sm text-white/80 leading-relaxed font-sans whitespace-pre-wrap">
                  {insights}
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="glass-panel p-8 rounded-2xl border border-white/5 bg-white/2 flex flex-col items-center justify-center gap-3 h-[300px] text-center"
              >
                <HelpCircle size={32} className="text-white/20" />
                <div>
                  <h4 className="text-sm font-bold text-white/50">Analyst Ready</h4>
                  <p className="text-[11px] text-white/30 mt-1 max-w-sm">
                    Select a month and year and click "Generate Insights" to query Google Gemini for financial analysis feedback.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
};

export default BusinessInsights;
