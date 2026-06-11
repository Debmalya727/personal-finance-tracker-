import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, TrendingDown, Wallet, ArrowUpRight, 
  Sparkles, Activity, BarChart3, Zap, Target, Link
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer
} from 'recharts';
import { ThreeDCard } from '../components/ThreeDCard';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

// ── Animated counter — BUG FIX: handles 0 correctly, handles negatives ──────
const useAnimatedCounter = (target, duration = 1400) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    // BUG FIX: was `if (!target) return` which bails on 0 — now checks undefined/null
    if (target === null || target === undefined) return;
    let startTime = null;
    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.round(target * eased));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [target, duration]);
  return count;
};

// Custom Chart Tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d0b25]/95 border border-white/10 rounded-2xl p-4 shadow-2xl">
      <p className="text-white/60 text-xs font-medium mb-2">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 text-sm font-bold">
          <div className="w-2 h-2 rounded-full" style={{ background: entry.color }} />
          <span style={{ color: entry.color }}>₹{(entry.value || 0).toLocaleString('en-IN')}</span>
        </div>
      ))}
    </div>
  );
};

// Stat Card — sparkline width is proportional to value vs maxValue
const StatCard = ({ title, value, subtitle, icon: Icon, gradient, glow, index, barPct }) => {
  const animatedValue = useAnimatedCounter(Math.abs(value ?? 0));
  const isNegative = value < 0;
  // Clamp bar percentage 0-100
  const barWidth = Math.min(100, Math.max(0, barPct ?? 60));

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 + 0.3, duration: 0.6, ease: 'easeOut' }}
    >
      <ThreeDCard className="h-44">
        <div className="flex items-start justify-between h-full flex-col">
          <div className="flex items-center justify-between w-full">
            <span className="text-xs font-bold text-white/50 uppercase tracking-widest font-sans">
              {title}
            </span>
            <div className={`p-2.5 rounded-xl bg-gradient-to-tr ${gradient} shadow-lg`}
              style={{ boxShadow: glow }}>
              <Icon size={16} className="text-white" />
            </div>
          </div>
          <div>
            <h3 className={`text-3xl font-black mt-2 tracking-tight font-sans ${
              isNegative ? 'text-rose-400' : 'text-white'
            }`}>
              {isNegative ? '−' : ''}₹{animatedValue.toLocaleString('en-IN')}
            </h3>
            <span className="text-xs text-white/40 font-medium block mt-1.5 font-sans">
              {subtitle}
            </span>
          </div>
          {/* Sparkline — width reflects actual proportion */}
          <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden mt-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${barWidth}%` }}
              transition={{ delay: index * 0.1 + 0.8, duration: 1.2, ease: 'easeOut' }}
              className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
            />
          </div>
        </div>
      </ThreeDCard>
    </motion.div>
  );
};

// Budget Ring — with empty/no-budget state
const BudgetRing = ({ budget }) => {
  const navigate = useNavigate();
  const pct    = budget?.percentage_spent ?? 0;          // 0-100 capped
  const rawPct = budget?.raw_percentage ?? 0;            // true value, can exceed 100
  const hasBudgets = budget?.has_budgets ?? false;
  const isOver = budget?.is_over_budget ?? false;
  const circumference = 2 * Math.PI * 60;

  const ringColor = isOver      ? '#f43f5e'
                  : pct > 70    ? '#f59e0b'
                  : pct > 0     ? '#7c3aed'
                  : 'rgba(255,255,255,0.1)';

  if (!hasBudgets) {
    // BUG FIX: Was showing "0% Spent ₹0 ₹0" — meaningless. Now shows actionable empty state.
    return (
      <div className="flex flex-col items-center justify-center flex-1 gap-4 py-4">
        <div className="w-20 h-20 rounded-2xl bg-brand-purple/10 border border-brand-purple/20 flex items-center justify-center">
          <Target size={32} className="text-brand-purple/60" />
        </div>
        <div className="text-center">
          <p className="text-white/60 text-sm font-semibold font-sans">No budgets set yet</p>
          <p className="text-white/30 text-xs font-sans mt-1">Set category budgets to track spending</p>
        </div>
        <button
          onClick={() => navigate('/budget')}
          className="mt-1 px-4 py-2 rounded-xl bg-brand-purple/15 border border-brand-purple/25 text-brand-purple text-xs font-bold hover:bg-brand-purple/25 transition-all font-sans"
        >
          Set up Budget →
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center flex-1 justify-center">
      <div className="relative w-40 h-40">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 144 144">
          {/* Track */}
          <circle cx="72" cy="72" r="60" stroke="rgba(255,255,255,0.05)" strokeWidth="10" fill="none" />
          {/* Glow */}
          <circle cx="72" cy="72" r="60" stroke={ringColor} strokeWidth="4" fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - (circumference * pct / 100)}
            strokeLinecap="round" opacity={0.25}
          />
          {/* Main arc */}
          <motion.circle
            cx="72" cy="72" r="60"
            stroke={ringColor}
            strokeWidth="10" fill="none"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - (circumference * pct / 100) }}
            transition={{ duration: 1.5, delay: 0.8, ease: 'easeOut' }}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.2, duration: 0.5 }}
            className="text-4xl font-black leading-none font-sans"
            style={{ color: isOver ? '#f43f5e' : 'white' }}
          >
            {/* BUG FIX: show raw% so over-budget is visible (e.g. 140%), ring is still capped at 100% */}
            {Math.round(rawPct)}%
          </motion.span>
          <span className="text-xs font-bold uppercase tracking-wider mt-1 font-sans"
            style={{ color: isOver ? '#f43f5e' : 'rgba(255,255,255,0.4)' }}
          >
            {isOver ? 'Over budget!' : 'Spent'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 w-full mt-5">
        <div className="text-center p-3 rounded-xl bg-white/5 border border-white/5">
          <span className="text-xs text-white/40 block font-sans">Budgeted</span>
          <span className="font-bold text-white text-sm font-sans mt-0.5 block">
            ₹{(budget?.total_budgeted || 0).toLocaleString('en-IN')}
          </span>
        </div>
        <div className="text-center p-3 rounded-xl bg-white/5 border border-white/5">
          <span className="text-xs text-white/40 block font-sans">Spent</span>
          <span className="font-bold text-sm font-sans mt-0.5 block"
            style={{ color: isOver ? '#f43f5e' : 'white' }}
          >
            ₹{(budget?.total_spent || 0).toLocaleString('en-IN')}
          </span>
        </div>
      </div>
      {/* Show total monthly expense separately for context */}
      {budget?.total_expense > 0 && budget.total_expense !== budget.total_spent && (
        <p className="text-[10px] text-white/25 mt-2 font-sans text-center">
          Total month expenses: ₹{budget.total_expense.toLocaleString('en-IN')}
        </p>
      )}
    </div>
  );
};

// ── Main Dashboard ─────────────────────────────────────────────────────────────
export const Dashboard = () => {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const containerRef          = useRef(null);
  const { user }              = useAuth();

  useEffect(() => {
    fetch('/api/dashboard')
      .then((res) => res.json())
      .then((resData) => { setData(resData); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex flex-col items-center justify-center gap-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-2 border-brand-purple/20 animate-ping" />
          <div className="absolute inset-2 rounded-full border-t-2 border-brand-purple animate-spin" />
          <div className="absolute inset-4 rounded-full bg-brand-purple/20 animate-pulse" />
        </div>
        <p className="text-white/40 text-sm font-medium font-sans animate-pulse">Loading your wealth data...</p>
      </div>
    );
  }

  const d = data || {
    balance: 0, monthly_income: 0, monthly_expense: 0,
    monthly_history: [],
    recent_transactions: [],
    budget: { total_budgeted: 0, total_spent: 0, total_expense: 0, percentage_spent: 0, raw_percentage: 0, is_over_budget: false, has_budgets: false }
  };

  // ── Compute sparkline bar percentages (proportional to real data) ─────────
  const savingsRate = d.monthly_income > 0
    ? Math.max(0, Math.round(((d.monthly_income - d.monthly_expense) / d.monthly_income) * 100))
    : null;

  // Income bar: how this month stacks vs peak of last 6 months
  const maxMonthlyIncome = Math.max(
    ...( d.monthly_history?.map(m => m.income) || []),
    d.monthly_income, 1
  );
  const incomeBarPct = Math.min(100, (d.monthly_income / maxMonthlyIncome) * 100);

  // Expense bar: % of monthly income spent (how much of income is going out)
  const expenseBarPct = d.monthly_income > 0
    ? Math.min(100, (d.monthly_expense / d.monthly_income) * 100)
    : 0;

  const stats = [
    {
      title: 'Net Savings',
      value: d.monthly_income - d.monthly_expense,
      subtitle: 'Saved this month',
      icon: Wallet,
      gradient: 'from-violet-600 to-purple-900',
      glow: '0 4px 20px rgba(124,58,237,0.5)',
      barPct: savingsRate !== null ? savingsRate : 0,
    },
    {
      title: 'Monthly Income',
      value: d.monthly_income,
      subtitle: 'Earned this month',
      icon: TrendingUp,
      gradient: 'from-cyan-500 to-emerald-700',
      glow: '0 4px 20px rgba(6,182,212,0.5)',
      barPct: incomeBarPct,
    },
    {
      title: 'Monthly Expense',
      value: d.monthly_expense,
      subtitle: d.monthly_income > 0
        ? `${Math.round((d.monthly_expense / d.monthly_income) * 100)}% of income`
        : 'Spent this month',
      icon: TrendingDown,
      gradient: 'from-pink-600 to-rose-900',
      glow: '0 4px 20px rgba(219,39,119,0.5)',
      barPct: expenseBarPct,
    },
  ];

  // Chart uses real 6-month history from the API
  const chartData = d.monthly_history?.length > 0
    ? d.monthly_history
    : [{ name: 'Now', income: d.monthly_income, expense: d.monthly_expense }];

  return (
    <motion.div
      ref={containerRef}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white pb-16"
    >
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.7 }}
        className="flex items-center justify-between"
      >
        <div>
          <p className="text-white/40 text-sm font-sans mb-1">
            {user?.username
              ? `Welcome back, ${user.username.charAt(0).toUpperCase() + user.username.slice(1)} 👋`
              : 'Welcome back 👋'}
          </p>
          <h1 className="text-4xl font-black tracking-tight font-sans bg-gradient-to-r from-white via-white to-white/60 bg-clip-text text-transparent">
            Financial Command Center
          </h1>
          <p className="text-white/40 text-sm mt-1 font-sans">
            {d.monthly_income > 0 || d.monthly_expense > 0
              ? `This month: ₹${d.monthly_income.toLocaleString('en-IN')} earned · ₹${d.monthly_expense.toLocaleString('en-IN')} spent · Total Balance: ₹${d.balance.toLocaleString('en-IN')}`
              : `Your personal wealth dashboard · Total Balance: ₹${d.balance.toLocaleString('en-IN')}`}
          </p>
        </div>

        {/* Savings rate chip — only shows when there's income */}
        {savingsRate !== null && (
          <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.5 }}
            className="hidden lg:flex flex-col items-center gap-0.5 px-5 py-3 rounded-2xl border border-brand-purple/20 font-sans"
            style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(219,39,119,0.06))' }}
          >
            <span className="text-[10px] text-white/35 uppercase tracking-widest font-bold">Savings Rate</span>
            <span className="text-2xl font-black text-white leading-none">{savingsRate}%</span>
            <span className="text-[10px] font-semibold"
              style={{ color: d.monthly_income > d.monthly_expense ? '#34d399' : '#f43f5e' }}
            >
              {d.monthly_income > d.monthly_expense ? '▲ saving' : '▼ over budget'}
            </span>
          </motion.div>
        )}
      </motion.div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((s, i) => (
          <StatCard key={s.title} {...s} index={i} />
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Area Chart — real historical data */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-white/5 shadow-xl"
        >
          <div className="flex items-start justify-between mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <BarChart3 size={16} className="text-brand-cyan" />
                <h3 className="text-base font-bold text-white font-sans">Cash Flow Timeline</h3>
              </div>
              <p className="text-white/40 text-xs font-sans">Monthly income vs expense comparison</p>
            </div>
            <div className="flex gap-4 text-xs font-semibold">
              <span className="flex items-center gap-1.5 text-brand-cyan"><span className="w-3 h-0.5 bg-brand-cyan rounded-full" />Income</span>
              <span className="flex items-center gap-1.5 text-brand-pink"><span className="w-3 h-0.5 bg-brand-pink rounded-full" />Expense</span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="incomeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.5} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="expenseGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#db2777" stopOpacity={0.5} />
                    <stop offset="95%" stopColor="#db2777" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" stroke="rgba(255,255,255,0.25)" fontSize={11} tickLine={false} />
                <YAxis stroke="rgba(255,255,255,0.25)" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `₹${(v/1000).toFixed(0)}k`} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="income" name="Income" stroke="#06b6d4" strokeWidth={2.5} fillOpacity={1} fill="url(#incomeGrad)" dot={false} />
                <Area type="monotone" dataKey="expense" name="Expense" stroke="#db2777" strokeWidth={2.5} fillOpacity={1} fill="url(#expenseGrad)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Budget Status Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="glass-panel rounded-2xl p-6 border border-white/5 shadow-xl flex flex-col"
        >
          <div className="flex items-center gap-2 mb-1">
            <Zap size={16} className="text-brand-purple" />
            <h3 className="text-base font-bold text-white font-sans">Budget Status</h3>
          </div>
          <p className="text-white/40 text-xs mb-4 font-sans">Spending vs set budget limits this month</p>
          <BudgetRing budget={d.budget} />
        </motion.div>
      </div>

      {/* Recent Transactions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass-panel rounded-2xl p-6 border border-white/5 shadow-xl"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Sparkles size={16} className="text-brand-purple" />
              <h3 className="text-base font-bold text-white font-sans">Recent Transactions</h3>
            </div>
            <p className="text-white/40 text-xs font-sans">Latest 5 financial activity entries</p>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          {d.recent_transactions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center">
                <Activity size={24} className="text-white/20" />
              </div>
              <p className="text-white/30 text-sm font-sans">No transactions yet</p>
            </div>
          ) : (
            d.recent_transactions.map((tx, i) => (
              <motion.div
                key={tx.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.9 + i * 0.08 }}
                className="flex items-center justify-between p-4 rounded-xl bg-white/3 border border-white/5 hover:border-brand-purple/20 hover:bg-white/6 transition-all group cursor-default"
              >
                <div className="flex items-center gap-4">
                  <div className={`w-11 h-11 rounded-xl flex items-center justify-center ${
                    tx.type === 'income'
                      ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
                      : 'bg-rose-500/10 border border-rose-500/20 text-rose-400'
                  }`}>
                    <ArrowUpRight size={18} className={tx.type === 'income' ? '' : 'rotate-180'} />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white group-hover:text-brand-purple transition-colors font-sans">
                      {tx.description}
                    </p>
                    <span className="text-xs text-white/40 font-medium font-sans">
                      {tx.category?.name || 'Uncategorized'} &bull; {tx.date}
                    </span>
                  </div>
                </div>
                <span className={`font-black text-sm font-sans ${tx.type === 'income' ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {tx.type === 'income' ? '+' : '−'}₹{(tx.amount || 0).toLocaleString('en-IN')}
                </span>
              </motion.div>
            ))
          )}
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Dashboard;
