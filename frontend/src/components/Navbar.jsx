import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Calendar, Menu, TrendingUp, TrendingDown } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

const PAGE_META = {
  '/dashboard':             { title: 'Financial Command Center', emoji: '⚡', color: 'text-brand-purple' },
  '/transactions':          { title: 'Ledger Entries',           emoji: '📒', color: 'text-brand-cyan'   },
  '/investments':           { title: 'Assets & Capital',         emoji: '📈', color: 'text-emerald-400'  },
  '/net-worth':             { title: 'Net Worth Overview',       emoji: '🏦', color: 'text-amber-400'    },
  '/schemes':               { title: 'Fixed Investments',        emoji: '🔒', color: 'text-brand-purple' },
  '/loans':                 { title: 'Liabilities',              emoji: '💳', color: 'text-brand-pink'   },
  '/tax':                   { title: 'Tax Optimization',         emoji: '🧮', color: 'text-amber-400'    },
  '/budget':                { title: 'Budget Planner',           emoji: '🎯', color: 'text-brand-cyan'   },
  '/categories':            { title: 'Categories',               emoji: '🏷️', color: 'text-white/60'    },
  '/salary':                { title: 'Salary Manager',           emoji: '💰', color: 'text-emerald-400'  },
  '/ai-insights':           { title: 'AI Insights',              emoji: '🤖', color: 'text-brand-purple' },
  '/reports':               { title: 'Reports & Analytics',      emoji: '📊', color: 'text-brand-cyan'   },
  '/profile':               { title: 'Profile',                  emoji: '👤', color: 'text-white/60'     },
  '/business':              { title: 'Business Console',         emoji: '🏢', color: 'text-brand-purple' },
  '/business/transactions': { title: 'Business Invoices',        emoji: '📑', color: 'text-brand-cyan'   },
  '/business/clients':      { title: 'Client Directory',         emoji: '👥', color: 'text-white/60'     },
  '/business/financials':   { title: 'Business Financials',      emoji: '📈', color: 'text-emerald-400'  },
  '/business/investments':  { title: 'Business Capital Assets',  emoji: '🏭', color: 'text-amber-400'    },
  '/business/loans':        { title: 'Business Liabilities',     emoji: '💳', color: 'text-brand-pink'   },
  '/business/reports':      { title: 'Business Reports',         emoji: '📊', color: 'text-brand-cyan'   },
  '/business/insights':     { title: 'Business AI Analysis',     emoji: '🤖', color: 'text-brand-purple' },
};

export const Navbar = ({ setMobileOpen }) => {
  const location = useLocation();
  const { user } = useAuth();
  const [time, setTime] = useState(new Date());
  const [balance, setBalance] = useState(null);
  const [balanceDelta, setBalanceDelta] = useState(null); // monthly income - expense

  // Live clock — updates every minute
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 60000);
    return () => clearInterval(id);
  }, []);

  // Fetch balance summary once on mount for the navbar chip
  useEffect(() => {
    fetch('/api/dashboard')
      .then(r => r.json())
      .then(d => {
        setBalance(d.balance ?? null);
        setBalanceDelta((d.monthly_income ?? 0) - (d.monthly_expense ?? 0));
      })
      .catch(() => {});
  }, []);

  const pageMeta = PAGE_META[location.pathname] || { title: 'FinVision Portal', emoji: '🚀', color: 'text-white/60' };

  const formattedDate = time.toLocaleDateString('en-IN', {
    weekday: 'short', month: 'short', day: 'numeric'
  });
  const formattedTime = time.toLocaleTimeString('en-IN', {
    hour: '2-digit', minute: '2-digit', hour12: true
  });

  // Format a number as compact INR: ₹1.6L, ₹80K, etc.
  const fmtInr = (n) => {
    if (n === null || n === undefined) return '—';
    const abs = Math.abs(n);
    if (abs >= 100000) return `₹${(n / 100000).toFixed(1)}L`;
    if (abs >= 1000)   return `₹${(n / 1000).toFixed(1)}K`;
    return `₹${Math.round(n)}`;
  };

  const deltaPositive = balanceDelta !== null && balanceDelta >= 0;

  return (
    <header className="h-[70px] w-full lg:w-[calc(100%-16rem)] fixed right-0 top-0 z-30"
      style={{
        background: 'rgba(8, 7, 26, 0.75)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        boxShadow: '0 4px 30px rgba(0,0,0,0.3)',
      }}
    >
      <div className="h-full flex items-center justify-between px-8">
        {/* Left: Mobile hamburger & Page title */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setMobileOpen(prev => !prev)}
            className="p-2 rounded-lg bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-colors lg:hidden"
          >
            <Menu size={16} />
          </button>

          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.35 }}
            className="flex items-center gap-2"
          >
            {/* Page emoji — replaces the useless LIVE badge */}
            <span className="text-lg leading-none">{pageMeta.emoji}</span>
            <h2 className="text-base font-bold text-white tracking-tight font-sans">
              {pageMeta.title}
            </h2>
          </motion.div>
        </div>

        {/* Right: Useful context chips */}
        <div className="flex items-center gap-3">
          {/* Date & time */}
          <div className="hidden md:flex items-center gap-2 px-3.5 py-2 rounded-xl bg-white/4 border border-white/6 text-white/50 text-xs font-medium font-sans">
            <Calendar size={12} className="text-white/30" />
            <span>{formattedDate}</span>
            <span className="text-white/20">·</span>
            <span className="tabular-nums">{formattedTime}</span>
          </div>

          {/* Monthly net flow — replaces the useless Online/Wifi chip */}
          {balanceDelta !== null && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className={`hidden md:flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold font-sans border ${
                deltaPositive
                  ? 'bg-emerald-500/8 border-emerald-500/20 text-emerald-400'
                  : 'bg-rose-500/8 border-rose-500/20 text-rose-400'
              }`}
              title="This month: Income minus Expenses"
            >
              {deltaPositive
                ? <TrendingUp size={12} />
                : <TrendingDown size={12} />
              }
              <span>{deltaPositive ? '+' : ''}{fmtInr(balanceDelta)} this month</span>
            </motion.div>
          )}

          {/* Savings balance chip */}
          {balance !== null && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="hidden lg:flex items-center gap-1.5 px-3 py-2 rounded-xl bg-brand-purple/8 border border-brand-purple/20 text-xs font-bold font-sans text-brand-purple"
              title="Total savings balance"
            >
              <span className="text-white/40 font-medium">Balance</span>
              <span>{fmtInr(balance)}</span>
            </motion.div>
          )}

          {/* User badge */}
          {user && (
            <div
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl border border-brand-purple/25 font-sans"
              style={{
                background: 'linear-gradient(135deg, rgba(124,58,237,0.12) 0%, rgba(219,39,119,0.08) 100%)',
                boxShadow: '0 0 16px rgba(124,58,237,0.12)',
              }}
            >
              <div className="w-6 h-6 rounded-lg bg-gradient-to-tr from-brand-purple to-brand-pink flex items-center justify-center">
                <span className="text-[10px] font-black text-white uppercase">{user.username[0]}</span>
              </div>
              <span className="text-xs text-white/80 font-semibold capitalize hidden md:block">
                {user.username}
              </span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Navbar;
