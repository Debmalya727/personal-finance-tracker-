import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Home, ArrowRightLeft, TrendingUp, Wallet, Briefcase,
  DollarSign, Percent, LogOut, Users, FileText, Sparkles,
  Activity, Tag, Target, IndianRupee, UserCircle, Brain,
  BarChart3, ChevronDown, ChevronRight
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const NavItem = ({ item, isActive }) => {
  const Icon = item.icon;
  return (
    <Link
      to={item.path}
      className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 relative group font-sans ${
        isActive ? 'text-white font-semibold' : 'text-white/50 hover:text-white hover:bg-white/5'
      }`}
    >
      {isActive && (
        <motion.div
          layoutId="activeIndicator"
          className="absolute inset-0 bg-gradient-to-r from-brand-purple/20 to-brand-pink/5 border-l-2 border-brand-purple rounded-xl -z-10"
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        />
      )}
      <Icon size={16} className={isActive ? 'text-brand-purple' : 'text-white/35 group-hover:text-white/70'} />
      <span>{item.name}</span>
    </Link>
  );
};

const NavSection = ({ title, items, location }) => {
  const [open, setOpen] = useState(true);
  const hasActive = items.some(i => location.pathname === i.path);

  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-3 py-1.5 mb-0.5 text-[10px] font-black uppercase tracking-widest text-white/25 hover:text-white/40 transition-colors"
      >
        <span>{title}</span>
        {open ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden flex flex-col gap-0.5"
          >
            {items.map(item => (
              <NavItem key={item.name} item={item} isActive={location.pathname === item.path} />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const Sidebar = ({ isBusiness, setIsBusiness, mobileOpen, setMobileOpen }) => {
  const location = useLocation();
  const { logout, user } = useAuth();

  const personalCore = [
    { name: 'Dashboard', path: '/dashboard', icon: Home },
    { name: 'Transactions', path: '/transactions', icon: ArrowRightLeft },
    { name: 'Investments', path: '/investments', icon: TrendingUp },
    { name: 'Net Worth', path: '/net-worth', icon: Wallet },
  ];

  const personalTools = [
    { name: 'Budget Planner', path: '/budget', icon: Target },
    { name: 'Fixed Schemes', path: '/schemes', icon: Briefcase },
    { name: 'Loans & EMIs', path: '/loans', icon: DollarSign },
    { name: 'Tax Estimator', path: '/tax', icon: Percent },
  ];

  const personalExtra = [
    { name: 'Categories', path: '/categories', icon: Tag },
    { name: 'Salary Manager', path: '/salary', icon: IndianRupee },
    { name: 'AI Insights', path: '/ai-insights', icon: Brain },
    { name: 'Reports', path: '/reports', icon: BarChart3 },
    { name: 'Profile', path: '/profile', icon: UserCircle },
  ];

  const businessItems = [
    { name: 'Dashboard', path: '/business', icon: Activity },
    { name: 'Invoices & Txs', path: '/business/transactions', icon: FileText },
    { name: 'Clients', path: '/business/clients', icon: Users },
    { name: 'Financials', path: '/business/financials', icon: Wallet },
    { name: 'Capital Assets', path: '/business/investments', icon: Briefcase },
    { name: 'Loans & EMIs', path: '/business/loans', icon: DollarSign },
    { name: 'Reports', path: '/business/reports', icon: BarChart3 },
    { name: 'AI Insights', path: '/business/insights', icon: Brain },
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <div 
        className={`w-64 h-screen fixed top-0 z-40 flex flex-col transition-all duration-300 lg:left-0 ${mobileOpen ? 'left-0' : '-left-64'}`}
        style={{
          background: 'rgba(8, 7, 26, 0.85)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        {/* Brand */}
      <div className="px-5 pt-6 pb-4 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #db2777)', boxShadow: '0 0 16px rgba(124,58,237,0.4)' }}>
            <TrendingUp size={18} className="text-white" />
          </div>
          <div>
            <h1 className="font-black text-base leading-none text-white font-sans">FinVision</h1>
            <span className="text-[9px] text-white/30 tracking-widest uppercase font-sans">3D Wealth Portal</span>
          </div>
        </div>

        {/* User pill */}
        {user && (
          <div className="mt-4 p-2.5 rounded-xl bg-white/4 border border-white/5 flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.4), rgba(219,39,119,0.3))' }}>
              <span className="text-xs font-black text-white uppercase">{user.username[0]}</span>
            </div>
            <div className="min-w-0">
              <p className="text-xs font-bold text-white leading-tight truncate font-sans">{user.username}</p>
              <p className="text-[10px] text-white/35 capitalize font-sans">{user.role} mode</p>
            </div>
          </div>
        )}
      </div>

      {/* Nav */}
      <div className="flex-1 overflow-y-auto py-3 px-3 flex flex-col gap-1">
        {isBusiness ? (
          <NavSection title="Business" items={businessItems} location={location} />
        ) : (
          <>
            <NavSection title="Overview" items={personalCore} location={location} />
            <NavSection title="Planning" items={personalTools} location={location} />
            <NavSection title="Tools" items={personalExtra} location={location} />
          </>
        )}

        {/* Business Toggle */}
        {user?.role === 'business' && (
          <button
            onClick={() => setIsBusiness(!isBusiness)}
            className="mt-2 w-full p-3 rounded-xl border border-brand-purple/20 bg-brand-purple/8 flex items-center justify-between text-xs font-bold text-white hover:bg-brand-purple/15 transition-all font-sans"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={13} className="text-brand-purple" />
              <span>{isBusiness ? 'Business Mode' : 'Personal Mode'}</span>
            </div>
            <span className="px-1.5 py-0.5 rounded bg-brand-purple/25 text-brand-purple uppercase text-[9px] tracking-widest">Switch</span>
          </button>
        )}
      </div>

      {/* Logout Footer */}
      <div className="px-3 pb-4 border-t border-white/5 pt-3">
        <button
          onClick={logout}
          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium text-white/40 hover:text-rose-400 hover:bg-rose-500/8 transition-all border border-transparent hover:border-rose-500/15 font-sans"
        >
          <LogOut size={15} />
          <span>Sign Out</span>
        </button>
      </div>
    </div>
    </>
  );
};

export default Sidebar;
