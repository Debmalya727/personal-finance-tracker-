import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, TrendingDown, Users, FileText, 
  Plus, Check, AlertCircle, Briefcase, Activity
} from 'lucide-react';
import { ThreeDCard } from '../components/ThreeDCard';

export const BusinessConsole = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/business/dashboard')
      .then((res) => res.json())
      .then((resData) => {
        setData(resData);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
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

  const bizData = data || {
    clients_count: 0,
    total_revenue: 0,
    total_expense: 0,
    net_profit: 0,
    total_outstanding: 0,
    recent_transactions: []
  };

  const cards = [
    { title: 'Net Profit', value: `₹${bizData.net_profit.toLocaleString()}`, change: 'Net Margin', color: 'from-brand-purple to-purple-600', glow: 'glow-purple', icon: Activity },
    { title: 'Total Revenue', value: `₹${bizData.total_revenue.toLocaleString()}`, change: 'Total Cash Inflow', color: 'from-brand-cyan to-emerald-600', glow: 'glow-cyan', icon: TrendingUp },
    { title: 'Receivables', value: `₹${bizData.total_outstanding.toLocaleString()}`, change: 'Unpaid Invoices', color: 'from-brand-pink to-rose-600', glow: 'glow-pink', icon: FileText },
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white "
    >
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight font-sans">
          Business Command Console
        </h1>
        <p className="text-white/40 text-sm mt-1 font-sans">
          Overview of client accounts, invoices, profit margins, and invoice statuses.
        </p>
      </div>

      {/* 3D Tilting Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {cards.map((card, idx) => {
          const Icon = card.icon;
          return (
            <ThreeDCard key={idx} className="h-44">
              <div className="flex items-start justify-between h-full flex-col">
                <div className="flex items-center justify-between w-full">
                  <span className="text-sm font-semibold text-white/50 uppercase tracking-wider font-sans">
                    {card.title}
                  </span>
                  <div className={`p-2 rounded-xl bg-gradient-to-tr ${card.color} shadow-lg ${card.glow}`}>
                    <Icon size={18} className="text-white" />
                  </div>
                </div>
                <div>
                  <h3 className="text-3xl font-extrabold mt-2 tracking-tight text-white font-sans">
                    {card.value}
                  </h3>
                  <span className="text-xs text-white/40 font-medium block mt-1 font-sans">
                    {card.change}
                  </span>
                </div>
              </div>
            </ThreeDCard>
          );
        })}
      </div>

      {/* Active invoices & operations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Recent Invoices list (2 columns) */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl">
          <h3 className="text-base font-bold mb-4 font-sans flex items-center gap-2">
            <FileText size={16} className="text-brand-purple" />
            <span>Outstanding Invoices</span>
          </h3>

          <div className="flex flex-col gap-3.5">
            {bizData.recent_transactions.length === 0 ? (
              <div className="text-center py-12 text-white/30 text-sm font-sans">
                No active business invoices found.
              </div>
            ) : (
              bizData.recent_transactions.map((tx) => (
                <div 
                  key={tx.id} 
                  className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5 hover:border-brand-purple/20 transition-all font-sans group"
                >
                  <div>
                    <h4 className="font-semibold text-sm text-white group-hover:text-brand-purple transition-all">{tx.description}</h4>
                    <p className="text-xs text-white/40 mt-1">
                      Client: {tx.client ? tx.client.name : 'Direct Sale'} • Category: {tx.category ? tx.category.name : 'Service'}
                    </p>
                  </div>

                  <div className="flex items-center gap-6">
                    <span className={`px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
                      tx.invoice_status === 'paid' 
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                        : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20 animate-pulse'
                    }`}>
                      {tx.invoice_status}
                    </span>
                    <span className="font-bold text-sm text-white">
                      ₹{tx.amount.toLocaleString()}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Dynamic client quick look */}
        <div className="glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold font-sans flex items-center gap-2 mb-1">
              <Users size={16} className="text-brand-cyan" />
              <span>Accounts Directory</span>
            </h3>
            <p className="text-white/40 text-xs font-sans">List profile summaries of clients registered with your account.</p>
          </div>

          <div className="flex flex-col items-center justify-center my-6">
            <div className="w-20 h-20 rounded-full bg-brand-cyan/10 border border-brand-cyan/30 flex items-center justify-center text-brand-cyan shadow-lg shadow-cyan-500/10 mb-3 animate-pulse">
              <Users size={32} />
            </div>
            <h4 className="text-2xl font-black text-white font-sans">{bizData.clients_count}</h4>
            <span className="text-xs text-white/40 font-semibold uppercase tracking-wider mt-1.5 font-sans">Registered Clients</span>
          </div>

          <button
            onClick={() => window.location.href = '/business/clients'}
            className="w-full bg-white/5 border border-white/5 text-white/80 hover:text-white py-2 rounded-xl hover:bg-white/10 transition-all text-xs font-semibold uppercase tracking-wider font-sans"
          >
            Manage Directories
          </button>
        </div>

      </div>
    </motion.div>
  );
};
export default BusinessConsole;

