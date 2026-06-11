import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Target, Plus, Save, AlertTriangle } from 'lucide-react';

export const Budget = () => {
  const [budgets, setBudgets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState({});
  const today = new Date();
  const [month] = useState(today.getMonth() + 1);
  const [year] = useState(today.getFullYear());

  const fetchBudgets = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/budgets?month=${month}&year=${year}`);
      const data = await res.json();
      setBudgets(data.budgets || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchBudgets(); }, []);

  const handleSave = async (catId, amount) => {
    setSaving(prev => ({ ...prev, [catId]: true }));
    try {
      await fetch(`/api/budgets?month=${month}&year=${year}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category_id: catId, amount: parseFloat(amount) || 0 })
      });
      await fetchBudgets();
    } finally {
      setSaving(prev => ({ ...prev, [catId]: false }));
    }
  };

  const [localAmounts, setLocalAmounts] = useState({});
  useEffect(() => {
    const initial = {};
    budgets.forEach(b => { initial[b.category_id] = b.amount; });
    setLocalAmounts(initial);
  }, [budgets]);

  const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-4xl flex flex-col gap-8 text-white pb-16"
    >
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Target size={16} className="text-brand-cyan" />
          <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">Planning</span>
        </div>
        <h1 className="text-4xl font-black tracking-tight font-sans">Budget Planner</h1>
        <p className="text-white/40 text-sm mt-1 font-sans">
          Set spending limits for {monthNames[month - 1]} {year}
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col gap-3">
          {[...Array(5)].map((_, i) => <div key={i} className="h-20 rounded-xl bg-white/5 animate-pulse" />)}
        </div>
      ) : budgets.length === 0 ? (
        <div className="glass-panel rounded-2xl p-12 border border-white/5 flex flex-col items-center justify-center text-center gap-3">
          <Target size={40} className="text-white/20" />
          <p className="text-white/40 font-sans">No expense categories found. Add categories first to set budgets.</p>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl p-6 border border-white/5 flex flex-col gap-4">
          {budgets.map((b, i) => {
            const pct = b.amount > 0 ? Math.min((b.spent / b.amount) * 100, 100) : 0;
            const isOver = b.spent > b.amount && b.amount > 0;
            const barColor = isOver ? '#db2777' : pct > 80 ? '#f59e0b' : '#7c3aed';

            return (
              <motion.div
                key={b.category_id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="p-4 rounded-xl bg-white/4 border border-white/5 flex flex-col gap-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-white font-sans">{b.category_name}</span>
                    {isOver && <AlertTriangle size={13} className="text-rose-400" />}
                  </div>
                  <span className="text-xs text-white/40 font-sans">
                    ₹{b.spent.toLocaleString()} / ₹{(localAmounts[b.category_id] || b.amount).toLocaleString()} spent
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="h-1.5 bg-white/8 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.8, delay: i * 0.05 + 0.3 }}
                    className="h-full rounded-full"
                    style={{ background: barColor }}
                  />
                </div>

                {/* Budget input + save */}
                <div className="flex items-center gap-3">
                  <div className="relative flex-1">
                    <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 text-sm">₹</span>
                    <input
                      type="number"
                      min="0"
                      value={localAmounts[b.category_id] ?? b.amount}
                      onChange={e => setLocalAmounts(prev => ({ ...prev, [b.category_id]: e.target.value }))}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-7 pr-3 text-white text-sm focus:outline-none focus:border-brand-purple focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans"
                    />
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                    onClick={() => handleSave(b.category_id, localAmounts[b.category_id])}
                    disabled={saving[b.category_id]}
                    className="px-4 py-2.5 rounded-xl bg-brand-purple/20 border border-brand-purple/30 text-brand-purple text-xs font-bold hover:bg-brand-purple/30 transition-all font-sans flex items-center gap-1.5"
                  >
                    <Save size={12} />
                    {saving[b.category_id] ? '...' : 'Set'}
                  </motion.button>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
};

export default Budget;
