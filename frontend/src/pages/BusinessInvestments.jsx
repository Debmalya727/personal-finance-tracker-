import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Plus, Trash2, ShieldAlert, Award, Briefcase, Calendar, Percent, PlusCircle } from 'lucide-react';

export const BusinessInvestments = () => {
  const [investments, setInvestments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    investment_name: '',
    investment_type: 'Capital Asset',
    amount_invested: '',
    purchase_date: new Date().toISOString().split('T')[0],
    useful_life_years: ''
  });
  const [error, setError] = useState(null);

  const fetchInvestments = () => {
    fetch('/api/business/investments')
      .then(res => res.json())
      .then(data => {
        setInvestments(data.investments || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load investments:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchInvestments();
  }, []);

  const handleAdd = (e) => {
    e.preventDefault();
    setError(null);

    const payload = {
      ...form,
      amount_invested: parseFloat(form.amount_invested),
      useful_life_years: form.useful_life_years ? parseInt(form.useful_life_years) : null
    };

    fetch('/api/business/investments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        if (data.error) {
          setError(data.error);
        } else {
          setForm({
            investment_name: '',
            investment_type: 'Capital Asset',
            amount_invested: '',
            purchase_date: new Date().toISOString().split('T')[0],
            useful_life_years: ''
          });
          fetchInvestments();
        }
      })
      .catch(err => {
        console.error(err);
        setError('Connection error');
      });
  };

  const handleDelete = (id) => {
    if (!window.confirm('Delete this investment?')) return;
    fetch(`/api/business/investments/${id}`, { method: 'DELETE' })
      .then(res => res.json())
      .then(() => fetchInvestments())
      .catch(err => console.error(err));
  };

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-12 h-12 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
      </div>
    );
  }

  const totalValue = investments.reduce((acc, curr) => acc + curr.current_value, 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col lg:grid lg:grid-cols-5 gap-8 text-white relative z-10 font-sans"
    >
      {/* Header and Left Content (3 columns) */}
      <div className="lg:col-span-3 flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Business Capital Assets</h1>
          <p className="text-white/40 text-sm mt-1">
            Track business equipment, property, and investments with auto-calculated depreciation models.
          </p>
        </div>

        {/* Aggregate Stats Card */}
        <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl flex items-center justify-between bg-gradient-to-r from-brand-purple/15 to-brand-pink/5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-brand-purple/20 border border-brand-purple/30 flex items-center justify-center glow-purple">
              <Briefcase size={22} className="text-brand-purple animate-bounce" />
            </div>
            <div>
              <span className="text-xs text-white/50 font-bold uppercase tracking-wider block">Total Capital Value</span>
              <h2 className="text-3xl font-extrabold tracking-tight mt-0.5">
                ₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
              </h2>
            </div>
          </div>
        </div>

        {/* Investment List */}
        <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4 bg-white/2">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white/50">Capital Asset Portfolio</h3>
          
          <div className="flex flex-col gap-3">
            <AnimatePresence>
              {investments.length === 0 ? (
                <div className="text-center py-8 text-white/30 text-xs">
                  No business investments or assets registered. Use the panel on the right to add assets.
                </div>
              ) : (
                investments.map((inv) => (
                  <motion.div
                    key={inv.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    className="flex justify-between items-center p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all font-sans relative group"
                  >
                    <div className="flex flex-col gap-1">
                      <span className="text-sm font-bold">{inv.investment_name}</span>
                      <div className="flex items-center gap-3 text-[10px] text-white/40">
                        <span className="px-1.5 py-0.5 rounded bg-brand-purple/15 text-brand-purple border border-brand-purple/20">{inv.investment_type}</span>
                        <span>Purchased: {inv.purchase_date}</span>
                        {inv.useful_life_years && (
                          <span>Life: {inv.useful_life_years} Years</span>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <span className="text-sm font-bold block">₹{inv.current_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                        <span className="text-[10px] text-white/40 block">Cost: ₹{inv.amount_invested.toLocaleString()}</span>
                      </div>
                      
                      <button
                        onClick={() => handleDelete(inv.id)}
                        className="p-2 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-rose-400 opacity-0 group-hover:opacity-100 transition-all"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Right Add Form (2 columns) */}
      <div className="lg:col-span-2 flex flex-col gap-6">
        <div className="glass-panel p-6 rounded-2xl border border-white/5 bg-white/2 flex flex-col gap-4">
          <div>
            <h3 className="text-lg font-bold">Register Capital Asset</h3>
            <p className="text-white/40 text-xs mt-0.5">
              Add business property, machinery, financial investments, or assets.
            </p>
          </div>

          <form onSubmit={handleAdd} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Asset Name</label>
              <input
                type="text"
                required
                placeholder="e.g. Office Macbook, Server Rack, Fixed Deposit"
                value={form.investment_name}
                onChange={(e) => setForm({ ...form, investment_name: e.target.value })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              />
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Asset Type</label>
              <select
                value={form.investment_type}
                onChange={(e) => setForm({ ...form, investment_type: e.target.value })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              >
                <option value="Capital Asset">Capital Asset (Depreciable)</option>
                <option value="Financial">Financial Investment</option>
                <option value="Inventory">Inventory</option>
                <option value="Intellectual Property">Intellectual Property</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Amount Invested</label>
                <input
                  type="number"
                  required
                  placeholder="₹"
                  value={form.amount_invested}
                  onChange={(e) => setForm({ ...form, amount_invested: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Purchase Date</label>
                <input
                  type="date"
                  required
                  value={form.purchase_date}
                  onChange={(e) => setForm({ ...form, purchase_date: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>
            </div>

            {form.investment_type === 'Capital Asset' && (
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Useful Life (Years)</label>
                <input
                  type="number"
                  placeholder="e.g. 5 (for MacBook)"
                  value={form.useful_life_years}
                  onChange={(e) => setForm({ ...form, useful_life_years: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>
            )}

            {error && (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
                <ShieldAlert size={14} />
                <span>{error}</span>
              </div>
            )}

            <button
              type="submit"
              className="mt-2 w-full py-3 rounded-xl font-bold bg-gradient-to-r from-brand-purple to-brand-pink text-white flex items-center justify-center gap-2 shadow-lg shadow-brand-purple/20 hover:shadow-brand-purple/35 transition-all text-sm"
            >
              <PlusCircle size={16} />
              <span>Add Capital Asset</span>
            </button>
          </form>
        </div>
      </div>
    </motion.div>
  );
};

export default BusinessInvestments;
