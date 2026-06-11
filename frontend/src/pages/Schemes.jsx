import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Plus, Briefcase, Trash2, ShieldCheck } from 'lucide-react';

export const Schemes = () => {
  const [schemes, setSchemes] = useState([]);
  const [loading, setLoading] = useState(true);

  // Form State
  const [name, setName] = useState('');
  const [principal, setPrincipal] = useState('');
  const [rate, setRate] = useState('');
  const [tenure, setTenure] = useState('');
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [penalty, setPenalty] = useState('1.0');

  const fetchSchemes = () => {
    setLoading(true);
    fetch('/api/schemes')
      .then((res) => res.json())
      .then((data) => {
        setSchemes(data.schemes);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchSchemes();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/schemes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scheme_name: name,
          principal_amount: parseFloat(principal),
          interest_rate: parseFloat(rate),
          tenure_months: parseInt(tenure),
          start_date: startDate,
          penalty_rate: parseFloat(penalty)
        })
      });
      if (response.ok) {
        setName('');
        setPrincipal('');
        setRate('');
        setTenure('');
        fetchSchemes();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this investment scheme?')) return;
    try {
      const response = await fetch(`/api/schemes/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchSchemes();
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white "
    >
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight font-sans">
          Fixed Investment Schemes
        </h1>
        <p className="text-white/40 text-sm mt-1 font-sans">
          Monitor your fixed deposits, recurring deposits, or provident fund assets.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Add Scheme Form */}
        <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl flex flex-col gap-4">
          <h3 className="text-base font-bold font-sans flex items-center gap-2">
            <Plus size={16} className="text-brand-purple" />
            <span>Add New Scheme</span>
          </h3>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4 font-sans text-sm">
            <div>
              <label className="text-xs text-white/50 mb-1.5 block font-semibold">Scheme Name</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. SBI Fixed Deposit"
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Principal (INR)</label>
                <input
                  type="number"
                  required
                  value={principal}
                  onChange={(e) => setPrincipal(e.target.value)}
                  placeholder="0.00"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Interest Rate (%)</label>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={rate}
                  onChange={(e) => setRate(e.target.value)}
                  placeholder="e.g. 7.1"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Tenure (Months)</label>
                <input
                  type="number"
                  required
                  value={tenure}
                  onChange={(e) => setTenure(e.target.value)}
                  placeholder="e.g. 12"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Penalty Rate (%)</label>
                <input
                  type="number"
                  step="0.01"
                  required
                  value={penalty}
                  onChange={(e) => setPenalty(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
            </div>
            <div>
              <label className="text-xs text-white/50 mb-1.5 block font-semibold">Start Date</label>
              <input
                type="date"
                required
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
              />
            </div>
            <button
              type="submit"
              className="w-full mt-2 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-bold py-2.5 rounded-xl hover:brightness-110 active:scale-[0.98] transition-all glow-purple"
            >
              Add Scheme
            </button>
          </form>
        </div>

        {/* Schemes List (2 columns) */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl flex flex-col gap-4">
          <h3 className="text-base font-bold font-sans flex items-center gap-2">
            <Briefcase size={16} className="text-brand-cyan" />
            <span>Active Schemes</span>
          </h3>

          {loading ? (
            <div className="py-12 flex justify-center">
              <div className="w-8 h-8 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
            </div>
          ) : schemes.length === 0 ? (
            <div className="py-12 text-center text-white/30 text-sm font-sans">
              No active schemes recorded yet.
            </div>
          ) : (
            <div className="flex flex-col gap-3.5">
              {schemes.map((s) => (
                <div 
                  key={s.id} 
                  className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5 hover:border-brand-purple/20 transition-all font-sans group"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/25 flex items-center justify-center text-cyan-400">
                      <ShieldCheck size={18} />
                    </div>
                    <div>
                      <h4 className="font-semibold text-sm text-white group-hover:text-brand-purple transition-all">{s.scheme_name}</h4>
                      <p className="text-xs text-white/40 mt-0.5">
                        Principal: ₹{s.principal_amount.toLocaleString()} • {s.interest_rate}% Yield • {s.tenure_months}m
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={() => handleDelete(s.id)}
                    className="p-1.5 text-white/30 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg border border-transparent hover:border-rose-500/20 transition-all"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </motion.div>
  );
};
export default Schemes;

