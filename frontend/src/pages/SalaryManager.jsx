import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { IndianRupee, Save, CheckCircle } from 'lucide-react';

export const SalaryManager = () => {
  const [form, setForm] = useState({ monthly_gross: '', deductions_80c: '', hra_exemption: '' });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch('/api/salary')
      .then(r => r.json())
      .then(d => {
        setForm({
          monthly_gross: d.monthly_gross || '',
          deductions_80c: d.deductions_80c || '',
          hra_exemption: d.hra_exemption || '',
        });
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await fetch('/api/salary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monthly_gross: parseFloat(form.monthly_gross) || 0,
          deductions_80c: parseFloat(form.deductions_80c) || 0,
          hra_exemption: parseFloat(form.hra_exemption) || 0,
        })
      });
      const data = await res.json();
      if (data.success) {
        setSaved(true);
        setTimeout(() => setSaved(false), 3000);
      }
    } finally {
      setSaving(false);
    }
  };

  const annualGross = (parseFloat(form.monthly_gross) || 0) * 12;
  const totalDeductions = (parseFloat(form.deductions_80c) || 0) + (parseFloat(form.hra_exemption) || 0);
  const estimatedTaxable = Math.max(0, annualGross - totalDeductions - 50000); // std deduction

  const fields = [
    { key: 'monthly_gross', label: 'Monthly Gross Salary', placeholder: '50000', desc: 'Your total monthly salary before deductions' },
    { key: 'deductions_80c', label: '80C Deductions (Annual)', placeholder: '150000', desc: 'PPF, ELSS, insurance premiums, etc. (max ₹1.5L)' },
    { key: 'hra_exemption', label: 'HRA Exemption (Annual)', placeholder: '60000', desc: 'House rent allowance exemption claim' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-3xl flex flex-col gap-8 text-white pb-16"
    >
      <div>
        <div className="flex items-center gap-2 mb-1">
          <IndianRupee size={16} className="text-brand-cyan" />
          <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">Compensation</span>
        </div>
        <h1 className="text-4xl font-black tracking-tight font-sans">Salary Manager</h1>
        <p className="text-white/40 text-sm mt-1 font-sans">Configure your salary for auto-crediting and tax calculations</p>
      </div>

      {/* Summary Cards */}
      {!loading && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Annual Gross', value: `₹${annualGross.toLocaleString()}`, color: 'text-brand-cyan' },
            { label: 'Total Deductions', value: `₹${totalDeductions.toLocaleString()}`, color: 'text-brand-purple' },
            { label: 'Est. Taxable', value: `₹${estimatedTaxable.toLocaleString()}`, color: 'text-amber-400' },
          ].map(({ label, value, color }) => (
            <div key={label} className="glass-panel rounded-xl p-4 border border-white/5 text-center">
              <span className="text-xs text-white/40 uppercase tracking-wider font-bold font-sans block">{label}</span>
              <span className={`text-xl font-black mt-1 block font-sans ${color}`}>{value}</span>
            </div>
          ))}
        </div>
      )}

      {/* Form */}
      <div className="glass-panel rounded-2xl p-6 border border-white/5">
        {loading ? (
          <div className="flex flex-col gap-4">
            {[...Array(3)].map((_, i) => <div key={i} className="h-16 rounded-xl bg-white/5 animate-pulse" />)}
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            {fields.map(({ key, label, placeholder, desc }) => (
              <div key={key}>
                <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-1 block font-sans">{label}</label>
                <p className="text-xs text-white/30 mb-2 font-sans">{desc}</p>
                <div className="relative">
                  <span className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30 text-sm font-bold">₹</span>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    value={form[key]}
                    onChange={e => setForm({ ...form, [key]: e.target.value })}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-8 pr-4 text-white text-sm focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all font-sans"
                    placeholder={placeholder}
                  />
                </div>
              </div>
            ))}
            <motion.button
              type="submit"
              disabled={saving}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="mt-2 w-full flex items-center justify-center gap-2 py-3.5 rounded-xl font-bold text-white text-sm font-sans transition-all"
              style={{ background: saved ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #06b6d4, #7c3aed)', boxShadow: '0 0 20px rgba(6,182,212,0.3)' }}
            >
              {saved ? <><CheckCircle size={16} />Saved!</> : saving ? 'Saving...' : <><Save size={16} />Save Salary Details</>}
            </motion.button>
          </form>
        )}
      </div>

      <div className="glass-panel rounded-xl p-4 border border-amber-500/20 bg-amber-500/5">
        <p className="text-xs text-amber-400/80 font-medium font-sans">
          💡 Once set, your monthly salary (₹{(parseFloat(form.monthly_gross) || 0).toLocaleString()}) will be auto-credited as an income transaction on the 1st of each month.
        </p>
      </div>
    </motion.div>
  );
};

export default SalaryManager;
