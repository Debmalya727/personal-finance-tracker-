import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PlusCircle, Trash2, ShieldAlert, Award, CreditCard, Landmark, Percent } from 'lucide-react';

export const BusinessLoans = () => {
  const [loans, setLoans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState({
    loan_name: '',
    principal_amount: '',
    interest_rate: '',
    tenure_months: '',
    start_date: new Date().toISOString().split('T')[0]
  });
  const [error, setError] = useState(null);

  const fetchLoans = () => {
    fetch('/api/business/loans')
      .then(res => res.json())
      .then(data => {
        setLoans(data.loans || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load loans:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchLoans();
  }, []);

  const handleAdd = (e) => {
    e.preventDefault();
    setError(null);

    const payload = {
      ...form,
      principal_amount: parseFloat(form.principal_amount),
      interest_rate: parseFloat(form.interest_rate),
      tenure_months: parseInt(form.tenure_months)
    };

    fetch('/api/business/loans', {
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
            loan_name: '',
            principal_amount: '',
            interest_rate: '',
            tenure_months: '',
            start_date: new Date().toISOString().split('T')[0]
          });
          fetchLoans();
        }
      })
      .catch(err => {
        console.error(err);
        setError('Connection error');
      });
  };

  const handleDelete = (id) => {
    if (!window.confirm('Delete this loan registry?')) return;
    fetch(`/api/business/loans/${id}`, { method: 'DELETE' })
      .then(res => res.json())
      .then(() => fetchLoans())
      .catch(err => console.error(err));
  };

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-12 h-12 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
      </div>
    );
  }

  const totalOutstanding = loans.reduce((acc, curr) => acc + curr.remaining_balance, 0);

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
          <h1 className="text-3xl font-extrabold tracking-tight">Business Liabilities</h1>
          <p className="text-white/40 text-sm mt-1">
            Manage business loans, lines of credit, outstanding principles, and EMI schedules.
          </p>
        </div>

        {/* Aggregate Stats Card */}
        <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl flex items-center justify-between bg-gradient-to-r from-brand-pink/10 to-transparent">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-brand-pink/20 border border-brand-pink/30 flex items-center justify-center glow-pink">
              <Landmark size={22} className="text-brand-pink animate-bounce" />
            </div>
            <div>
              <span className="text-xs text-white/50 font-bold uppercase tracking-wider block">Total Outstanding Balance</span>
              <h2 className="text-3xl font-extrabold tracking-tight mt-0.5 text-brand-pink">
                ₹{totalOutstanding.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
              </h2>
            </div>
          </div>
        </div>

        {/* Loan List */}
        <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-4 bg-white/2">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white/50">Liability Accounts</h3>
          
          <div className="flex flex-col gap-3">
            <AnimatePresence>
              {loans.length === 0 ? (
                <div className="text-center py-8 text-white/30 text-xs">
                  No registered business loans. Add one using the panel on the right.
                </div>
              ) : (
                loans.map((loan) => (
                  <motion.div
                    key={loan.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    className="flex justify-between items-center p-4 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all font-sans relative group"
                  >
                    <div className="flex flex-col gap-1">
                      <span className="text-sm font-bold">{loan.loan_name}</span>
                      <div className="flex items-center gap-3 text-[10px] text-white/40">
                        <span className="px-1.5 py-0.5 rounded bg-brand-pink/15 text-brand-pink border border-brand-pink/20">{loan.interest_rate}% Interest</span>
                        <span>Tenure: {loan.tenure_months} Months</span>
                        <span>EMI: ₹{loan.emi.toLocaleString('en-IN', { maximumFractionDigits: 0 })}/mo</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <span className="text-sm font-bold block">₹{loan.remaining_balance.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                        <span className="text-[10px] text-white/40 block">Principal: ₹{loan.principal_amount.toLocaleString()}</span>
                      </div>
                      
                      <button
                        onClick={() => handleDelete(loan.id)}
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
            <h3 className="text-lg font-bold">Register Business Loan</h3>
            <p className="text-white/40 text-xs mt-0.5">
              Record a mortgage, commercial loan, line of credit, or debt.
            </p>
          </div>

          <form onSubmit={handleAdd} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Loan Provider / Name</label>
              <input
                type="text"
                required
                placeholder="e.g. HDFC Commercial Loan"
                value={form.loan_name}
                onChange={(e) => setForm({ ...form, loan_name: e.target.value })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Principal Amount</label>
                <input
                  type="number"
                  required
                  placeholder="₹"
                  value={form.principal_amount}
                  onChange={(e) => setForm({ ...form, principal_amount: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Start Date</label>
                <input
                  type="date"
                  required
                  value={form.start_date}
                  onChange={(e) => setForm({ ...form, start_date: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Interest Rate (%)</label>
                <input
                  type="number"
                  step="0.01"
                  required
                  placeholder="e.g. 8.5"
                  value={form.interest_rate}
                  onChange={(e) => setForm({ ...form, interest_rate: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Tenure (Months)</label>
                <input
                  type="number"
                  required
                  placeholder="e.g. 36"
                  value={form.tenure_months}
                  onChange={(e) => setForm({ ...form, tenure_months: e.target.value })}
                  className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
                />
              </div>
            </div>

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
              <span>Register Loan Account</span>
            </button>
          </form>
        </div>
      </div>
    </motion.div>
  );
};

export default BusinessLoans;
