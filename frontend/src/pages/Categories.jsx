import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Tag, Plus, Trash2, FolderOpen } from 'lucide-react';

export const Categories = () => {
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState({ name: '', type: 'expense' });
  const [submitting, setSubmitting] = useState(false);

  const fetchCategories = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/categories');
      const data = await res.json();
      setCategories(data.categories || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchCategories(); }, []);

  const handleAdd = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const res = await fetch('/api/categories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      });
      const data = await res.json();
      if (data.success) {
        setForm({ name: '', type: 'expense' });
        setShowAdd(false);
        fetchCategories();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('Delete this category? Transactions will become uncategorized.')) return;
    await fetch(`/api/categories/${id}`, { method: 'DELETE' });
    fetchCategories();
  };

  const income = categories.filter(c => c.type === 'income');
  const expense = categories.filter(c => c.type === 'expense');

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-4xl flex flex-col gap-8 text-white pb-16"
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Tag size={16} className="text-brand-purple" />
            <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">Organization</span>
          </div>
          <h1 className="text-4xl font-black tracking-tight font-sans">Categories</h1>
          <p className="text-white/40 text-sm mt-1 font-sans">Manage income and expense categories for your transactions</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowAdd(!showAdd)}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-white text-sm font-sans"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #db2777)', boxShadow: '0 0 20px rgba(124,58,237,0.3)' }}
        >
          <Plus size={16} />
          Add Category
        </motion.button>
      </div>

      {/* Add Form */}
      <AnimatePresence>
        {showAdd && (
          <motion.form
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            onSubmit={handleAdd}
            className="glass-panel rounded-2xl p-6 border border-brand-purple/20 flex flex-col gap-4"
          >
            <h3 className="text-sm font-bold text-white uppercase tracking-wider font-sans">New Category</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">Name</label>
                <input
                  type="text"
                  value={form.name}
                  onChange={e => setForm({ ...form, name: e.target.value })}
                  required
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-white text-sm focus:outline-none focus:border-brand-purple focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans"
                  placeholder="e.g. Groceries"
                />
              </div>
              <div>
                <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">Type</label>
                <select
                  value={form.type}
                  onChange={e => setForm({ ...form, type: e.target.value })}
                  className="w-full bg-[#0d0b25] border border-white/10 rounded-xl py-3 px-4 text-white text-sm focus:outline-none focus:border-brand-purple focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans"
                >
                  <option value="expense">Expense</option>
                  <option value="income">Income</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3">
              <button type="submit" disabled={submitting}
                className="px-6 py-2.5 rounded-xl bg-brand-purple text-white text-sm font-bold hover:bg-purple-600 transition-all font-sans">
                {submitting ? 'Adding...' : 'Add Category'}
              </button>
              <button type="button" onClick={() => setShowAdd(false)}
                className="px-6 py-2.5 rounded-xl bg-white/5 text-white/60 text-sm font-bold hover:bg-white/10 transition-all font-sans">
                Cancel
              </button>
            </div>
          </motion.form>
        )}
      </AnimatePresence>

      {loading ? (
        <div className="flex flex-col gap-3">
          {[...Array(6)].map((_, i) => <div key={i} className="h-14 rounded-xl bg-white/5 animate-pulse" />)}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Income Categories */}
          <div className="glass-panel rounded-2xl p-5 border border-white/5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 rounded-full bg-emerald-400" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white/70 font-sans">Income ({income.length})</h3>
            </div>
            {income.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <FolderOpen size={28} className="text-white/15 mb-2" />
                <p className="text-white/25 text-xs font-sans">No income categories yet</p>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {income.map((cat, i) => (
                  <motion.div key={cat.id}
                    initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                    className="flex items-center justify-between p-3 rounded-xl bg-white/4 border border-white/5 hover:border-emerald-500/20 group transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
                      <span className="text-sm font-semibold text-white font-sans">{cat.name}</span>
                    </div>
                    <button onClick={() => handleDelete(cat.id)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-rose-400 hover:bg-rose-500/10 transition-all">
                      <Trash2 size={13} />
                    </button>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {/* Expense Categories */}
          <div className="glass-panel rounded-2xl p-5 border border-white/5">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 rounded-full bg-rose-400" />
              <h3 className="text-sm font-bold uppercase tracking-wider text-white/70 font-sans">Expense ({expense.length})</h3>
            </div>
            {expense.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <FolderOpen size={28} className="text-white/15 mb-2" />
                <p className="text-white/25 text-xs font-sans">No expense categories yet</p>
              </div>
            ) : (
              <div className="flex flex-col gap-2">
                {expense.map((cat, i) => (
                  <motion.div key={cat.id}
                    initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                    className="flex items-center justify-between p-3 rounded-xl bg-white/4 border border-white/5 hover:border-rose-500/20 group transition-all"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-2.5 h-2.5 rounded-full bg-rose-400" />
                      <span className="text-sm font-semibold text-white font-sans">{cat.name}</span>
                    </div>
                    <button onClick={() => handleDelete(cat.id)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg text-rose-400 hover:bg-rose-500/10 transition-all">
                      <Trash2 size={13} />
                    </button>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default Categories;
