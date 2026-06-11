import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, Search, Trash2, ArrowUpRight, ArrowDownRight, 
  Sparkles, Camera, X, Check, Loader
} from 'lucide-react';
import { ThreeDCard } from '../components/ThreeDCard';

export const Transactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [categories, setCategories] = useState([]);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [loading, setLoading] = useState(true);
  
  // Modals state
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [isScanOpen, setIsScanOpen] = useState(false);
  
  // Add Form state
  const [description, setDescription] = useState('');
  const [amount, setAmount] = useState('');
  const [type, setType] = useState('expense');
  const [categoryId, setCategoryId] = useState('');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);

  // Scanner state
  const [scanFile, setScanFile] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);

  const fetchTransactions = () => {
    setLoading(true);
    fetch('/api/transactions')
      .then((res) => res.json())
      .then((data) => {
        setTransactions(data.transactions);
        setCategories(data.categories);
        if (data.categories.length > 0) setCategoryId(data.categories[0].id);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load transactions:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  const handleAddSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/transactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description, amount, type, category_id: categoryId, date })
      });
      if (response.ok) {
        setIsAddOpen(false);
        // Clear fields
        setDescription('');
        setAmount('');
        fetchTransactions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this transaction?')) return;
    try {
      const response = await fetch(`/api/transactions/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchTransactions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Receipt scanning handler
  const handleScanSubmit = async (e) => {
    e.preventDefault();
    if (!scanFile) return;

    setScanning(true);
    setScanResult(null);
    const formData = new FormData();
    formData.append('receipt_file', scanFile);

    try {
      const response = await fetch('/api/process-receipt-fast', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (response.ok && !data.error) {
        setScanResult(data);
        // Prefill add form fields
        setDescription(data.description || '');
        setAmount(data.amount || '');
        setType(data.type || 'expense');
        if (data.category_id) setCategoryId(data.category_id);
        if (data.date) setDate(data.date);
      } else {
        alert(data.error || 'Scan failed.');
      }
    } catch (err) {
      console.error(err);
      alert('Scanning failed.');
    } finally {
      setScanning(false);
    }
  };

  const filteredTransactions = transactions.filter((tx) => {
    const matchesSearch = tx.description.toLowerCase().includes(search.toLowerCase());
    const matchesType = filterType === 'all' || tx.type === filterType;
    return matchesSearch && matchesType;
  });

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white "
    >
      {/* Header and Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight font-sans">
            Ledger Entries
          </h1>
          <p className="text-white/40 text-sm mt-1 font-sans">
            Review, add, or scan receipt items into your secure financial command.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsScanOpen(true)}
            className="flex items-center gap-2 bg-gradient-to-r from-cyan-500/20 to-teal-500/20 border border-cyan-500/30 text-cyan-300 font-semibold px-4 py-2.5 rounded-xl hover:bg-cyan-500/30 active:scale-[0.98] transition-all text-sm font-sans"
          >
            <Camera size={16} />
            <span>AI Scan Receipt</span>
          </button>
          <button
            onClick={() => setIsAddOpen(true)}
            className="flex items-center gap-2 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-semibold px-4 py-2.5 rounded-xl hover:brightness-110 active:scale-[0.98] transition-all text-sm glow-purple font-sans"
          >
            <Plus size={16} />
            <span>Add Entry</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col md:flex-row gap-4 items-center bg-white/5 border border-white/5 p-4 rounded-2xl w-full">
        <div className="relative w-full md:flex-1">
          <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-white/30 pointer-events-none">
            <Search size={16} />
          </span>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-white/5 border border-white/5 rounded-xl py-2 pl-10 pr-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
            placeholder="Search entries..."
          />
        </div>

        <div className="flex gap-2 w-full md:w-auto">
          {['all', 'income', 'expense'].map((t) => (
            <button
              key={t}
              onClick={() => setFilterType(t)}
              className={`flex-1 md:flex-initial text-xs uppercase tracking-wider font-semibold font-sans px-4 py-2.5 rounded-xl border transition-all ${
                filterType === t 
                  ? 'bg-brand-purple/20 border-brand-purple text-white' 
                  : 'bg-white/5 border-white/5 text-white/40 hover:text-white'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Transactions Table/Cards */}
      <div className="glass-panel rounded-2xl border border-white/5 shadow-2xl overflow-hidden">
        {loading ? (
          <div className="py-20 flex items-center justify-center">
            <div className="w-10 h-10 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
          </div>
        ) : filteredTransactions.length === 0 ? (
          <div className="py-20 text-center text-white/30 text-sm font-sans">
            No entries found. Try adding a new transaction or clearing filters.
          </div>
        ) : (
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-white/5 bg-white/5 text-xs font-semibold uppercase tracking-wider text-white/40 font-sans">
                  <th className="py-4 px-6">Description</th>
                  <th className="py-4 px-6 text-center">Category</th>
                  <th className="py-4 px-6 text-center">Date</th>
                  <th className="py-4 px-6 text-right">Amount</th>
                  <th className="py-4 px-6 text-center">Action</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filteredTransactions.map((tx) => (
                    <motion.tr
                      key={tx.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="border-b border-white/5 hover:bg-white/5 transition-all text-sm group"
                    >
                      <td className="py-4 px-6 font-semibold text-white group-hover:text-brand-purple transition-all font-sans">
                        {tx.description}
                      </td>
                      <td className="py-4 px-6 text-center">
                        <span className={`px-2.5 py-1 rounded-full text-xs font-semibold font-sans bg-white/5 text-white/60`}>
                          {tx.category ? tx.category.name : 'General'}
                        </span>
                      </td>
                      <td className="py-4 px-6 text-center text-white/50 font-sans">
                        {tx.date}
                      </td>
                      <td className={`py-4 px-6 text-right font-bold font-sans ${
                        tx.type === 'income' ? 'text-emerald-400' : 'text-brand-pink'
                      }`}>
                        {tx.type === 'income' ? '+' : '-'} ₹{tx.amount.toLocaleString()}
                      </td>
                      <td className="py-4 px-6 text-center">
                        <button
                          onClick={() => handleDelete(tx.id)}
                          className="p-2 text-white/30 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg border border-transparent hover:border-rose-500/20 transition-all"
                        >
                          <Trash2 size={15} />
                        </button>
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* --- ADD DIALOG MODAL --- */}
      <AnimatePresence>
        {isAddOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-lg glass-panel p-6 rounded-3xl border border-white/10"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white font-sans">Add Ledger Entry</h3>
                <button onClick={() => setIsAddOpen(false)} className="text-white/40 hover:text-white">
                  <X size={18} />
                </button>
              </div>

              <form onSubmit={handleAddSubmit} className="flex flex-col gap-4">
                <div>
                  <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                    Description
                  </label>
                  <input
                    type="text"
                    required
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    placeholder="E.g., Grocery Shopping"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Amount (INR)
                    </label>
                    <input
                      type="number"
                      required
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                      placeholder="0.00"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Date
                    </label>
                    <input
                      type="date"
                      required
                      value={date}
                      onChange={(e) => setDate(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Transaction Type
                    </label>
                    <select
                      value={type}
                      onChange={(e) => setType(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    >
                      <option value="expense">Expense</option>
                      <option value="income">Income</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Category
                    </label>
                    <select
                      value={categoryId}
                      onChange={(e) => setCategoryId(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    >
                      {categories.map((c) => (
                        <option key={c.id} value={c.id}>{c.name}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full mt-4 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-bold py-3 rounded-xl hover:brightness-110 active:scale-[0.99] transition-all glow-purple font-sans"
                >
                  Confirm Entry
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>

      {/* --- AI RECEIPT SCANNER MODAL --- */}
      <AnimatePresence>
        {isScanOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-lg glass-panel p-6 rounded-3xl border border-white/10 overflow-hidden relative"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white flex items-center gap-2 font-sans">
                  <Sparkles size={18} className="text-brand-cyan animate-pulse" />
                  <span>AI Receipt Scan</span>
                </h3>
                <button onClick={() => setIsScanOpen(false)} className="text-white/40 hover:text-white">
                  <X size={18} />
                </button>
              </div>

              <form onSubmit={handleScanSubmit} className="flex flex-col gap-4">
                <div className="border border-dashed border-white/10 hover:border-brand-purple/40 transition-all rounded-2xl p-8 flex flex-col items-center justify-center gap-3 bg-white/5 relative">
                  {/* Scanner laser animation */}
                  {scanning && (
                    <motion.div 
                      initial={{ top: '0%' }}
                      animate={{ top: '100%' }}
                      transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }}
                      className="absolute left-0 right-0 h-1 bg-cyan-400 blur-[2px] opacity-70"
                    />
                  )}

                  <Camera size={32} className="text-white/30" />
                  <span className="text-sm font-semibold font-sans text-white/70">Upload receipt image</span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setScanFile(e.target.files[0])}
                    required
                    className="text-xs text-white/50 border border-white/5 rounded-lg p-2 bg-white/5"
                  />
                </div>

                <button
                  type="submit"
                  disabled={scanning}
                  className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 text-white font-bold py-3 rounded-xl hover:brightness-110 active:scale-[0.99] transition-all flex items-center justify-center gap-2 font-sans"
                >
                  {scanning ? (
                    <>
                      <Loader size={16} className="animate-spin" />
                      <span>Reading Receipt...</span>
                    </>
                  ) : (
                    <span>Process Receipt</span>
                  )}
                </button>
              </form>

              {/* Show Extracted Scan Result */}
              {scanResult && (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-6 p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 flex flex-col gap-2"
                >
                  <h4 className="text-sm font-bold text-emerald-400 flex items-center gap-1.5 font-sans">
                    <Check size={16} />
                    <span>Receipt Extracted Successfully!</span>
                  </h4>
                  <div className="text-xs text-white/70 flex flex-col gap-1 font-sans">
                    <p><strong>Merchant:</strong> {scanResult.description}</p>
                    <p><strong>Amount:</strong> ₹{parseFloat(scanResult.amount).toLocaleString()}</p>
                    <p><strong>Date:</strong> {scanResult.date}</p>
                    <p><strong>Category Suggestion:</strong> {scanResult.category_name || 'General'}</p>
                  </div>
                  <button
                    onClick={() => {
                      setIsScanOpen(false);
                      setIsAddOpen(true);
                    }}
                    className="mt-2 w-full border border-emerald-500/30 text-emerald-300 text-xs font-semibold py-2 rounded-lg hover:bg-emerald-500/15 transition-all font-sans"
                  >
                    Load into Form
                  </button>
                </motion.div>
              )}
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
export default Transactions;

