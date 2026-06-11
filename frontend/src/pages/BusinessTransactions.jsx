import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, Search, Trash2, FileText, X, Check, Loader, 
  ArrowUpRight, ArrowDownRight, Edit2 
} from 'lucide-react';

export const BusinessTransactions = () => {
  const [transactions, setTransactions] = useState([]);
  const [clients, setClients] = useState([]);
  const [categories, setCategories] = useState([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  
  const [isAddOpen, setIsAddOpen] = useState(false);

  // Form State
  const [description, setDescription] = useState('');
  const [amount, setAmount] = useState('');
  const [type, setType] = useState('revenue');
  const [categoryId, setCategoryId] = useState('');
  const [clientId, setClientId] = useState('');
  const [invoiceStatus, setInvoiceStatus] = useState('unpaid');
  const [date, setDate] = useState(new Date().toISOString().split('T')[0]);
  const [dueDate, setDueDate] = useState('');

  const fetchTransactions = () => {
    setLoading(true);
    fetch('/api/business/transactions')
      .then((res) => res.json())
      .then((data) => {
        setTransactions(data.transactions);
        setClients(data.clients || []);
        setCategories(data.categories || []);
        if (data.categories.length > 0) setCategoryId(data.categories[0].id);
        if (data.clients.length > 0) setClientId(data.clients[0].id);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchTransactions();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/business/transactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description,
          amount: parseFloat(amount),
          type,
          category_id: categoryId,
          client_id: clientId ? parseInt(clientId) : null,
          invoice_status: invoiceStatus,
          date,
          due_date: dueDate || null
        })
      });
      if (response.ok) {
        setIsAddOpen(false);
        setDescription('');
        setAmount('');
        fetchTransactions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this business invoice?')) return;
    try {
      const response = await fetch(`/api/business/transactions/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchTransactions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleToggleStatus = async (tx) => {
    const newStatus = tx.invoice_status === 'paid' ? 'unpaid' : 'paid';
    try {
      const response = await fetch(`/api/business/transactions/${tx.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ invoice_status: newStatus })
      });
      if (response.ok) {
        fetchTransactions();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const filteredTransactions = transactions.filter((tx) => {
    return tx.description.toLowerCase().includes(search.toLowerCase());
  });

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white "
    >
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight font-sans">
            Business Ledger & Invoices
          </h1>
          <p className="text-white/40 text-sm mt-1 font-sans">
            Track customer invoicing, balance accounts, and manage operating revenues.
          </p>
        </div>

        <button
          onClick={() => setIsAddOpen(true)}
          className="flex items-center gap-2 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-semibold px-4 py-2.5 rounded-xl hover:brightness-110 active:scale-[0.98] transition-all text-sm glow-purple font-sans"
        >
          <Plus size={16} />
          <span>New Invoice / Tx</span>
        </button>
      </div>

      {/* Search Filter */}
      <div className="relative w-full">
        <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center text-white/30 pointer-events-none">
          <Search size={16} />
        </span>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-white/5 border border-white/5 rounded-xl py-2 pl-10 pr-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
          placeholder="Search invoices..."
        />
      </div>

      {/* Invoices list table */}
      <div className="glass-panel rounded-2xl border border-white/5 shadow-2xl overflow-hidden">
        {loading ? (
          <div className="py-20 flex justify-center">
            <div className="w-10 h-10 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
          </div>
        ) : filteredTransactions.length === 0 ? (
          <div className="py-20 text-center text-white/30 text-sm font-sans">
            No business invoices or transactions recorded yet.
          </div>
        ) : (
          <div className="overflow-x-auto w-full">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-white/5 bg-white/5 text-xs font-semibold uppercase tracking-wider text-white/40 font-sans">
                  <th className="py-4 px-6">Description</th>
                  <th className="py-4 px-6 text-center">Client</th>
                  <th className="py-4 px-6 text-center">Due Date</th>
                  <th className="py-4 px-6 text-center">Status</th>
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
                      <td className="py-4 px-6 text-center text-white/70 font-sans">
                        {tx.client ? tx.client.name : 'Direct Sale'}
                      </td>
                      <td className="py-4 px-6 text-center text-white/50 font-sans">
                        {tx.due_date || 'None'}
                      </td>
                      <td className="py-4 px-6 text-center">
                        <button
                          onClick={() => handleToggleStatus(tx)}
                          className={`px-2.5 py-1 rounded-full text-xs font-bold font-sans transition-all hover:brightness-110 active:scale-[0.98] ${
                            tx.invoice_status === 'paid' 
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                              : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
                          }`}
                        >
                          {tx.invoice_status}
                        </button>
                      </td>
                      <td className={`py-4 px-6 text-right font-bold font-sans ${
                        tx.type === 'revenue' ? 'text-emerald-400' : 'text-brand-pink'
                      }`}>
                        {tx.type === 'revenue' ? '+' : '-'} ₹{tx.amount.toLocaleString()}
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

      {/* --- ADD MODAL --- */}
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
                <h3 className="text-xl font-bold text-white font-sans">New Invoice Details</h3>
                <button onClick={() => setIsAddOpen(false)} className="text-white/40 hover:text-white">
                  <X size={18} />
                </button>
              </div>

              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
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
                    placeholder="E.g., Consulting Fees Q3"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Invoice Amount (INR)
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
                      Type
                    </label>
                    <select
                      value={type}
                      onChange={(e) => setType(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    >
                      <option value="revenue">Revenue (Inflow)</option>
                      <option value="expense">Expense (Outflow)</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Client
                    </label>
                    <select
                      value={clientId}
                      onChange={(e) => setClientId(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    >
                      <option value="">None (Direct)</option>
                      {clients.map((c) => (
                        <option key={c.id} value={c.id}>{c.name}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Invoice Status
                    </label>
                    <select
                      value={invoiceStatus}
                      onChange={(e) => setInvoiceStatus(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    >
                      <option value="unpaid">Unpaid (Outstanding)</option>
                      <option value="paid">Paid</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Issue Date
                    </label>
                    <input
                      type="date"
                      required
                      value={date}
                      onChange={(e) => setDate(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                      Due Date (Optional)
                    </label>
                    <input
                      type="date"
                      value={dueDate}
                      onChange={(e) => setDueDate(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full mt-4 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-bold py-3 rounded-xl hover:brightness-110 active:scale-[0.99] transition-all glow-purple font-sans"
                >
                  Create Invoice
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
export default BusinessTransactions;

