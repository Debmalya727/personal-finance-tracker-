import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Plus, Users, Mail, Phone, Briefcase, Trash2 } from 'lucide-react';

export const Clients = () => {
  const [clients, setClients] = useState([]);
  const [loading, setLoading] = useState(true);

  // Form State
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [company, setCompany] = useState('');

  const fetchClients = () => {
    setLoading(true);
    fetch('/api/business/clients')
      .then((res) => res.json())
      .then((data) => {
        setClients(data.clients);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchClients();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/business/clients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, phone, company })
      });
      if (response.ok) {
        setName('');
        setEmail('');
        setPhone('');
        setCompany('');
        fetchClients();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this client account?')) return;
    try {
      const response = await fetch(`/api/business/clients/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchClients();
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
          Client Directory
        </h1>
        <p className="text-white/40 text-sm mt-1 font-sans">
          Register new clients, edit billing profiles, and manage outstanding invoices.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Add Client Form */}
        <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl flex flex-col gap-4">
          <h3 className="text-base font-bold font-sans flex items-center gap-2">
            <Plus size={16} className="text-brand-purple" />
            <span>Add Client Account</span>
          </h3>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4 font-sans text-sm">
            <div>
              <label className="text-xs text-white/50 mb-1.5 block font-semibold">Client Name</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. John Doe"
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
              />
            </div>
            <div>
              <label className="text-xs text-white/50 mb-1.5 block font-semibold">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="john@example.com"
                className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Phone Number</label>
                <input
                  type="text"
                  required
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="+91..."
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
              <div>
                <label className="text-xs text-white/50 mb-1.5 block font-semibold">Company / Org</label>
                <input
                  type="text"
                  required
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="Acme Corp"
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-3 text-white focus:outline-none focus:border-brand-purple transition-all"
                />
              </div>
            </div>
            <button
              type="submit"
              className="w-full mt-2 bg-gradient-to-r from-brand-purple to-brand-pink text-white font-bold py-2.5 rounded-xl hover:brightness-110 active:scale-[0.98] transition-all glow-purple"
            >
              Register Client
            </button>
          </form>
        </div>

        {/* Clients list (2 columns) */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl flex flex-col gap-4">
          <h3 className="text-base font-bold font-sans flex items-center gap-2">
            <Users size={16} className="text-brand-cyan" />
            <span>Accounts Database</span>
          </h3>

          {loading ? (
            <div className="py-12 flex justify-center">
              <div className="w-8 h-8 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
            </div>
          ) : clients.length === 0 ? (
            <div className="py-12 text-center text-white/30 text-sm font-sans">
              No clients recorded in directory yet.
            </div>
          ) : (
            <div className="flex flex-col gap-3.5">
              {clients.map((c) => (
                <div 
                  key={c.id} 
                  className="flex flex-col md:flex-row md:items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5 hover:border-brand-purple/20 transition-all font-sans group gap-4"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/25 flex items-center justify-center text-cyan-400">
                      <Users size={18} />
                    </div>
                    <div>
                      <h4 className="font-semibold text-sm text-white group-hover:text-brand-purple transition-all">{c.name}</h4>
                      <p className="text-xs text-white/40 mt-0.5">{c.company || 'Individual Client'}</p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between md:justify-end gap-6 text-xs text-white/60">
                    <div className="flex items-center gap-1.5">
                      <Mail size={13} className="text-white/40" />
                      <span>{c.email}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <Phone size={13} className="text-white/40" />
                      <span>{c.phone}</span>
                    </div>
                    <button
                      onClick={() => handleDelete(c.id)}
                      className="p-1.5 text-white/30 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg border border-transparent hover:border-rose-500/20 transition-all"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </motion.div>
  );
};
export default Clients;

