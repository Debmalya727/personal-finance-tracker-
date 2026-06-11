import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { UserCircle, Save, CheckCircle, Calendar, Shield } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Profile = () => {
  const { user } = useAuth();
  const [form, setForm] = useState({ dob: '', role: 'personal' });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch('/api/profile')
      .then(r => r.json())
      .then(d => {
        setForm({ dob: d.dob || '', role: d.role || 'personal' });
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await fetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
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

  const age = form.dob
    ? Math.floor((new Date() - new Date(form.dob)) / (365.25 * 24 * 60 * 60 * 1000))
    : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-2xl flex flex-col gap-8 text-white pb-16"
    >
      <div>
        <div className="flex items-center gap-2 mb-1">
          <UserCircle size={16} className="text-brand-cyan" />
          <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">Account</span>
        </div>
        <h1 className="text-4xl font-black tracking-tight font-sans">Profile</h1>
        <p className="text-white/40 text-sm mt-1 font-sans">Manage your account details and preferences</p>
      </div>

      {/* Avatar card */}
      <div className="glass-panel rounded-2xl p-6 border border-white/5 flex items-center gap-5">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center text-3xl font-black text-white"
          style={{ background: 'linear-gradient(135deg, #7c3aed, #db2777)', boxShadow: '0 0 30px rgba(124,58,237,0.4)' }}>
          {user?.username?.[0]?.toUpperCase()}
        </div>
        <div>
          <h2 className="text-2xl font-black text-white font-sans">{user?.username}</h2>
          <div className="flex items-center gap-2 mt-1">
            <Shield size={12} className="text-brand-purple" />
            <span className="text-xs text-white/50 capitalize font-medium font-sans">{user?.role} Account</span>
            {age && <span className="text-xs text-white/30 font-sans">• Age {age}</span>}
          </div>
        </div>
      </div>

      {/* Edit Form */}
      <div className="glass-panel rounded-2xl p-6 border border-white/5">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white/50 mb-5 font-sans">Update Details</h3>
        {loading ? (
          <div className="flex flex-col gap-4">
            <div className="h-16 rounded-xl bg-white/5 animate-pulse" />
            <div className="h-16 rounded-xl bg-white/5 animate-pulse" />
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <div>
              <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">Date of Birth</label>
              <div className="relative">
                <Calendar size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
                <input type="date" value={form.dob} onChange={e => setForm({ ...form, dob: e.target.value })}
                  className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-10 pr-4 text-white text-sm focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all font-sans [color-scheme:dark]" />
              </div>
            </div>

            <div>
              <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">Account Role</label>
              <div className="grid grid-cols-2 gap-3">
                {['personal', 'business'].map(r => (
                  <button key={r} type="button" onClick={() => setForm({ ...form, role: r })}
                    className={`p-4 rounded-xl border-2 text-left transition-all ${form.role === r ? 'border-brand-cyan bg-brand-cyan/10' : 'border-white/10 hover:border-white/20 bg-white/3'}`}>
                    <span className={`text-sm font-bold capitalize font-sans ${form.role === r ? 'text-white' : 'text-white/50'}`}>{r}</span>
                  </button>
                ))}
              </div>
            </div>

            <motion.button type="submit" disabled={saving} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
              className="mt-2 w-full flex items-center justify-center gap-2 py-3.5 rounded-xl font-bold text-white text-sm font-sans transition-all"
              style={{ background: saved ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #06b6d4, #7c3aed)', boxShadow: '0 0 20px rgba(6,182,212,0.3)' }}>
              {saved ? <><CheckCircle size={16} />Profile Saved!</> : saving ? 'Saving...' : <><Save size={16} />Save Changes</>}
            </motion.button>
          </form>
        )}
      </div>
    </motion.div>
  );
};

export default Profile;
