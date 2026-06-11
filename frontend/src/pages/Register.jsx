import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Lock, User, AlertCircle, Eye, EyeOff, Calendar, Briefcase, UserCheck, Sparkles } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { ThreeBackground } from '../components/ThreeBackground';

export const Register = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [dob, setDob] = useState('');
  const [role, setRole] = useState('personal');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const result = await register(username, password, dob, role);
    setLoading(false);
    if (result.success) {
      navigate('/dashboard');
    } else {
      setError(result.message);
    }
  };

  return (
    <div className="relative w-screen min-h-screen flex items-center justify-center overflow-hidden bg-[#030014] py-10">
      {/* 3D Background */}
      <ThreeCanvas cameraPosition={[0, 0, 7]}>
        <ThreeBackground />
      </ThreeCanvas>

      {/* Ambient glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] right-[-10%] w-[600px] h-[600px] rounded-full bg-cyan-700/15 blur-[120px]" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[500px] h-[500px] rounded-full bg-purple-600/15 blur-[100px]" />
      </div>
      <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none" />

      {/* Card */}
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md px-4"
      >
        <div className="relative p-[1px] rounded-3xl overflow-hidden">
          <div
            className="absolute inset-0 rounded-3xl"
            style={{
              background: 'linear-gradient(135deg, rgba(6,182,212,0.5) 0%, rgba(124,58,237,0.4) 50%, rgba(219,39,119,0.3) 100%)',
            }}
          />
          <div className="relative bg-[#08071a]/90 backdrop-blur-2xl rounded-3xl p-8">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-center mb-8"
            >
              <div className="relative inline-flex mb-5">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-brand-cyan via-blue-500 to-brand-purple flex items-center justify-center shadow-2xl shadow-cyan-500/30">
                  <Sparkles size={36} className="text-white" />
                </div>
                <div className="absolute inset-0 rounded-2xl border-2 border-cyan-400/30 animate-ping" style={{ animationDuration: '2s' }} />
              </div>
              <h1 className="text-4xl font-black tracking-tight text-white font-sans">Join FinVision</h1>
              <p className="text-sm text-white/40 mt-1 tracking-widest uppercase font-sans">Your Wealth Journey Begins</p>
            </motion.div>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0, marginBottom: 0 }}
                  animate={{ opacity: 1, height: 'auto', marginBottom: 20 }}
                  exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                  className="flex items-center gap-3 p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm"
                >
                  <AlertCircle size={16} className="shrink-0" />
                  <span className="font-medium">{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <form onSubmit={handleSubmit} className="flex flex-col gap-5">
              {/* Username */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-2 font-sans">
                  Username
                </label>
                <div className="relative">
                  <User size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30" />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-11 pr-4 text-white text-sm placeholder-white/20 focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all font-sans"
                    placeholder="Choose a username"
                  />
                </div>
              </div>

              {/* Password */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-2 font-sans">
                  Password
                </label>
                <div className="relative">
                  <Lock size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-11 pr-12 text-white text-sm placeholder-white/20 focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all font-sans"
                    placeholder="Create a strong password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-white/30 hover:text-white/60 transition-colors"
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              {/* DOB */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-2 font-sans">
                  Date of Birth
                </label>
                <div className="relative">
                  <Calendar size={16} className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30" />
                  <input
                    type="date"
                    value={dob}
                    onChange={(e) => setDob(e.target.value)}
                    required
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-11 pr-4 text-white text-sm focus:outline-none focus:border-brand-cyan focus:ring-2 focus:ring-brand-cyan/20 transition-all font-sans [color-scheme:dark]"
                  />
                </div>
              </div>

              {/* Account Type */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-3 font-sans">
                  Account Type
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { value: 'personal', label: 'Personal', Icon: UserCheck, desc: 'Individual wealth tracking' },
                    { value: 'business', label: 'Business', Icon: Briefcase, desc: 'Corporate financials' },
                  ].map(({ value, label, Icon, desc }) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setRole(value)}
                      className={`p-4 rounded-xl border-2 text-left transition-all ${
                        role === value
                          ? 'border-brand-cyan bg-brand-cyan/10 shadow-lg shadow-cyan-500/20'
                          : 'border-white/10 bg-white/3 hover:border-white/20'
                      }`}
                    >
                      <Icon size={18} className={role === value ? 'text-brand-cyan mb-1.5' : 'text-white/40 mb-1.5'} />
                      <p className={`text-sm font-bold font-sans ${role === value ? 'text-white' : 'text-white/60'}`}>{label}</p>
                      <p className="text-xs text-white/30 font-sans mt-0.5">{desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Submit */}
              <motion.button
                type="submit"
                disabled={loading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="relative mt-2 w-full py-4 rounded-xl font-bold text-white text-sm overflow-hidden font-sans"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4 0%, #7c3aed 100%)',
                  boxShadow: '0 0 30px rgba(6, 182, 212, 0.35)',
                }}
              >
                <span className="relative flex items-center justify-center gap-2">
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Creating Account...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Launch My Portal
                    </>
                  )}
                </span>
              </motion.button>
            </form>

            <p className="text-center text-sm text-white/40 mt-7 font-sans">
              Already a member?{' '}
              <Link to="/login" className="text-brand-cyan hover:text-cyan-300 font-bold transition-colors">
                Sign In
              </Link>
            </p>
          </div>
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-center text-xs text-white/20 mt-6 tracking-wider uppercase font-sans"
        >
          256-bit Encrypted • Zero Data Sharing
        </motion.p>
      </motion.div>
    </div>
  );
};

export default Register;

