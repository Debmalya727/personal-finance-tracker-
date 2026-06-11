import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Lock, LogIn, AlertCircle, Eye, EyeOff, User } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { ThreeBackground } from '../components/ThreeBackground';

export const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const result = await login(username, password);
    setLoading(false);
    if (result.success) {
      navigate('/dashboard');
    } else {
      setError(result.message);
    }
  };

  return (
    <div className="relative w-screen h-screen flex items-center justify-center overflow-hidden bg-[#030014]">
      {/* Full-Screen 3D Background */}
      <ThreeCanvas cameraPosition={[0, 0, 7]}>
        <ThreeBackground />
      </ThreeCanvas>

      {/* Ambient Glow Orbs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-purple-700/20 blur-[120px] animate-pulse" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-cyan-500/15 blur-[100px] animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-[30%] right-[20%] w-[300px] h-[300px] rounded-full bg-pink-600/15 blur-[80px] animate-pulse" style={{ animationDelay: '4s' }} />
      </div>

      {/* Grid overlay */}
      <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

      {/* Center Login Card */}
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md px-4"
      >
        {/* Glowing border container */}
        <div className="relative p-[1px] rounded-3xl overflow-hidden">
          <div
            className="absolute inset-0 rounded-3xl"
            style={{
              background: 'linear-gradient(135deg, rgba(124,58,237,0.6) 0%, rgba(6,182,212,0.3) 50%, rgba(219,39,119,0.4) 100%)',
            }}
          />
          <div className="relative bg-[#08071a]/90 backdrop-blur-2xl rounded-3xl p-8">
            {/* Brand Header */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.7 }}
              className="text-center mb-8"
            >
              <div className="relative inline-flex mb-5">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-tr from-brand-purple via-purple-500 to-brand-pink flex items-center justify-center shadow-2xl shadow-purple-500/40">
                  <svg viewBox="0 0 32 32" className="w-10 h-10 text-white fill-white">
                    <path d="M16 2 L28 8 L28 16 C28 23 22 29 16 30 C10 29 4 23 4 16 L4 8 Z" opacity="0.3" />
                    <path d="M10 16 L14 20 L22 12" stroke="white" strokeWidth="2.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                    <circle cx="16" cy="16" r="10" stroke="white" strokeWidth="1.5" fill="none" opacity="0.5" />
                  </svg>
                </div>
                {/* Pulse ring */}
                <div className="absolute inset-0 rounded-2xl border-2 border-purple-500/40 animate-ping" style={{ animationDuration: '2s' }} />
              </div>
              <h1 className="text-4xl font-black tracking-tight text-white font-sans">FinVision</h1>
              <p className="text-sm text-white/40 mt-1 tracking-widest uppercase font-sans">3D Wealth Intelligence</p>
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
              {/* Username Field */}
              <div className="group">
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-2 font-sans">
                  Username
                </label>
                <div className="relative">
                  <User
                    size={16}
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30 group-focus-within:text-brand-purple transition-colors"
                  />
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    autoComplete="username"
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-11 pr-4 text-white text-sm placeholder-white/20 focus:outline-none focus:border-brand-purple focus:bg-white/8 focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans"
                    placeholder="Enter your username"
                  />
                </div>
              </div>

              {/* Password Field */}
              <div className="group">
                <label className="block text-xs font-bold uppercase tracking-widest text-white/50 mb-2 font-sans">
                  Password
                </label>
                <div className="relative">
                  <Lock
                    size={16}
                    className="absolute left-4 top-1/2 -translate-y-1/2 text-white/30 group-focus-within:text-brand-purple transition-colors"
                  />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoComplete="current-password"
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3.5 pl-11 pr-12 text-white text-sm placeholder-white/20 focus:outline-none focus:border-brand-purple focus:bg-white/8 focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans"
                    placeholder="••••••••••••"
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

              {/* Submit Button */}
              <motion.button
                type="submit"
                disabled={loading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="relative mt-2 w-full py-4 rounded-xl font-bold text-white text-sm overflow-hidden transition-all font-sans"
                style={{
                  background: 'linear-gradient(135deg, #7c3aed 0%, #db2777 100%)',
                  boxShadow: '0 0 30px rgba(124, 58, 237, 0.4)',
                }}
              >
                <span className="relative flex items-center justify-center gap-2">
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Authenticating...
                    </>
                  ) : (
                    <>
                      <LogIn size={16} />
                      Enter Portal
                    </>
                  )}
                </span>
              </motion.button>
            </form>

            {/* Footer */}
            <p className="text-center text-sm text-white/40 mt-7 font-sans">
              New to FinVision?{' '}
              <Link
                to="/register"
                className="text-brand-purple hover:text-purple-300 font-bold transition-colors"
              >
                Create Account
              </Link>
            </p>
          </div>
        </div>

        {/* Bottom tagline */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="text-center text-xs text-white/20 mt-6 tracking-wider uppercase font-sans"
        >
          Secured • Encrypted • Real-Time
        </motion.p>
      </motion.div>
    </div>
  );
};

export default Login;

