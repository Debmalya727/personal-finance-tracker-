import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Wallet, ShieldAlert, Award, ArrowUpRight, IndianRupee } from 'lucide-react';
import { ThreeCanvas } from '../components/ThreeCanvas';
import { ThreeDChart } from '../components/ThreeDChart';

export const NetWorth = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [webglSupported, setWebglSupported] = useState(true);

  useEffect(() => {
    // Check WebGL support
    try {
      const canvas = document.createElement('canvas');
      const supported = !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
      setWebglSupported(supported);
    } catch (e) {
      setWebglSupported(false);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000); // 20s max

    fetch('/api/net_worth', { signal: controller.signal })
      .then((res) => res.json())
      .then((resData) => {
        clearTimeout(timeoutId);
        setData(resData);
        setLoading(false);
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        if (err.name !== 'AbortError') console.error('Failed to load net worth:', err);
        setLoading(false);
      });

    return () => { clearTimeout(timeoutId); controller.abort(); };
  }, []);

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-12 h-12 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
      </div>
    );
  }

  const nwData = data || {
    net_worth: 0,
    assets: { cash: 0, schemes: 0, stocks: 0, crypto: 0, total: 0 },
    liabilities: { loans: 0, total: 0 },
    details: { schemes: [], investments: [] }
  };

  // Setup data for 3D allocation blocks
  const chartData = [
    { name: 'Cash', value: nwData.assets?.cash || 0, color: '#06b6d4' },
    { name: 'Schemes', value: nwData.assets?.schemes || 0, color: '#7c3aed' },
    { name: 'Stocks', value: nwData.assets?.stocks || 0, color: '#db2777' },
    { name: 'Crypto', value: nwData.assets?.crypto || 0, color: '#f59e0b' }
  ].filter(d => d.value > 0); // Only render assets the user actually owns

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col lg:grid lg:grid-cols-5 gap-8 text-white relative z-10"
    >
      {/* Left Details Panel (3 columns) */}
      <div className="lg:col-span-3 flex flex-col gap-6">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight font-sans">
            Net Worth Console
          </h1>
          <p className="text-white/40 text-sm mt-1 font-sans">
            Aggregated look at balances, investments, and liabilities.
          </p>
        </div>

        {/* Net Worth Glass Banner */}
        <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl flex items-center justify-between bg-gradient-to-r from-brand-purple/15 to-brand-pink/5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-brand-purple/20 border border-brand-purple/30 flex items-center justify-center glow-purple">
              <Award size={22} className="text-brand-purple animate-bounce" />
            </div>
            <div>
              <span className="text-xs text-white/50 font-bold uppercase tracking-wider font-sans block">Aggregate Net Worth</span>
              <h2 className="text-3xl font-extrabold tracking-tight text-white font-sans mt-0.5">
                ₹{nwData.net_worth.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
              </h2>
            </div>
          </div>
        </div>

        {/* Assets & Liabilities Summary Card */}
        <div className="grid grid-cols-2 gap-4">
          <div className="glass-panel p-4 rounded-xl border border-white/5 bg-white/2">
            <span className="text-[10px] text-white/40 font-bold uppercase tracking-wider font-sans block">Total Assets</span>
            <span className="text-xl font-extrabold text-emerald-400 font-sans mt-1 block">
              ₹{(nwData.assets?.total || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
          </div>
          <div className="glass-panel p-4 rounded-xl border border-white/5 bg-white/2">
            <span className="text-[10px] text-white/40 font-bold uppercase tracking-wider font-sans block">Total Liabilities</span>
            <span className="text-xl font-extrabold text-brand-pink font-sans mt-1 block">
              ₹{(nwData.liabilities?.total || 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
          </div>
        </div>

        {/* Asset Category list */}
        <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col gap-3.5 shadow-xl bg-white/2">
          <h3 className="text-sm font-bold uppercase tracking-wider text-white/50 font-sans">Asset Allocation</h3>
          
          <div className="flex flex-col gap-2">
            {[
              { name: 'Cash Reserves', value: nwData.assets?.cash || 0, color: 'bg-brand-cyan/20 border-brand-cyan/40 text-brand-cyan' },
              { name: 'Fixed Income Schemes', value: nwData.assets?.schemes || 0, color: 'bg-brand-purple/20 border-brand-purple/40 text-brand-purple' },
              { name: 'Stock Equities', value: nwData.assets?.stocks || 0, color: 'bg-brand-pink/20 border-brand-pink/40 text-brand-pink' },
              { name: 'Cryptocurrency', value: nwData.assets?.crypto || 0, color: 'bg-yellow-500/20 border-yellow-500/40 text-yellow-500' }
            ].map((asset) => (
              <div key={asset.name} className="flex justify-between items-center p-3 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all font-sans">
                <span className="text-sm text-white/70 font-semibold">{asset.name}</span>
                <span className="font-bold text-sm">₹{asset.value.toLocaleString('en-IN', { maximumFractionDigits: 2 })}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right 3D Visualizer Panel (2 columns) */}
      <div className="lg:col-span-2 glass-panel rounded-3xl border border-white/5 shadow-2xl relative min-h-[450px] overflow-hidden flex flex-col justify-between p-6 bg-white/2">
        <div className="relative z-10">
          <h3 className="text-lg font-bold font-sans">3D Space Allocation</h3>
          <p className="text-white/40 text-xs mt-1 font-sans">
            Interactive block heights represent absolute asset distributions. Hover blocks to focus.
          </p>
        </div>

        {/* R3F Canvas Container */}
        <div className="absolute inset-0 top-20 bottom-0 z-0 flex items-center justify-center">
          {chartData.length > 0 && webglSupported ? (
            <ThreeCanvas className="z-0" cameraPosition={[0, 0, 6]}>
              <ThreeDChart data={chartData} />
            </ThreeCanvas>
          ) : chartData.length > 0 ? (
            // Beautiful CSS fallback
            <div className="w-full h-full flex flex-col justify-center gap-4 px-6 relative z-10 bg-[#08071a]/40">
              <span className="text-white/30 text-[10px] font-bold uppercase tracking-widest block text-center mb-4">
                2D Allocation Distribution
              </span>
              {chartData.map((bar) => {
                const total = nwData.assets?.total || 1;
                const percentage = ((bar.value / total) * 100).toFixed(1);
                return (
                  <div key={bar.name} className="flex flex-col gap-1.5 font-sans">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-white/70">{bar.name}</span>
                      <span style={{ color: bar.color }}>₹{bar.value.toLocaleString('en-IN', { maximumFractionDigits: 0 })} ({percentage}%)</span>
                    </div>
                    <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden border border-white/5">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: bar.color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-white/30 text-xs font-sans relative z-10">
              Enter assets to render allocation blocks.
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default NetWorth;
