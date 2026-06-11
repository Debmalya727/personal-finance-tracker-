import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Percent, ShieldAlert, Sparkles, HelpCircle } from 'lucide-react';

export const TaxEstimator = () => {
  const [grossAnnual, setGrossAnnual] = useState(1200000);
  const [deductions80c, setDeductions80c] = useState(150000);
  const [hraExemption, setHraExemption] = useState(100000);
  const [loading, setLoading] = useState(true);

  // Load defaults from database
  useEffect(() => {
    fetch('/api/salary')
      .then((res) => res.json())
      .then((data) => {
        if (data.monthly_gross > 0) {
          setGrossAnnual(data.monthly_gross * 12);
          setDeductions80c(data.deductions_80c || 150000);
          setHraExemption(data.hra_exemption || 100000);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  // Instant Javascript Tax Calculations
  const calculateNewRegime = (gross) => {
    const stdDeduction = 50000;
    const taxable = Math.max(gross - stdDeduction, 0);
    let tax = 0;

    if (taxable <= 300000) tax = 0;
    else if (taxable <= 600000) tax = (taxable - 300000) * 0.05;
    else if (taxable <= 900000) tax = 15000 + (taxable - 600000) * 0.10;
    else if (taxable <= 1200000) tax = 45000 + (taxable - 900000) * 0.15;
    else if (taxable <= 1500000) tax = 90000 + (taxable - 1200000) * 0.20;
    else tax = 150000 + (taxable - 1500000) * 0.30;

    const cess = tax * 0.04;
    return { taxable, tax, cess, total: tax + cess };
  };

  const calculateOldRegime = (gross, deductions, hra) => {
    const stdDeduction = 50000;
    const totalDeductions = deductions + hra;
    const taxable = Math.max(gross - totalDeductions - stdDeduction, 0);
    let tax = 0;

    if (taxable <= 250000) tax = 0;
    else if (taxable <= 500000) tax = (taxable - 250000) * 0.05;
    else if (taxable <= 1000000) tax = 12500 + (taxable - 500000) * 0.20;
    else tax = 112500 + (taxable - 1000000) * 0.30;

    // Rebate u/s 87A
    if (taxable <= 500000) tax = 0;

    const cess = tax * 0.04;
    return { taxable, tax, cess, total: tax + cess };
  };

  if (loading) {
    return (
      <div className="w-full h-[calc(100vh-80px)] flex items-center justify-center">
        <div className="w-12 h-12 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
      </div>
    );
  }

  const newRegime = calculateNewRegime(grossAnnual);
  const oldRegime = calculateOldRegime(grossAnnual, deductions80c, hraExemption);
  const recommended = newRegime.total <= oldRegime.total ? 'New Tax Regime' : 'Old Tax Regime';
  const taxSaved = Math.abs(newRegime.total - oldRegime.total);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white "
    >
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight font-sans">
          Tax Optimization
        </h1>
        <p className="text-white/40 text-sm mt-1 font-sans">
          Compare Old vs New Indian Income Tax Regimes instantly. Drag sliders to optimize.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        
        {/* Sliders Input Panel (2 columns) */}
        <div className="lg:col-span-2 flex flex-col gap-6 glass-panel rounded-2xl p-6 border border-white/5 shadow-xl">
          <h3 className="text-base font-bold font-sans">Salary Parameters</h3>

          {/* Gross Annual Salary Slider */}
          <div className="flex flex-col gap-2 font-sans">
            <div className="flex justify-between text-xs font-semibold text-white/60">
              <span>Gross Annual Income</span>
              <span className="text-white font-bold">₹{grossAnnual.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="300000"
              max="3000000"
              step="50000"
              value={grossAnnual}
              onChange={(e) => setGrossAnnual(Number(e.target.value))}
              className="w-full accent-brand-purple bg-white/10 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* 80C Deductions Slider */}
          <div className="flex flex-col gap-2 font-sans">
            <div className="flex justify-between text-xs font-semibold text-white/60">
              <span>Sec 80C Deductions</span>
              <span className="text-white font-bold">₹{deductions80c.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="0"
              max="150000"
              step="5000"
              value={deductions80c}
              onChange={(e) => setDeductions80c(Number(e.target.value))}
              className="w-full accent-brand-purple bg-white/10 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* HRA Slider */}
          <div className="flex flex-col gap-2 font-sans">
            <div className="flex justify-between text-xs font-semibold text-white/60">
              <span>Annual HRA / Housing Exemption</span>
              <span className="text-white font-bold">₹{hraExemption.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="0"
              max="400000"
              step="10000"
              value={hraExemption}
              onChange={(e) => setHraExemption(Number(e.target.value))}
              className="w-full accent-brand-purple bg-white/10 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>

        {/* Side-by-Side Comparison Panel (3 columns) */}
        <div className="lg:col-span-3 flex flex-col gap-6">
          
          {/* Recommendation Banner */}
          <div className="glass-panel p-5 rounded-2xl border border-emerald-500/20 bg-emerald-500/10 shadow-xl flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center">
              <Sparkles size={20} className="text-emerald-400 animate-pulse" />
            </div>
            <div>
              <p className="text-xs text-emerald-400 font-bold uppercase tracking-wider font-sans">Optimal Choice</p>
              <h3 className="text-lg font-bold text-white font-sans mt-0.5">
                We recommend the <span className="text-emerald-400">{recommended}</span>.
              </h3>
              {taxSaved > 0 && (
                <p className="text-xs text-white/60 mt-1 font-sans">
                  Saves you approximately <strong>₹{taxSaved.toLocaleString()}</strong> in annual taxes.
                </p>
              )}
            </div>
          </div>

          {/* S-B-S Split Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* New Regime Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col justify-between h-72">
              <div>
                <h4 className="font-extrabold text-brand-purple uppercase text-xs tracking-wider font-sans">New Tax Regime</h4>
                <p className="text-[10px] text-white/40 font-sans mt-0.5">Slabs updated for financial year</p>
                
                <div className="mt-6 flex flex-col gap-2.5 text-xs text-white/70 font-sans">
                  <div className="flex justify-between">
                    <span>Taxable Income:</span>
                    <span className="font-semibold text-white">₹{newRegime.taxable.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Base Tax:</span>
                    <span className="font-semibold text-white">₹{newRegime.tax.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Education Cess (4%):</span>
                    <span className="font-semibold text-white">₹{newRegime.cess.toLocaleString()}</span>
                  </div>
                </div>
              </div>

              <div className="border-t border-white/5 pt-4 flex justify-between items-baseline font-sans">
                <span className="text-[10px] text-white/40 uppercase tracking-wider font-bold">Estimated Tax</span>
                <span className="text-xl font-black text-white">₹{newRegime.total.toLocaleString()}</span>
              </div>
            </div>

            {/* Old Regime Card */}
            <div className="glass-panel p-5 rounded-2xl border border-white/5 flex flex-col justify-between h-72">
              <div>
                <h4 className="font-extrabold text-brand-pink uppercase text-xs tracking-wider font-sans">Old Tax Regime</h4>
                <p className="text-[10px] text-white/40 font-sans mt-0.5">Includes 80C and HRA deductions</p>
                
                <div className="mt-6 flex flex-col gap-2.5 text-xs text-white/70 font-sans">
                  <div className="flex justify-between">
                    <span>Taxable Income:</span>
                    <span className="font-semibold text-white">₹{oldRegime.taxable.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Base Tax:</span>
                    <span className="font-semibold text-white">₹{oldRegime.tax.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Education Cess (4%):</span>
                    <span className="font-semibold text-white">₹{oldRegime.cess.toLocaleString()}</span>
                  </div>
                </div>
              </div>

              <div className="border-t border-white/5 pt-4 flex justify-between items-baseline font-sans">
                <span className="text-[10px] text-white/40 uppercase tracking-wider font-bold">Estimated Tax</span>
                <span className="text-xl font-black text-white">₹{oldRegime.total.toLocaleString()}</span>
              </div>
            </div>

          </div>
        </div>

      </div>
    </motion.div>
  );
};
export default TaxEstimator;

