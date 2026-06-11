import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Calendar, Download, FileText, CheckCircle2, ShieldAlert } from 'lucide-react';

export const BusinessReports = () => {
  const [form, setForm] = useState({
    start_date: new Date(new Date().getFullYear(), new Date().getMonth(), 1).toISOString().split('T')[0],
    end_date: new Date().toISOString().split('T')[0],
    format: 'pdf'
  });
  const [generating, setGenerating] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);

  const handleDownload = (e) => {
    e.preventDefault();
    setGenerating(true);
    setSuccess(false);
    setError(null);

    fetch('/api/business/reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form)
    })
      .then(res => {
        if (!res.ok) {
          return res.json().then(data => {
            throw new Error(data.error || 'Failed to generate report');
          });
        }
        return res.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `business_report_${form.start_date}_to_${form.end_date}.${form.format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        
        setGenerating(false);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 4000);
      })
      .catch(err => {
        console.error(err);
        setGenerating(false);
        setError(err.message || 'Connection error');
      });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-2xl flex flex-col gap-8 text-white relative z-10 font-sans"
    >
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight">Business Reports</h1>
        <p className="text-white/40 text-sm mt-1">
          Export tax ledgers, client billing statements, and monthly revenue records in PDF or CSV formats.
        </p>
      </div>

      <div className="glass-panel p-6 rounded-2xl border border-white/5 bg-white/2 flex flex-col gap-6">
        <h3 className="text-lg font-bold flex items-center gap-2">
          <Calendar size={18} className="text-brand-purple" />
          Select Report Date Range
        </h3>

        <form onSubmit={handleDownload} className="flex flex-col gap-6">
          <div className="grid grid-cols-2 gap-4">
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

            <div className="flex flex-col gap-1">
              <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">End Date</label>
              <input
                type="date"
                required
                value={form.end_date}
                onChange={(e) => setForm({ ...form, end_date: e.target.value })}
                className="w-full px-4 py-2.5 rounded-xl bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-brand-purple/50 transition-colors"
              />
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] text-white/50 font-bold uppercase tracking-wider">Export Format</label>
            <div className="grid grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => setForm({ ...form, format: 'pdf' })}
                className={`py-3 rounded-xl border flex items-center justify-center gap-2 text-sm font-semibold transition-all ${
                  form.format === 'pdf'
                    ? 'bg-brand-purple/20 border-brand-purple text-brand-purple font-bold'
                    : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10'
                }`}
              >
                <FileText size={16} />
                <span>PDF Statement</span>
              </button>

              <button
                type="button"
                onClick={() => setForm({ ...form, format: 'csv' })}
                className={`py-3 rounded-xl border flex items-center justify-center gap-2 text-sm font-semibold transition-all ${
                  form.format === 'csv'
                    ? 'bg-brand-purple/20 border-brand-purple text-brand-purple font-bold'
                    : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10'
                }`}
              >
                <Download size={16} />
                <span>CSV Spreadsheet</span>
              </button>
            </div>
          </div>

          {error && (
            <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs flex items-center gap-2">
              <ShieldAlert size={14} />
              <span>{error}</span>
            </div>
          )}

          {success && (
            <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs flex items-center gap-2">
              <CheckCircle2 size={14} />
              <span>Report downloaded successfully! Check your downloads directory.</span>
            </div>
          )}

          <button
            type="submit"
            disabled={generating}
            className={`mt-2 w-full py-3.5 rounded-xl font-bold bg-gradient-to-r from-brand-purple to-brand-pink text-white flex items-center justify-center gap-2 shadow-lg shadow-brand-purple/20 hover:shadow-brand-purple/35 transition-all text-sm ${
              generating ? 'opacity-55 cursor-not-allowed' : ''
            }`}
          >
            {generating ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Compiling Financials...</span>
              </>
            ) : (
              <>
                <Download size={16} />
                <span>Generate Business Statement</span>
              </>
            )}
          </button>
        </form>
      </div>
    </motion.div>
  );
};

export default BusinessReports;
