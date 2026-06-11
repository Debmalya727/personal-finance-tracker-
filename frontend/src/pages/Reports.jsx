import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Download, Calendar, BarChart3, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = ['#7c3aed', '#06b6d4', '#db2777', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'];

const CustomPieTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d0b25]/95 border border-white/10 rounded-xl p-3 shadow-2xl">
      <p className="text-white font-bold text-sm">{payload[0].name}</p>
      <p className="text-brand-cyan text-sm">₹{payload[0].value?.toLocaleString()}</p>
    </div>
  );
};

export const Reports = () => {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_date: startDate, end_date: endDate })
      });
      const data = await res.json();
      if (data.error) { setError(data.error); setReport(null); }
      else setReport(data);
    } catch (err) {
      setError('Failed to generate report.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    if (!report) return;
    const rows = [
      ['Income & Expense Report'],
      ['Period', `${report.start_date} to ${report.end_date}`],
      [],
      ['Metric', 'Amount (₹)'],
      ['Total Income', report.total_income],
      ['Total Expense', report.total_expense],
      ['Net Savings', report.net_savings],
      [],
      ['Category Breakdown'],
      ['Category', 'Amount (₹)'],
      ...Object.entries(report.expenses_by_category).map(([k, v]) => [k, v]),
    ];
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'finvision_report.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const pieData = report ? Object.entries(report.expenses_by_category).map(([name, value]) => ({ name, value })) : [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="p-8 w-full max-w-5xl flex flex-col gap-8 text-white pb-16"
    >
      <div>
        <div className="flex items-center gap-2 mb-1">
          <FileText size={16} className="text-brand-purple" />
          <span className="text-xs text-white/40 uppercase tracking-widest font-bold font-sans">Analytics</span>
        </div>
        <h1 className="text-4xl font-black tracking-tight font-sans">Reports</h1>
        <p className="text-white/40 text-sm mt-1 font-sans">Generate income & expense reports for any date range</p>
      </div>

      {/* Date Range Form */}
      <div className="glass-panel rounded-2xl p-6 border border-white/5">
        <form onSubmit={handleGenerate} className="flex flex-col md:flex-row gap-4 items-end">
          <div className="flex-1">
            <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">Start Date</label>
            <div className="relative">
              <Calendar size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
              <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} required
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white text-sm focus:outline-none focus:border-brand-purple focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans [color-scheme:dark]" />
            </div>
          </div>
          <div className="flex-1">
            <label className="text-xs font-bold uppercase tracking-widest text-white/50 mb-2 block font-sans">End Date</label>
            <div className="relative">
              <Calendar size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-white/30" />
              <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} required
                className="w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-10 pr-4 text-white text-sm focus:outline-none focus:border-brand-purple focus:ring-2 focus:ring-brand-purple/20 transition-all font-sans [color-scheme:dark]" />
            </div>
          </div>
          <motion.button type="submit" disabled={loading} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
            className="px-6 py-3 rounded-xl font-bold text-white text-sm font-sans whitespace-nowrap"
            style={{ background: 'linear-gradient(135deg, #7c3aed, #db2777)', boxShadow: '0 0 20px rgba(124,58,237,0.3)' }}>
            {loading ? 'Generating...' : 'Generate Report'}
          </motion.button>
        </form>
        {error && <p className="mt-3 text-rose-400 text-sm font-medium font-sans">{error}</p>}
      </div>

      {/* Report Output */}
      {report && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-6">
          {/* Summary Cards */}
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'Total Income', value: report.total_income, Icon: TrendingUp, color: 'text-emerald-400' },
              { label: 'Total Expense', value: report.total_expense, Icon: TrendingDown, color: 'text-rose-400' },
              { label: 'Net Savings', value: report.net_savings, Icon: Minus, color: report.net_savings >= 0 ? 'text-brand-cyan' : 'text-rose-400' },
            ].map(({ label, value, Icon, color }) => (
              <div key={label} className="glass-panel rounded-xl p-5 border border-white/5 flex items-center gap-4">
                <div className={`p-2.5 rounded-xl bg-white/5 ${color}`}><Icon size={18} /></div>
                <div>
                  <span className="text-xs text-white/40 uppercase tracking-wider font-bold font-sans block">{label}</span>
                  <span className={`text-xl font-black mt-0.5 block font-sans ${color}`}>₹{Math.abs(value).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Category Breakdown */}
          {pieData.length > 0 && (
            <div className="glass-panel rounded-2xl p-6 border border-white/5">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  <BarChart3 size={16} className="text-brand-purple" />
                  <h3 className="text-base font-bold text-white font-sans">Expense by Category</h3>
                </div>
                <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleDownloadCSV}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-white/60 text-xs font-bold hover:text-white hover:bg-white/10 transition-all font-sans">
                  <Download size={13} /> Export CSV
                </motion.button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={100} paddingAngle={3} dataKey="value">
                        {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip content={<CustomPieTooltip />} />
                      <Legend formatter={(v) => <span className="text-xs text-white/60 font-sans">{v}</span>} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex flex-col gap-2 justify-center">
                  {pieData.map((item, i) => (
                    <div key={item.name} className="flex items-center justify-between p-3 rounded-xl bg-white/4 border border-white/5">
                      <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                        <span className="text-sm text-white/70 font-sans">{item.name}</span>
                      </div>
                      <span className="font-bold text-sm text-white font-sans">₹{item.value.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

export default Reports;
