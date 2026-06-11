import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, TrendingUp, DollarSign, RefreshCw, X, AlertTriangle, 
  Trash2, ShoppingBag 
} from 'lucide-react';

export const Investments = () => {
  const [investments, setInvestments] = useState([]);
  const [sales, setSales] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [usdToInrRate, setUsdToInrRate] = useState(83.5);

  // Forms states
  const [assetType, setAssetType] = useState('Stock');
  const [ticker, setTicker] = useState('');
  const [quantity, setQuantity] = useState('');
  const [purchasePrice, setPurchasePrice] = useState('');
  const [purchaseCurrency, setPurchaseCurrency] = useState('INR');
  const [purchaseDate, setPurchaseDate] = useState(new Date().toISOString().split('T')[0]);

  // Sell modal states
  const [sellModalOpen, setSellModalOpen] = useState(false);
  const [selectedInv, setSelectedInv] = useState(null);
  const [sellQuantity, setSellQuantity] = useState('');
  const [sellPrice, setSellPrice] = useState('');
  const [sellDate, setSellDate] = useState(new Date().toISOString().split('T')[0]);

  const fetchInvestments = async (isRef = false) => {
    if (isRef) setRefreshing(true);
    else setLoading(true);
    
    try {
      // First fetch standard investments
      const res = await fetch('/api/investments');
      const data = await res.json();
      
      // Hit net_worth endpoint to get computed current prices from APIs
      const nwRes = await fetch('/api/net_worth');
      const nwData = await nwRes.json();
      
      if (nwData.usd_to_inr_rate) {
        setUsdToInrRate(nwData.usd_to_inr_rate);
      }
      
      // Join standard investments with active net worth prices
      const detailed = data.investments.map(inv => {
        const matchingDetail = nwData.details.investments.find(d => d.id === inv.id);
        return {
          ...inv,
          current_price: matchingDetail ? matchingDetail.current_price : 0,
          current_value_inr: matchingDetail ? matchingDetail.current_value_inr : 0,
        };
      });

      setInvestments(detailed);
      setSales(data.sales || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchInvestments();
  }, []);

  const handleAddSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('/api/investments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          asset_type: assetType,
          ticker_symbol: ticker,
          quantity: parseFloat(quantity),
          purchase_price: parseFloat(purchasePrice),
          purchase_currency: purchaseCurrency,
          purchase_date: purchaseDate
        })
      });
      if (response.ok) {
        setTicker('');
        setQuantity('');
        setPurchasePrice('');
        fetchInvestments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Remove this asset from your portfolio?')) return;
    try {
      const response = await fetch(`/api/investments/${id}`, { method: 'DELETE' });
      if (response.ok) {
        fetchInvestments();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSellClick = (inv) => {
    setSelectedInv(inv);
    setSellQuantity(inv.quantity);
    setSellPrice(inv.current_price || inv.purchase_price);
    setSellModalOpen(true);
  };

  const handleSellSubmit = async (e) => {
    e.preventDefault();
    if (!selectedInv) return;
    try {
      const response = await fetch(`/api/investments/sell/${selectedInv.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sell_quantity: parseFloat(sellQuantity),
          sell_price: parseFloat(sellPrice),
          sell_date: sellDate
        })
      });
      if (response.ok) {
        setSellModalOpen(false);
        fetchInvestments();
      } else {
        const data = await response.json();
        alert(data.message || 'Sell failed.');
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
      className="p-8 w-full max-w-7xl flex flex-col gap-8 text-white relative z-10"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight font-sans">
            Assets & Capital
          </h1>
          <p className="text-white/40 text-sm mt-1 font-sans">
            Track active holdings with live values and manage purchase/sales logs.
          </p>
        </div>

        <button
          onClick={() => fetchInvestments(true)}
          disabled={refreshing}
          className="flex items-center gap-2 bg-white/5 border border-white/5 text-white/80 px-4 py-2 rounded-xl hover:bg-white/10 transition-all text-sm font-sans"
        >
          <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} />
          <span>{refreshing ? 'Refreshing...' : 'Refresh Prices'}</span>
        </button>
      </div>

      {/* Forms & Add Panel */}
      <div className="glass-panel p-6 rounded-2xl border border-white/5 shadow-xl bg-white/2">
        <h3 className="text-lg font-bold mb-4 font-sans flex items-center gap-2">
          <ShoppingBag size={18} className="text-brand-purple" />
          <span>Add New Portfolio Item</span>
        </h3>
        
        <form onSubmit={handleAddSubmit} className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 items-end">
          <div>
            <label className="text-xs text-white/50 font-semibold uppercase tracking-wider mb-1.5 block font-sans">Asset Type</label>
            <select
              value={assetType}
              onChange={(e) => {
                setAssetType(e.target.value);
                setPurchaseCurrency(e.target.value === 'Stock' ? 'INR' : 'USD');
              }}
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-3 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
            >
              <option value="Stock">Stock</option>
              <option value="Crypto">Crypto</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-white/50 font-semibold uppercase tracking-wider mb-1.5 block font-sans">Ticker / ID</label>
            <input
              type="text"
              required
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder={assetType === 'Stock' ? 'e.g. RELIANCE.NS' : 'e.g. bitcoin'}
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-3 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
            />
          </div>
          <div>
            <label className="text-xs text-white/50 font-semibold uppercase tracking-wider mb-1.5 block font-sans">Quantity</label>
            <input
              type="number"
              step="0.0001"
              required
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="0.00"
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-3 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
            />
          </div>
          <div>
            <label className="text-xs text-white/50 font-semibold uppercase tracking-wider mb-1.5 block font-sans">Purchase Price</label>
            <input
              type="number"
              step="0.01"
              required
              value={purchasePrice}
              onChange={(e) => setPurchasePrice(e.target.value)}
              placeholder="0.00"
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-3 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
            />
          </div>
          <div>
            <label className="text-xs text-white/50 font-semibold uppercase tracking-wider mb-1.5 block font-sans">Currency</label>
            <select
              value={purchaseCurrency}
              onChange={(e) => setPurchaseCurrency(e.target.value)}
              disabled={assetType === 'Stock'}
              className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-3 text-white text-sm focus:outline-none focus:border-brand-purple transition-all disabled:opacity-50 font-sans"
            >
              <option value="INR">INR (₹)</option>
              <option value="USD">USD ($)</option>
            </select>
          </div>
          <button
            type="submit"
            className="w-full bg-gradient-to-r from-brand-purple to-brand-pink text-white font-bold py-2.5 px-4 rounded-xl hover:brightness-110 active:scale-[0.98] transition-all glow-purple font-sans"
          >
            Buy Asset
          </button>
        </form>
      </div>

      {/* Main Portfolios / Tabs Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Active Holdings List */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl bg-white/2">
          <h3 className="text-lg font-bold mb-4 font-sans flex items-center gap-2">
            <TrendingUp size={18} className="text-brand-cyan" />
            <span>Active Holdings</span>
          </h3>

          {loading ? (
            <div className="py-12 flex justify-center">
              <div className="w-8 h-8 border-t-2 border-brand-purple border-solid rounded-full animate-spin"></div>
            </div>
          ) : investments.length === 0 ? (
            <div className="py-12 text-center text-white/30 text-sm font-sans">
              No holdings recorded yet. Buy an asset to get started.
            </div>
          ) : (
            <div className="flex flex-col gap-3.5">
              {investments.map((inv) => {
                const isUSD = inv.purchase_currency === 'USD';
                const currencySymbol = isUSD ? '$' : '₹';
                
                // Live profit in its traded/purchase currency
                const profitInCurrency = (inv.current_price - inv.purchase_price) * inv.quantity;
                const isProfit = profitInCurrency >= 0;
                
                // Converted profit in base INR for comparison
                const profitInInr = isUSD ? profitInCurrency * usdToInrRate : profitInCurrency;

                return (
                  <div 
                    key={inv.id} 
                    className="flex flex-col md:flex-row md:items-center justify-between p-4 rounded-xl bg-white/5 border border-white/5 hover:border-brand-purple/20 hover:bg-white/10 transition-all gap-4 group"
                  >
                    <div>
                      <div className="flex items-center gap-2.5">
                        <span className="font-bold text-base text-white font-sans">{inv.ticker_symbol.toUpperCase()}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider bg-white/10 text-white/60 font-sans">
                          {inv.asset_type}
                        </span>
                      </div>
                      <div className="text-xs text-white/40 font-semibold font-sans mt-1">
                        Qty: {inv.quantity} • Buy: {currencySymbol}{inv.purchase_price.toLocaleString(isUSD ? 'en-US' : 'en-IN')}
                      </div>
                    </div>

                    <div className="flex items-center gap-8 justify-between md:justify-end">
                      <div className="text-right">
                        <span className="text-white/40 text-[10px] font-semibold uppercase tracking-wider block font-sans">Current Price</span>
                        <span className="font-bold text-white text-sm font-sans">
                          {currencySymbol}{inv.current_price ? inv.current_price.toLocaleString(isUSD ? 'en-US' : 'en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '---'}
                        </span>
                      </div>

                      <div className="text-right">
                        <span className="text-white/40 text-[10px] font-semibold uppercase tracking-wider block font-sans">Profit / Loss</span>
                        <span className={`font-extrabold text-sm font-sans ${isProfit ? 'text-emerald-400' : 'text-brand-pink'} block`}>
                          {isProfit ? '+' : '-'}{currencySymbol}{Math.abs(profitInCurrency).toLocaleString(isUSD ? 'en-US' : 'en-IN', { maximumFractionDigits: 2 })}
                        </span>
                        {isUSD && (
                          <span className={`text-[10px] font-semibold block ${isProfit ? 'text-emerald-500/60' : 'text-brand-pink/60'}`}>
                            ({isProfit ? '+' : '-'}₹{Math.abs(profitInInr).toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                          </span>
                        )}
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={() => handleSellClick(inv)}
                          className="px-3 py-1.5 bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/25 rounded-lg text-xs font-bold transition-all font-sans"
                        >
                          Sell
                        </button>
                        <button
                          onClick={() => handleDelete(inv.id)}
                          className="p-1.5 text-white/30 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg border border-transparent hover:border-rose-500/20 transition-all"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Sales History Log */}
        <div className="glass-panel rounded-2xl p-6 border border-white/5 shadow-2xl flex flex-col bg-white/2">
          <h3 className="text-lg font-bold mb-4 font-sans flex items-center gap-2">
            <DollarSign size={18} className="text-brand-pink" />
            <span>Sales History</span>
          </h3>

          <div className="flex flex-col gap-3.5 overflow-y-auto max-h-[400px]">
            {sales.length === 0 ? (
              <div className="text-center py-12 text-white/30 text-xs font-sans">
                No past transactions recorded.
              </div>
            ) : (
              sales.map((sale) => (
                <div key={sale.id} className="p-3 rounded-xl bg-white/5 border border-white/5 text-xs flex flex-col gap-1.5 font-sans">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-white">{sale.ticker_symbol.toUpperCase()}</span>
                    <span className={`font-bold px-1.5 py-0.5 rounded text-[10px] ${
                      sale.capital_gain >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-brand-pink/10 text-brand-pink'
                    }`}>
                      {sale.capital_gain >= 0 ? '+' : '-'} ₹{Math.abs(sale.capital_gain).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-white/40">
                    Sold {sale.quantity} units on {sale.sell_date}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* --- SELL MODAL --- */}
      <AnimatePresence>
        {sellModalOpen && selectedInv && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 z-50">
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="w-full max-w-md glass-panel p-6 rounded-3xl border border-white/10 bg-white/2"
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-white font-sans flex items-center gap-2">
                  <span>Sell {selectedInv.ticker_symbol.toUpperCase()}</span>
                </h3>
                <button onClick={() => setSellModalOpen(false)} className="text-white/40 hover:text-white">
                  <X size={18} />
                </button>
              </div>

              <form onSubmit={handleSellSubmit} className="flex flex-col gap-4">
                <div className="bg-white/5 border border-white/5 p-3 rounded-xl text-xs text-white/60 mb-2 font-sans">
                  Available Holdings: <strong>{selectedInv.quantity} units</strong><br/>
                  Purchase Price: <strong>{selectedInv.purchase_currency === 'USD' ? '$' : '₹'}{selectedInv.purchase_price}</strong>
                </div>

                <div>
                  <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                    Quantity to Sell
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    required
                    max={selectedInv.quantity}
                    value={sellQuantity}
                    onChange={(e) => setSellQuantity(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                  />
                </div>

                <div>
                  <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                    Selling Price (Per Unit)
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    required
                    value={sellPrice}
                    onChange={(e) => setSellPrice(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                  />
                </div>

                <div>
                  <label className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-1.5 block font-sans">
                    Sell Date
                  </label>
                  <input
                    type="date"
                    required
                    value={sellDate}
                    onChange={(e) => setSellDate(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 px-4 text-white text-sm focus:outline-none focus:border-brand-purple transition-all font-sans"
                  />
                </div>

                <button
                  type="submit"
                  className="w-full mt-4 bg-gradient-to-r from-rose-600 to-brand-pink text-white font-bold py-3 rounded-xl hover:brightness-110 active:scale-[0.99] transition-all font-sans shadow-lg shadow-rose-500/10"
                >
                  Confirm Sale
                </button>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Investments;
