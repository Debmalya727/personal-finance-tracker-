import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { AuthProvider, useAuth } from './context/AuthContext';

// Navigation Components
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';

// Pages — Core
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { Dashboard } from './pages/Dashboard';
import { Transactions } from './pages/Transactions';
import { Investments } from './pages/Investments';
import { NetWorth } from './pages/NetWorth';
import { Schemes } from './pages/Schemes';
import { Loans } from './pages/Loans';
import { TaxEstimator } from './pages/TaxEstimator';

// Pages — Tools (previously in old navbar only)
import { Budget } from './pages/Budget';
import { Categories } from './pages/Categories';
import { SalaryManager } from './pages/SalaryManager';
import { Profile } from './pages/Profile';
import { Reports } from './pages/Reports';
import { AiInsights } from './pages/AiInsights';

// Pages — Business
import { BusinessConsole } from './pages/BusinessConsole';
import { Clients } from './pages/Clients';
import { BusinessTransactions } from './pages/BusinessTransactions';
import { BusinessFinancials } from './pages/BusinessFinancials';
import { BusinessInvestments } from './pages/BusinessInvestments';
import { BusinessLoans } from './pages/BusinessLoans';
import { BusinessReports } from './pages/BusinessReports';
import { BusinessInsights } from './pages/BusinessInsights';

// Protected Route
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="w-screen h-screen flex flex-col items-center justify-center bg-[#030014] gap-4">
        <div className="relative w-16 h-16">
          <div className="absolute inset-0 rounded-full border-2 border-brand-purple/20 animate-ping" />
          <div className="absolute inset-2 rounded-full border-t-2 border-brand-purple animate-spin" />
        </div>
        <p className="text-white/30 text-sm font-medium font-sans tracking-widest uppercase">Initializing...</p>
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;
  return children;
};

const MainLayout = () => {
  const [isBusiness, setIsBusiness] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  // Close mobile sidebar on route change
  React.useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-[#030014] relative overflow-hidden">
      {/* Global Ambient Glow */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-[-30%] left-[-10%] w-[700px] h-[700px] rounded-full bg-purple-900/15 blur-[150px]" />
        <div className="absolute bottom-[-30%] right-[-10%] w-[600px] h-[600px] rounded-full bg-cyan-900/10 blur-[120px]" />
        <div className="absolute top-[40%] left-[40%] w-[400px] h-[400px] rounded-full bg-pink-900/8 blur-[100px]" />
      </div>
      <div className="fixed inset-0 bg-grid-pattern opacity-20 pointer-events-none -z-10" />

      <Sidebar 
        isBusiness={isBusiness} 
        setIsBusiness={setIsBusiness} 
        mobileOpen={mobileOpen} 
        setMobileOpen={setMobileOpen} 
      />
      <Navbar setMobileOpen={setMobileOpen} />

      <main className="lg:ml-64 pt-[70px] min-h-screen flex justify-center px-4 md:px-8">
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            {/* Personal */}
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="/investments" element={<Investments />} />
            <Route path="/net-worth" element={<NetWorth />} />
            <Route path="/schemes" element={<Schemes />} />
            <Route path="/loans" element={<Loans />} />
            <Route path="/tax" element={<TaxEstimator />} />

            {/* Tools (previously missing) */}
            <Route path="/budget" element={<Budget />} />
            <Route path="/categories" element={<Categories />} />
            <Route path="/salary" element={<SalaryManager />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/ai-insights" element={<AiInsights />} />

            {/* Business */}
            <Route path="/business" element={<BusinessConsole />} />
            <Route path="/business/transactions" element={<BusinessTransactions />} />
            <Route path="/business/clients" element={<Clients />} />
            <Route path="/business/financials" element={<BusinessFinancials />} />
            <Route path="/business/investments" element={<BusinessInvestments />} />
            <Route path="/business/loans" element={<BusinessLoans />} />
            <Route path="/business/reports" element={<BusinessReports />} />
            <Route path="/business/insights" element={<BusinessInsights />} />

            <Route path="*" element={<Navigate to={isBusiness ? "/business" : "/dashboard"} replace />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  );
};

export const App = () => (
  <AuthProvider>
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/*" element={<ProtectedRoute><MainLayout /></ProtectedRoute>} />
      </Routes>
    </Router>
  </AuthProvider>
);

export default App;
