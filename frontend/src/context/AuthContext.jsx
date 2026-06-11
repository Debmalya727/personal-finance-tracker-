import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('/api/auth/status');
      if (response.ok) {
        const data = await response.json();
        if (data.authenticated) {
          setUser(data.user);
        } else {
          setUser(null);
        }
      }
    } catch (error) {
      console.error('Failed to check auth status:', error);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const login = async (username, password) => {
    setLoading(true);
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await response.json();
      if (data.success) {
        setUser(data.user);
        return { success: true };
      }
      return { success: false, message: data.message || 'Login failed.' };
    } catch (error) {
      return { success: false, message: 'Server error. Please try again.' };
    } finally {
      setLoading(false);
    }
  };

  const register = async (username, password, dob, role) => {
    setLoading(true);
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, dob, role })
      });
      const data = await response.json();
      if (data.success) {
        setUser(data.user);
        return { success: true };
      }
      return { success: false, message: data.message || 'Registration failed.' };
    } catch (error) {
      return { success: false, message: 'Server error. Please try again.' };
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/auth/logout', { method: 'POST' });
      if (response.ok) {
        setUser(null);
        return { success: true };
      }
    } catch (error) {
      console.error('Logout failed:', error);
    } finally {
      setLoading(false);
    }
    return { success: false };
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, checkAuthStatus }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
