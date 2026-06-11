import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// CSRF Token fetch interceptor for Flask-WTF CSRFProtect
const originalFetch = window.fetch;
window.fetch = async (url, options = {}) => {
  const method = (options.method || 'GET').toUpperCase();
  if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS') {
    const match = document.cookie.match(/csrf_token=([^;]+)/);
    const csrfToken = match ? decodeURIComponent(match[1]) : null;
    if (csrfToken) {
      options.headers = {
        ...options.headers,
        'X-CSRFToken': csrfToken
      };
    }
  }
  return originalFetch(url, options);
};

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
