---
title: AI-Powered Finance Tracker
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
app_port: 7860
---

# See live at [personal-finance-tracker-wnnr.onrender.com](https://personal-finance-tracker-wnnr.onrender.com)

# 🤖 Fin-AI: An Intelligent Personal & Business Finance Manager

**An advanced, dual-purpose financial management application built with a modern React SPA frontend and Flask backend, leveraging AI for intelligent data extraction and machine learning for proactive financial insights.**

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a comprehensive, full-stack web application designed to solve real-world personal and business financial tracking challenges through modern technology.



## 🎯 Project Motivation & Problem Statement

In a market saturated with generic expense trackers, many users still struggle with the tedious task of manual data entry and the lack of personalized, actionable insights. Personal finance is not one-size-fits-all, and the needs of an individual are vastly different from those of a freelancer or small business owner.

This project tackles these problems by:
1.  **Automating Data Entry:** Utilizing AI to scan and parse receipts, drastically reducing the time and effort required to log transactions.
2.  **Providing Proactive Insights:** Employing machine learning to detect spending anomalies in real-time, offering a layer of financial security and awareness.
3.  **Serving a Dual Role:** Offering a tailored experience for both personal and business finance within a single, cohesive platform, recognizing that many individuals manage both.

---

## ✨ Upgraded Features & Architecture

The application has been upgraded with a modern SPA (Single Page Application) frontend client and a production-hardened Flask API backend:

### 1. **Client-Side (React 18+ SPA)**
* **Modern UI Components**: Sleek, responsive layout built using React, Vite, Tailwind CSS, and Lucide React.
* **Interactive 3D Visualizations**: Real-time asset allocation models using **React Three Fiber (R3F) & Three.js** to display heights of allocation blocks interactively.
* **Advanced Charting**: High-fidelity chart dashboards built with **Recharts** showing cash flow trends.
* **Micro-Animations**: Butter-smooth transitions powered by **Framer Motion**.

### 2. **Production-Grade CSRF Protection**
* Enforced security by implementing a transparent CSRF token verification system.
* Flask-WTF generates a cryptographically secure token on every response.
* A client-side fetch interceptor dynamically captures the `csrf_token` cookie and automatically appends it as the `X-CSRFToken` header to all API mutations (`POST`/`PUT`/`PATCH`/`DELETE`).

### 3. **AI & ML Integration**
* **Gemini AI Receipt Scan**: Upload receipt images to extract description, amount, date, and category automatically.
* **Isolation Forest Anomaly Detection**: Per-user outlier detection to flag suspicious, off-trend purchases.
* **Live Market Rates**: Real-time USD/INR conversions and cryptocurrency price fetches using the CoinGecko API.

---

## 🛠️ Tech Stack

### **Backend**
* **Framework:** Flask (Python 3.10+)
* **ORM:** SQLAlchemy / Flask-SQLAlchemy (relational database target)
* **Migrations:** Flask-Migrate
* **Auth:** Flask-Login
* **Security:** Flask-WTF (Secure CSRF Token verification)

### **Frontend**
* **Framework:** React 18+ (Vite)
* **Styling:** Tailwind CSS, Vanilla CSS
* **Animations:** Framer Motion
* **Visuals:** React Three Fiber, Three.js, Recharts, Lucide Icons

---

## 🚀 Getting Started

To run a local copy of this project:

### **Backend Setup**
1. Clone the repository and navigate inside:
   ```bash
   git clone https://github.com/Debmalya727/personal-finance-tracker-.git
   cd personal-finance-tracker-
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your `.env` file using the keys from `.env.example`.

### **Frontend Setup**
1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install Node modules:
   ```bash
   npm install
   ```
3. Compile and build the React SPA bundle:
   ```bash
   npm run build
   ```
   *(This outputs the static build files directly into the Flask public static folder, enabling the single-port serving configuration).*

### **Run Application**
From the root directory, activate your virtual environment and run:
```bash
flask run
```
Open `http://127.0.0.1:5000` in your web browser.

---

## 🗺️ API Endpoints

The Flask server hosts secure REST API routes under the `/api` prefix:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/dashboard` | Fetches personal monthly stats, balance, and transaction history. |
| `GET` | `/api/net_worth` | Computes assets (cash, schemes, stocks, crypto) and liabilities. |
| `POST` | `/api/reports` | Generates detailed PDF/CSV export data within date limits. |
| `GET` | `/api/business/financials` | Returns business monthly revenue, expenses, and clients data. |
| `POST` | `/api/auth/login` | Secure session sign-in endpoint. |
| `POST` | `/api/auth/register` | User signup endpoint. |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Debmalya Panda - debmalyapanda2004@gmail.com

Project Link: [https://github.com/Debmalya727/personal-finance-tracker-](https://github.com/Debmalya727/personal-finance-tracker-)
