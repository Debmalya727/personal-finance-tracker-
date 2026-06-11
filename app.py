# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, Response, send_from_directory
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from models import (
    db, User, Transaction, FixedScheme, Salary, Investment, SoldInvestment, Loan,
    BusinessTransaction, BusinessClient, BusinessInvestment, BusinessLoan,
    BusinessMetrics, Category, Budget
)
from datetime import datetime, date, timedelta
import json
import os
import re
import io
import csv
import uuid
import traceback

from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect, generate_csrf
from dateutil.relativedelta import relativedelta
from dateutil import parser as dateparser
from sqlalchemy import func
from dotenv import load_dotenv

import yfinance as yf
from pycoingecko import CoinGeckoAPI
import cloudinary
import cloudinary.uploader

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import joblib

from fpdf import FPDF
from PIL import Image
import requests
from huggingface_hub import InferenceClient
from google import genai

# --- CoinGecko Ticker to ID Mapping Cache ---
COINGECKO_SYMBOL_MAP = {}
LAST_MAP_FETCH = None

def get_coingecko_id(symbol):
    global COINGECKO_SYMBOL_MAP, LAST_MAP_FETCH
    symbol_lower = symbol.lower().strip()
    
    # Common cryptos static mapping to avoid API calls completely for them
    common_map = {
        'btc': 'bitcoin',
        'bitcoin': 'bitcoin',
        'eth': 'ethereum',
        'ethereum': 'ethereum',
        'usdt': 'tether',
        'tether': 'tether',
        'bnb': 'binancecoin',
        'sol': 'solana',
        'solana': 'solana',
        'xrp': 'ripple',
        'ada': 'cardano',
        'cardano': 'cardano',
        'doge': 'dogecoin',
        'dogecoin': 'dogecoin',
        'shib': 'shiba-inu',
        'avax': 'avalanche-2',
        'dot': 'polkadot',
        'matic': 'matic-network',
        'link': 'chainlink',
        'trx': 'tron',
        'ltc': 'litecoin',
        'near': 'near',
        'uni': 'uniswap',
        'bch': 'bitcoin-cash',
        'etc': 'ethereum-classic',
        'xlm': 'stellar',
        'fil': 'filecoin',
        'ldo': 'lido-dao',
        'hbar': 'hbar',
        'apt': 'aptos',
        'vet': 'vechain',
        'icp': 'internet-computer',
        'grt': 'the-graph',
        'ftm': 'fantom',
        'theta': 'theta-token',
        'algo': 'algorand',
        'imx': 'immutable-x',
        'stx': 'blockstack',
        'eos': 'eos',
        'egld': 'elrond-erd-2',
        'sand': 'the-sandbox',
        'mana': 'decentraland',
        'axs': 'axie-infinity',
        'aave': 'aave',
        'flow': 'flow',
        'chz': 'chiliz',
        'crv': 'curve-dao-token',
        'mkr': 'maker',
        'gala': 'gala',
        'snx': 'havven',
        'dydx': 'dydx',
        'lrc': 'loopring',
        'enj': 'enjincoin',
        'bat': 'basic-attention-token',
        'zil': 'zilliqa',
        'one': 'harmony',
        'celo': 'celo',
        'rvn': 'ravencoin',
        'waves': 'waves',
        'dash': 'dash',
        'zec': 'zcash',
        'xmr': 'monero',
    }
    
    if symbol_lower in common_map:
        return common_map[symbol_lower]

    # Fetch from CoinGecko list with cache expiration (1 day)
    now = datetime.now()
    if not COINGECKO_SYMBOL_MAP or not LAST_MAP_FETCH or (now - LAST_MAP_FETCH).days >= 1:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/coins/list', timeout=5)
            if response.status_code == 200:
                coins_list = response.json()
                new_map = {}
                for coin in coins_list:
                    sym = coin['symbol'].lower()
                    cid = coin['id']
                    # Keep the shortest ID for conflicts
                    if sym not in new_map or len(cid) < len(new_map[sym]):
                        new_map[sym] = cid
                COINGECKO_SYMBOL_MAP = new_map
                LAST_MAP_FETCH = now
        except Exception as e:
            print(f"Error fetching CoinGecko coins list: {e}")
            
    # Check if the symbol_lower is already a valid CoinGecko ID
    if COINGECKO_SYMBOL_MAP:
        if symbol_lower in COINGECKO_SYMBOL_MAP.values():
            return symbol_lower
        if symbol_lower in COINGECKO_SYMBOL_MAP:
            return COINGECKO_SYMBOL_MAP[symbol_lower]
        
    return symbol_lower

load_dotenv()
app = Flask(__name__)

# --- Environment Detection ---
IS_PRODUCTION = os.getenv('FLASK_ENV', 'development') == 'production'

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'a-super-secret-key-that-is-long-and-random')

app.config.update(
    WTF_CSRF_ENABLED=True,
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True
)


db_uri = os.getenv('DATABASE_URI')
if db_uri:
    db_uri = db_uri.strip()
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# --- CONFIGURE CLOUDINARY USING ENVIRONMENT VARIABLES ---
cloudinary.config(
  cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME'),
  api_key = os.getenv('CLOUDINARY_API_KEY'),
  api_secret = os.getenv('CLOUDINARY_API_SECRET'),
  secure=True
)

# AI runs entirely via Google Gemini cloud API — no local models needed.

# Configure Google Gemini API Client
_gemini_client = None
try:
    _api_key = os.getenv("GOOGLE_API_KEY")
    if _api_key:
        _gemini_client = genai.Client(api_key=_api_key)
        print("Google Gemini AI client configured successfully.")
    else:
        print("Warning: GOOGLE_API_KEY not set. Gemini features will be unavailable.")
except Exception as e:
    print(f"Could not configure Google Gemini AI. Check your GOOGLE_API_KEY. Error: {e}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

csrf = CSRFProtect(app)
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)
migrate = Migrate(app, db)

from api import api_bp
app.register_blueprint(api_bp)
# Enable CSRF cookie injection on all responses
@app.after_request
def set_csrf_cookie(response):
    response.set_cookie(
        'csrf_token',
        generate_csrf(),
        samesite='Lax',
        secure=IS_PRODUCTION,
        httponly=False
    )
    return response

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Tax Helper Functions ---
def calculate_new_regime_tax(gross_income):
    standard_deduction = 50000
    taxable_income = gross_income - standard_deduction
    if taxable_income < 0: taxable_income = 0
    tax = 0
    if taxable_income <= 300000: tax = 0
    elif taxable_income <= 600000: tax = (taxable_income - 300000) * 0.05
    elif taxable_income <= 900000: tax = 15000 + (taxable_income - 600000) * 0.10
    elif taxable_income <= 1200000: tax = 45000 + (taxable_income - 900000) * 0.15
    elif taxable_income <= 1500000: tax = 90000 + (taxable_income - 1200000) * 0.20
    else: tax = 150000 + (taxable_income - 1500000) * 0.30
    cess = tax * 0.04
    total_tax = tax + cess
    return {'regime': 'New', 'gross_income': gross_income, 'taxable_income': taxable_income, 'total_deductions': 0, 'standard_deduction': standard_deduction, 'income_tax': tax, 'cess': cess, 'total_tax': total_tax}

def calculate_old_regime_tax(gross_income, total_deductions, age):
    standard_deduction = 50000
    taxable_income = gross_income - total_deductions - standard_deduction
    if taxable_income < 0: taxable_income = 0
    tax = 0
    if age < 60:
        if taxable_income <= 250000: tax = 0
        elif taxable_income <= 500000: tax = (taxable_income - 250000) * 0.05
        elif taxable_income <= 1000000: tax = 12500 + (taxable_income - 500000) * 0.20
        else: tax = 112500 + (taxable_income - 1000000) * 0.30
    elif age < 80:
        if taxable_income <= 300000: tax = 0
        elif taxable_income <= 500000: tax = (taxable_income - 300000) * 0.05
        elif taxable_income <= 1000000: tax = 10000 + (taxable_income - 500000) * 0.20
        else: tax = 110000 + (taxable_income - 1000000) * 0.30
    else:
        if taxable_income <= 500000: tax = 0
        elif taxable_income <= 1000000: tax = (taxable_income - 500000) * 0.20
        else: tax = 100000 + (taxable_income - 1000000) * 0.30
    if taxable_income <= 500000: tax = 0
    cess = tax * 0.04
    total_tax = tax + cess
    return {'regime': 'Old', 'gross_income': gross_income, 'taxable_income': taxable_income, 'total_deductions': total_deductions, 'standard_deduction': standard_deduction, 'income_tax': tax, 'cess': cess, 'total_tax': total_tax}

@app.route('/healthz')
def healthz():
    return "OK", 200

# app.py (Replace the process_receipt function)
    
# --- AI & Utility Routes ---
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    transaction = BusinessTransaction.query.filter_by(receipt_filename=filename, user_id=current_user.id).first_or_404()
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- AI Receipt Scanner: Google Gemini (Cloud-based, Fast & Free) ---
@app.route('/api/process-receipt-fast', methods=['POST'])
@login_required
def process_receipt_fast():
    if 'receipt_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['receipt_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no file selected'}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        if not _gemini_client:
            return jsonify({'error': 'AI scanner not configured. Please set GOOGLE_API_KEY.'}), 503
        user_categories = [cat.name for cat in current_user.categories]
        prompt = f"""
Analyze this receipt image and extract the following fields:
- description: the merchant or store name
- amount: the total amount paid (number only, no currency symbol)
- date: the transaction date in YYYY-MM-DD format
- type: either "expense" or "revenue"
- category_name: best match from this list: {user_categories}

Return ONLY valid JSON with exactly these keys: description, amount, date, type, category_name
"""
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image]
        )
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(json_text)
        # Resolve category to ID
        category_id = None
        if extracted_data.get("category_name"):
            category_obj = Category.query.filter_by(
                user_id=current_user.id,
                name=extracted_data["category_name"]
            ).first()
            if category_obj:
                category_id = category_obj.id
        extracted_data["category_id"] = category_id
        return jsonify(extracted_data)
    except Exception as e:
        print(f"Error processing receipt with Gemini: {e}")
        return jsonify({'error': 'AI receipt scan failed. Please try again or enter details manually.'}), 500


# --- AI Method 2: Also uses Gemini (accurate mode) ---
@app.route('/api/process-receipt-accurate', methods=['POST'])
@login_required
def process_receipt_accurate():
    """Routes to the same Gemini-powered scanner for consistency."""
    return process_receipt_fast()


@app.route('/business/insights', methods=['GET', 'POST'])
@login_required
def business_insights():
    if request.method == 'POST':
        try:
            month = int(request.form.get('month'))
            year = int(request.form.get('year'))

            start_date = date(year, month, 1)
            end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

            transactions = BusinessTransaction.query.filter(
                BusinessTransaction.user_id == current_user.id,
                BusinessTransaction.date.between(start_date, end_date)
            ).all()

            if not transactions:
                flash('No transactions found for the selected period.', 'info')
                return redirect(url_for('business_insights'))

            # Aggregate data for the AI
            total_revenue = sum(t.amount for t in transactions if t.type == 'revenue')
            total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
            net_profit = total_revenue - total_expenses
            
            expenses_by_category = {}
            for t in transactions:
                if t.type == 'expense':
                    expenses_by_category[t.category.name] = expenses_by_category.get(t.category.name, 0) + t.amount
            
            # Find the top expense category
            top_expense_category = max(expenses_by_category, key=expenses_by_category.get) if expenses_by_category else "N/A"

            # --- AI Prompt Engineering ---
            prompt = f"""
            As a friendly financial analyst, analyze the following monthly data for a small business owner and provide a concise, easy-to-understand summary in bullet points. Focus on key takeaways.

            DATA FOR {start_date.strftime('%B %Y')}:
            - Total Revenue: {total_revenue:.2f}
            - Total Expenses: {total_expenses:.2f}
            - Net Profit/Loss: {net_profit:.2f}
            - Total Number of Transactions: {len(transactions)}
            - Top Expense Category: {top_expense_category} with an amount of {expenses_by_category.get(top_expense_category, 0):.2f}

            Based on this data, generate a short summary.
            """

            client = InferenceClient()
            response_text = client.text_generation(prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=250)
            
            return render_template('business/insights.html', insights=response_text, selected_month=month, selected_year=year)

        except Exception as e:
            print(f"Error generating insights: {e}")
            flash('An AI error occurred while generating the report.', 'danger')

    # For GET request, show the form with the current month selected
    today = date.today()
    return render_template('business/insights.html', insights=None, selected_month=today.month, selected_year=today.year)

def check_transaction_anomaly(new_transaction, user_id):
    """
    Checks if a new transaction is an anomaly based on historical data.
    Returns True if it's an anomaly, False otherwise.
    """
    # We only check expenses for anomalies for now
    if new_transaction.type != 'expense':
        return False

    # Fetch historical expenses for the user
    historical_expenses = BusinessTransaction.query.filter_by(
        user_id=user_id, 
        type='expense'
    ).order_by(BusinessTransaction.date.desc()).limit(200).all()

    # Need at least 20 historical data points to make a sensible model
    if len(historical_expenses) < 20:
        return False

    # Prepare data for the model
    df = pd.DataFrame([{'amount': t.amount} for t in historical_expenses])
    
    # Train the Isolation Forest model
    # Contamination defines the expected proportion of anomalies (e.g., 5%)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df[['amount']])
    
    # Predict if the new transaction is an anomaly (-1 for anomalies, 1 for inliers)
    new_amount_df = pd.DataFrame([{'amount': new_transaction.amount}])
    prediction = model.predict(new_amount_df[['amount']])
    
    return prediction[0] == -1
# --- Routes ---
# -------------------
# Login Route
# -------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role.strip().lower() == 'business':
            return redirect(url_for('business_dashboard'))
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role.strip().lower() == 'business':
                return redirect(url_for('business_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')

    return render_template('login.html')

# -------------------
# Register Route
# -------------------
@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    dob_str = request.form.get('dob')
    role = request.form.get('role')

    # Prevent duplicate usernames
    if User.query.filter_by(username=username).first():
        flash('Username already exists.', 'error')
        return redirect(url_for('login'))

    dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, password=hashed_password, dob=dob, role=role)
    db.session.add(new_user)
    db.session.commit()
    flash('Registration successful! Please log in.', 'success')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    # Redirect based on role
    if current_user.role.strip().lower() == 'business':
        return redirect(url_for('business_dashboard'))
    return redirect(url_for('dashboard'))

# app.py (New routes for Category Management)

# In app.py

@app.route('/categories', methods=['GET', 'POST'])
@login_required
def manage_categories():
    if request.method == 'POST':
        # Get data from the updated form
        category_name = request.form.get('name')
        category_type = request.form.get('type') # Gets 'income' or 'expense'

        # Validate that both fields are present
        if not category_name or not category_type:
            flash('Category name and type are required.', 'danger')
        elif len(category_name) < 2:
            flash('Category name is too short.', 'warning')
        else:
            # Create the new category object with the type included
            new_category = Category(
                name=category_name,
                type=category_type,
                user_id=current_user.id
            )
            db.session.add(new_category)
            db.session.commit()
            flash('Category added successfully!', 'success')
        
        return redirect(url_for('manage_categories'))

    # This part remains the same: get all categories to display in the list
    categories = Category.query.filter_by(user_id=current_user.id).order_by(Category.name).all()
    # CORRECTED LINE
    return render_template('categories.html', categories=categories)

@app.route('/delete_category/<int:category_id>', methods=['POST'])
@login_required
def delete_category(category_id):
    category = db.session.get(Category, category_id)
    if not category or category.user_id != current_user.id:
        flash('Category not found or unauthorized.', 'danger')
    # Prevent deleting a category if it's in use
    elif category.transactions or category.business_transactions:
        flash(f'Cannot delete category "{category.name}" because it is currently in use.', 'danger')
    else:
        db.session.delete(category)
        db.session.commit()
        flash('Category deleted.', 'success')
    return redirect(url_for('manage_categories'))

# In app.py - Replace your existing dashboard function with this one

@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    start_of_month = today.replace(day=1)
    end_of_month = start_of_month + relativedelta(months=1)
    
    # --- SALARY AUTO-CREDIT LOGIC ---
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    if salary_details and salary_details.monthly_gross > 0:
        salary_credited_this_month = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description == "Monthly Salary",
            Transaction.date >= start_of_month,
            Transaction.date < end_of_month
        ).first()

        if not salary_credited_this_month:
            # FIX: Find or create the 'Salary' category object
            salary_category = Category.query.filter_by(user_id=current_user.id, name='Salary').first()
            if not salary_category:
                salary_category = Category(name='Salary', type='income', user_id=current_user.id)
                db.session.add(salary_category)
                db.session.commit()

            salary_transaction = Transaction(
                description="Monthly Salary",
                amount=salary_details.monthly_gross,
                type="income",
                category_id=salary_category.id, # CORRECT: Use the category's ID
                date=start_of_month,
                user_id=current_user.id
            )
            db.session.add(salary_transaction)
            db.session.commit()
            flash(f"Auto-credited salary of ₹{salary_details.monthly_gross} for this month.", "info")

    # --- EMI AUTO-DEBIT LOGIC ---
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    if user_loans:
        # FIX: Find or create the 'EMI' category object once before the loop
        emi_category = Category.query.filter_by(user_id=current_user.id, name='EMI').first()
        if not emi_category:
            emi_category = Category(name='EMI', type='expense', user_id=current_user.id)
            db.session.add(emi_category)
            db.session.commit()

        for loan in user_loans:
            emi_debited = Transaction.query.filter(
                Transaction.user_id == current_user.id,
                Transaction.description == f"EMI for {loan.loan_name}",
                Transaction.date >= start_of_month,
                Transaction.date < end_of_month
            ).first()
            
            loan_end_date = loan.start_date + relativedelta(months=+loan.tenure_months)
            if not emi_debited and today <= loan_end_date:
                db.session.add(Transaction(
                    description=f"EMI for {loan.loan_name}",
                    amount=loan.emi_amount,
                    type="expense",
                    category_id=emi_category.id, # CORRECT: Use the category's ID
                    date=start_of_month,
                    user_id=current_user.id
                ))
                db.session.commit()
                flash(f"Auto-debited EMI of ₹{loan.emi_amount} for {loan.loan_name}.", "info")

    # --- YOUR EXISTING ANOMALY DETECTION LOGIC (UNCHANGED) ---
    all_expenses = Transaction.query.filter_by(user_id=current_user.id, type='expense').all()
    if len(all_expenses) > 20:
        df = pd.DataFrame([(t.amount, t.description) for t in all_expenses], columns=['amount', 'description'])
        model = IsolationForest(contamination=0.05) 
        df['anomaly'] = model.fit_predict(df[['amount']])
        if not df.empty:
            last_transaction = all_expenses[-1]
            if df.iloc[-1]['anomaly'] == -1:
                 flash(f"Unusual spending detected: ₹{last_transaction.amount} for '{last_transaction.description}'. Please review.", "warning")

    # --- YOUR EXISTING MONTHLY TOTALS LOGIC (UNCHANGED) ---
    monthly_transactions = Transaction.query.filter(
        Transaction.user_id == current_user.id,
        Transaction.date >= start_of_month,
        Transaction.date < end_of_month).all()
    monthly_income = sum(t.amount for t in monthly_transactions if t.type == 'income')
    monthly_expense = sum(t.amount for t in monthly_transactions if t.type == 'expense')
    
    all_transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    balance = sum(t.amount for t in all_transactions if t.type == 'income') - sum(t.amount for t in all_transactions if t.type == 'expense')
    
    recent_transactions = Transaction.query.filter_by(
        user_id=current_user.id).order_by(Transaction.date.desc()).limit(5).all()

    chart_data = {'labels': ['Monthly Income', 'Monthly Expense'], 'data': [monthly_income, monthly_expense]}

    # --- YOUR EXISTING BUDGET CALCULATION LOGIC (UNCHANGED) ---
    current_month_num = datetime.utcnow().month
    current_year = datetime.utcnow().year

    total_budgeted = db.session.query(func.sum(Budget.amount)).filter(
        Budget.user_id == current_user.id,
        Budget.month == current_month_num,
        Budget.year == current_year
    ).scalar() or 0.0
    
    total_spent_this_month = monthly_expense

    budget_summary = {
        'total_budgeted': total_budgeted,
        'total_spent': total_spent_this_month,
        'percentage_spent': int((total_spent_this_month / total_budgeted) * 100) if total_budgeted > 0 else 0
    }
    
    return render_template('dashboard.html', 
                           user=current_user, 
                           balance=balance, 
                           monthly_income=monthly_income, 
                           monthly_expense=monthly_expense, 
                           recent_transactions=recent_transactions,
                           chart_data=chart_data,
                           budget_summary=budget_summary)


@app.route('/business_dashboard')
@login_required
def business_dashboard():
    # Ensure user has business role
    if current_user.role.strip().lower() != 'business':
        return redirect(url_for('dashboard'))
        
    # --- YOUR EXISTING BUSINESS LOGIC (UNCHANGED) ---
    today = date.today()
    start_of_month = today.replace(day=1)
    end_of_month = start_of_month + relativedelta(months=1)

    transactions_this_month = BusinessTransaction.query.filter(
        BusinessTransaction.user_id == current_user.id,
        BusinessTransaction.date >= start_of_month,
        BusinessTransaction.date < end_of_month
    ).all()
    
    monthly_revenue = sum(t.amount for t in transactions_this_month if t.type == 'revenue')
    monthly_expenses = sum(t.amount for t in transactions_this_month if t.type == 'expense')
    net_profit = monthly_revenue - monthly_expenses
    
    investments = BusinessInvestment.query.filter_by(user_id=current_user.id).all()
    loans = BusinessLoan.query.filter_by(user_id=current_user.id).all()
    
    summary_data = {
        'active_clients': BusinessClient.query.filter_by(user_id=current_user.id, status='Active').count(),
        'total_investment_value': sum(inv.current_value for inv in investments),
        'total_outstanding_loans': sum(loan.remaining_balance for loan in loans)
    }

    chart_data = {"labels": ["Revenue", "Expenses"], "values": [monthly_revenue, monthly_expenses]}

    # --- NEW BUDGET CALCULATION LOGIC FOR BUSINESS ---
    # Note: This assumes business budgets are set using the same Category/Budget models.
    current_month_num = datetime.utcnow().month
    current_year = datetime.utcnow().year

    total_budgeted = db.session.query(func.sum(Budget.amount)).filter(
        Budget.user_id == current_user.id,
        Budget.month == current_month_num,
        Budget.year == current_year
    ).scalar() or 0.0
    
    # We use the monthly_expenses calculated from BusinessTransactions
    total_spent_this_month = monthly_expenses

    budget_summary = {
        'total_budgeted': total_budgeted,
        'total_spent': total_spent_this_month,
        'percentage_spent': int((total_spent_this_month / total_budgeted) * 100) if total_budgeted > 0 else 0
    }
    # --- END OF NEW BUDGET LOGIC ---

    # Add budget_summary to the return statement
    return render_template(
        'dashboard_business.html', 
        user=current_user, 
        monthly_revenue=monthly_revenue, 
        monthly_expenses=monthly_expenses, 
        net_profit=net_profit, 
        chart_data=chart_data, 
        summary=summary_data,
        budget_summary=budget_summary # <-- Added this
    )
# In app.py - Add this entire new function at the end

@app.route('/budget', methods=['GET', 'POST'])
@login_required
def budget():
    current_month = datetime.utcnow().month
    current_year = datetime.utcnow().year

    if request.method == 'POST':
        for key, value in request.form.items():
            if key.startswith('budget_'):
                category_id = int(key.split('_')[1])
                amount = float(value) if value and value.strip() else 0.0
                
                existing_budget = Budget.query.filter_by(
                    user_id=current_user.id,
                    category_id=category_id,
                    month=current_month,
                    year=current_year
                ).first()

                if existing_budget:
                    existing_budget.amount = amount
                else:
                    new_budget = Budget(
                        user_id=current_user.id,
                        category_id=category_id,
                        month=current_month,
                        year=current_year,
                        amount=amount
                    )
                    db.session.add(new_budget)
        
        db.session.commit()
        flash('Budgets updated successfully!', 'success')
        return redirect(url_for('budget'))

    # Logic for GET request
    # Note: For business users, this will show the same personal expense categories for budgeting.
    expense_categories = Category.query.filter_by(user_id=current_user.id, type='expense').all()
    
    # Determine whether to query Transaction or BusinessTransaction based on user role
    if current_user.role.strip().lower() == 'business':
        TransactionModel = BusinessTransaction
    else:
        TransactionModel = Transaction

    spent_data = db.session.query(
        TransactionModel.category_id,  # Assumes BusinessTransaction also has category_id
        func.sum(TransactionModel.amount)
    ).filter(
        TransactionModel.user_id == current_user.id,
        TransactionModel.type == 'expense',
        func.extract('month', TransactionModel.date) == current_month,
        func.extract('year', TransactionModel.date) == current_year
    ).group_by(TransactionModel.category_id).all()
    
    spent_map = dict(spent_data)

    budgets_data = []
    for cat in expense_categories:
        budget = Budget.query.filter_by(user_id=current_user.id, category_id=cat.id, month=current_month, year=current_year).first()
        spent = spent_map.get(cat.id, 0.0)
        budget_amount = budget.amount if budget else 0.0
        percentage = int((spent / budget_amount) * 100) if budget_amount > 0 else 0
        
        budgets_data.append({
            'category_id': cat.id,
            'category_name': cat.name,
            'budget_amount': budget_amount,
            'spent_amount': spent,
            'percentage': min(100, percentage)
        })

    return render_template('budget.html', budgets_data=budgets_data, month_name=datetime.utcnow().strftime('%B'))

@app.route('/business/transactions', methods=['GET', 'POST'])
@login_required
def business_transactions():
    if request.method == 'POST':
        try:
            # Accept both possible field names
            receipt_file = request.files.get('receipt_file') or request.files.get('receipt_file_for_save')
            receipt_url = None

            if receipt_file and allowed_file(receipt_file.filename):
                # Upload directly to Cloudinary
                upload_result = cloudinary.uploader.upload(
                    receipt_file,
                    folder=f"receipts/{current_user.id}"
                )
                receipt_url = upload_result.get("secure_url")

            desc = request.form.get('description')
            amount = float(request.form.get('amount'))
            trans_type = request.form.get('type')
            category_id = request.form.get('category_id')
            trans_date = datetime.strptime(request.form.get('date'), '%Y-%m-%d').date()

            if not desc or amount <= 0 or not category_id:
                flash('Valid description, category, and positive amount are required.', 'danger')
            else:
                new_trans = BusinessTransaction(
                    user_id=current_user.id,
                    description=desc,
                    amount=amount,
                    type=trans_type,
                    category_id=category_id,
                    date=trans_date,
                    receipt_filename=receipt_url  # ✅ Cloudinary URL only
                )
                db.session.add(new_trans)
                db.session.commit()
                flash('Business transaction added successfully!', 'success')

                # Retraining counter
                session['new_business_transactions_count'] = session.get('new_business_transactions_count', 0) + 1
                if session['new_business_transactions_count'] >= 20:
                    if retrain_business_model(current_user.id):
                        flash('AI model was automatically updated in the background with your new data.', 'info')
                    session['new_business_transactions_count'] = 0

                # Anomaly detection
                if check_transaction_anomaly(new_trans, current_user.id):
                    flash(
                        f"AI Alert: The transaction '{new_trans.description}' for ₹{new_trans.amount:.2f} is unusually high. Please review.",
                        'warning'
                    )

        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')
        return redirect(url_for('business_transactions'))

    # --- GET logic ---
    categories = Category.query.filter_by(user_id=current_user.id).order_by(Category.name).all()

    if not categories:
        default_categories = [
            ('Client Revenue', 'income'), ('Supplies', 'expense'), ('Rent', 'expense'),
            ('Utilities', 'expense'), ('Marketing', 'expense'), ('Other', 'expense')
        ]
        for cat_name, cat_type in default_categories:
            db.session.add(Category(name=cat_name, type=cat_type, user_id=current_user.id))
        db.session.commit()
        categories = Category.query.filter_by(user_id=current_user.id).order_by(Category.name).all()
        flash('Some default business categories have been created for you. You can manage them in the Categories page.', 'info')

    search_query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    query = BusinessTransaction.query.filter_by(user_id=current_user.id)
    if search_query:
        query = query.filter(BusinessTransaction.description.ilike(f'%{search_query}%'))

    transactions_pagination = query.order_by(BusinessTransaction.date.desc()).paginate(page=page, per_page=10, error_out=False)

    return render_template(
        'business/transactions.html',
        transactions_pagination=transactions_pagination,
        today=date.today().strftime('%Y-%m-%d'),
        categories=categories,
        search_query=search_query
    )



@app.route('/business/financials')
@login_required
def business_financials():
    # This route now calculates and displays data, no manual entry.
    today = date.today()
    start_of_month = today.replace(day=1)
    end_of_month = start_of_month + relativedelta(months=1)

    transactions_this_month = BusinessTransaction.query.filter(
        BusinessTransaction.user_id == current_user.id,
        BusinessTransaction.date >= start_of_month,
        BusinessTransaction.date < end_of_month
    ).all()

    revenue = sum(t.amount for t in transactions_this_month if t.type == 'revenue')
    expenses = sum(t.amount for t in transactions_this_month if t.type == 'expense')
    net_profit = revenue - expenses
    profit_margin = (net_profit / revenue) * 100 if revenue > 0 else 0
    
    # Update metrics for dashboard
    metrics = BusinessMetrics.query.filter_by(user_id=current_user.id).first()
    if not metrics:
        metrics = BusinessMetrics(user_id=current_user.id)
        db.session.add(metrics)
    
    metrics.monthly_revenue = revenue
    metrics.monthly_expenses = expenses
    metrics.net_profit = net_profit
    metrics.profit_margin = profit_margin
    db.session.commit()

    return render_template('business/financials.html', metrics=metrics)

@app.route('/business/investments', methods=['GET', 'POST'])
@login_required
def business_investments():
    if request.method == 'POST':
        try:
            name = request.form.get('investment_name')
            inv_type = request.form.get('investment_type')
            amount = float(request.form.get('amount_invested'))
            p_date = datetime.strptime(request.form.get('purchase_date'), '%Y-%m-%d').date()
            life_years_str = request.form.get('useful_life_years')
            life_years = int(life_years_str) if life_years_str else None

            if not name or not inv_type or amount <= 0:
                 flash('Valid name, type, and positive amount are required.', 'danger')
            else:
                new_inv = BusinessInvestment(user_id=current_user.id, investment_name=name, investment_type=inv_type, amount_invested=amount, purchase_date=p_date, useful_life_years=life_years)
                db.session.add(new_inv)
                db.session.commit()
                flash('Business investment added!', 'success')
        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')
        return redirect(url_for('business_investments'))

    investments = BusinessInvestment.query.filter_by(user_id=current_user.id).all()
    return render_template('business/investments.html', investments=investments, today=date.today().strftime('%Y-%m-%d'))


@app.route('/business/loans', methods=['GET', 'POST'])
@login_required
def business_loans():
    if request.method == 'POST':
        try:
            name = request.form.get('loan_name')
            principal = float(request.form.get('principal_amount'))
            rate = float(request.form.get('interest_rate'))
            tenure = int(request.form.get('tenure_months'))
            s_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()

            if not name or principal <= 0 or rate <= 0 or tenure <= 0:
                flash('All fields are required with positive numbers.', 'danger')
            else:
                # Calculate EMI
                r = (rate / 12) / 100
                emi = (principal * r * (1 + r)**tenure) / (((1 + r)**tenure) - 1)
                new_loan = BusinessLoan(user_id=current_user.id, loan_name=name, principal_amount=principal, interest_rate=rate, tenure_months=tenure, start_date=s_date, emi=emi)
                db.session.add(new_loan)
                db.session.commit()
                flash(f'Loan added with a calculated EMI of ₹{emi:.2f}', 'success')
        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')
        return redirect(url_for('business_loans'))

    loans = BusinessLoan.query.filter_by(user_id=current_user.id).all()
    return render_template('business/loans.html', loans=loans, today=date.today().strftime('%Y-%m-%d'))

# Keep the /business/clients route as is, since it doesn't require complex calculations yet.
# You can later link BusinessTransactions to clients for more detailed revenue tracking.
@app.route('/business/clients', methods=['GET', 'POST'])
@login_required
def business_clients():
    if request.method == 'POST':
        name = request.form.get('client_name')
        project = request.form.get('project')
        revenue = float(request.form.get('revenue_contribution'))
        status = request.form.get('status')
        new_client = BusinessClient(user_id=current_user.id, client_name=name, project=project, revenue_contribution=revenue, status=status)
        db.session.add(new_client)
        db.session.commit()
        flash(f'Client "{name}" added successfully!', 'success')
        return redirect(url_for('business_clients'))
    clients = BusinessClient.query.filter_by(user_id=current_user.id).all()
    return render_template('business/clients.html', clients=clients)

@app.route('/business/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
@login_required
def edit_business_transaction(transaction_id):
    trans = db.session.get(BusinessTransaction, transaction_id)
    if not trans or trans.user_id != current_user.id:
        flash('Transaction not found or unauthorized.', 'danger')
        return redirect(url_for('business_transactions'))
    
    if request.method == 'POST':
        try:
            trans.description = request.form.get('description')
            trans.amount = float(request.form.get('amount'))
            trans.type = request.form.get('type')
            trans.category_id = request.form.get('category_id')
            trans.date = datetime.strptime(request.form.get('date'), '%Y-%m-%d').date()
            if not trans.description or trans.amount <= 0:
                flash('Valid description and positive amount are required.', 'danger')
            else:
                db.session.commit()
                flash('Transaction updated successfully!', 'success')
                return redirect(url_for('business_transactions'))
        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')
    
    return render_template('business/edit_transaction.html', transaction=trans)

@app.route('/business/delete_transaction/<int:transaction_id>', methods=['POST'])
@login_required
def delete_business_transaction(transaction_id):
    trans = db.session.get(BusinessTransaction, transaction_id)
    if trans and trans.user_id == current_user.id:
        # Add logic here to delete the associated receipt file if it exists
        if trans.receipt_filename:
            try:
                # Receipt is stored on Cloudinary — extract public_id and delete
                if 'cloudinary' in (trans.receipt_filename or ''):
                    import cloudinary.uploader
                    # Extract public_id from Cloudinary URL
                    public_id = trans.receipt_filename.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                    cloudinary.uploader.destroy(f"receipts/{trans.user_id}/{public_id}")
                else:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], trans.receipt_filename))
            except Exception as e:
                print(f"Error deleting receipt {trans.receipt_filename}: {e}")

        db.session.delete(trans)
        db.session.commit()
        flash('Business transaction deleted.', 'success')
    else:
        flash('Transaction not found or unauthorized.', 'danger')
    return redirect(url_for('business_transactions'))

@app.route('/business/edit_investment/<int:investment_id>', methods=['GET', 'POST'])
@login_required
def edit_business_investment(investment_id):
    inv = db.session.get(BusinessInvestment, investment_id)
    if not inv or inv.user_id != current_user.id:
        flash('Investment not found or unauthorized.', 'danger')
        return redirect(url_for('business_investments'))

    if request.method == 'POST':
        try:
            inv.investment_name = request.form.get('investment_name')
            inv.investment_type = request.form.get('investment_type')
            inv.amount_invested = float(request.form.get('amount_invested'))
            inv.purchase_date = datetime.strptime(request.form.get('purchase_date'), '%Y-%m-%d').date()
            life_years_str = request.form.get('useful_life_years')
            inv.useful_life_years = int(life_years_str) if life_years_str else None
            db.session.commit()
            flash('Investment updated successfully!', 'success')
            return redirect(url_for('business_investments'))
        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')

    return render_template('business/edit_investment.html', investment=inv)

@app.route('/business/delete_investment/<int:investment_id>', methods=['POST'])
@login_required
def delete_business_investment(investment_id):
    inv = db.session.get(BusinessInvestment, investment_id)
    if inv and inv.user_id == current_user.id:
        db.session.delete(inv)
        db.session.commit()
        flash('Business investment deleted.', 'success')
    else:
        flash('Investment not found or unauthorized.', 'danger')
    return redirect(url_for('business_investments'))

@app.route('/business/edit_loan/<int:loan_id>', methods=['GET', 'POST'])
@login_required
def edit_business_loan(loan_id):
    loan = db.session.get(BusinessLoan, loan_id)
    if not loan or loan.user_id != current_user.id:
        flash('Loan not found or unauthorized.', 'danger')
        return redirect(url_for('business_loans'))

    if request.method == 'POST':
        try:
            loan.loan_name = request.form.get('loan_name')
            loan.principal_amount = float(request.form.get('principal_amount'))
            loan.interest_rate = float(request.form.get('interest_rate'))
            loan.tenure_months = int(request.form.get('tenure_months'))
            loan.start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
            
            # Recalculate EMI
            r = (loan.interest_rate / 12) / 100
            tenure = loan.tenure_months
            loan.emi = (loan.principal_amount * r * (1 + r)**tenure) / (((1 + r)**tenure) - 1)
            
            db.session.commit()
            flash(f'Loan updated successfully! New EMI is ₹{loan.emi:.2f}', 'success')
            return redirect(url_for('business_loans'))
        except (ValueError, TypeError):
            flash('Invalid data provided.', 'danger')
            
    return render_template('business/edit_loan.html', loan=loan)

@app.route('/business/delete_loan/<int:loan_id>', methods=['POST'])
@login_required
def delete_business_loan(loan_id):
    loan = db.session.get(BusinessLoan, loan_id)
    if loan and loan.user_id == current_user.id:
        db.session.delete(loan)
        db.session.commit()
        flash('Business loan deleted.', 'success')
    else:
        flash('Loan not found or unauthorized.', 'danger')
    return redirect(url_for('business_loans'))

def retrain_business_model(user_id):
    """
    A helper function to retrain and save the business category prediction model.
    Returns True on success, False on failure.
    """
    user = db.session.get(User, user_id)
    transactions = user.business_transactions # Assumes backref is set up
    
    if len(transactions) < 15:
        # Not enough data to train, but not an error.
        return False

    df = pd.DataFrame([(t.description, t.category.name) for t in transactions], columns=['description', 'category'])
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['description'], df['category'])
    joblib.dump(model, f'user_{user_id}_business_category_model.pkl')
    return True

@app.route('/train_business_model')
@login_required
def train_business_model():
    # This route now just calls the helper function.
    if retrain_business_model(current_user.id):
        flash('Successfully trained the AI category model on your business data!', 'success')
    else:
        flash('You need at least 15 business transactions to train the AI category model.', 'info')
    return redirect(url_for('business_transactions'))

@app.route('/predict_business_category', methods=['POST'])
@login_required
def predict_business_category():
    description = request.json['description']
    model_path = f'user_{current_user.id}_business_category_model.pkl'

    try:
        # Load the user's trained model
        model = joblib.load(model_path)
        prediction_name = model.predict([description])[0]
        
        # Find the category ID for the predicted name
        predicted_category = Category.query.filter_by(user_id=current_user.id, name=prediction_name).first()
        
        if predicted_category:
            return jsonify({'category_id': predicted_category.id})
        else:
            return jsonify({'category_id': None}) # Predicted category name doesn't exist
            
    except FileNotFoundError:
        return jsonify({'category_id': None}) # Model hasn't been trained yet
# app.py (Add this new route at the end of the file)

@app.route('/predict_business_cashflow')
@login_required
def predict_business_cashflow():
    try:
        # Fetch all business transactions for the user
        transactions = BusinessTransaction.query.filter_by(user_id=current_user.id).order_by(BusinessTransaction.date.asc()).all()

        if not transactions or len(transactions) < 2:
            return jsonify({"error": "Not enough data for a forecast."})

        # Prepare DataFrame
        df = pd.DataFrame([{
            'date': t.date,
            'amount': t.amount if t.type == 'revenue' else -t.amount
        } for t in transactions])

        # Group by date to get daily net cash flow
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.groupby('date').sum().reset_index().dropna(subset=['date'])

        if len(df) < 2:
            return jsonify({"error": "Not enough distinct data points for a forecast."})

        # --- RandomForest Forecast (lightweight, free-tier compatible) ---
        df['days'] = (df['date'] - df['date'].min()).dt.days
        X = df[['days']]
        y = df['amount']
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        last_day = df['days'].max()
        future_days_arr = np.arange(last_day + 1, last_day + 61).reshape(-1, 1)
        future_predictions = rf.predict(future_days_arr).tolist()

        # Generate future dates
        base_date = df['date'].min()
        future_dates = [(base_date + pd.Timedelta(days=int(d))).strftime('%Y-%m-%d')
                        for d in np.arange(last_day + 1, last_day + 61)]

        response_data = {
            'dates': future_dates,
            'predicted_flow': future_predictions,
            'lower_bound': [v * 0.85 for v in future_predictions],
            'upper_bound': [v * 1.15 for v in future_predictions]
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

    
# --- BUSINESS REPORTING HELPERS ---

def generate_business_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 10, 'Profit & Loss Statement', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"For the period: {data['start_date']} to {data['end_date']}", 0, 1, 'C')
    pdf.ln(10)

    # Summary Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Financial Summary', 0, 1)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(95, 10, 'Total Revenue:', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['total_revenue']:.2f}", 1, 1, 'R')
    pdf.cell(95, 10, 'Total Expenses:', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['total_expenses']:.2f}", 1, 1, 'R')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(95, 10, 'Net Profit / (Loss):', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['net_profit']:.2f}", 1, 1, 'R')
    pdf.ln(10)

    # Expense Breakdown
    if data['expenses_by_category']:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Expense Breakdown by Category', 0, 1)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(95, 10, 'Category', 1, 0)
        pdf.cell(95, 10, 'Amount', 1, 1, 'C')
        pdf.set_font('Helvetica', '', 12)
        for category, amount in data['expenses_by_category'].items():
            pdf.cell(95, 10, f"  {category}", 1, 0)
            pdf.cell(95, 10, f"Rs. {amount:.2f}", 1, 1, 'R')

    # FPDF.output returns bytes, encode to latin-1 for Flask Response
    return pdf.output(dest='S').encode('latin-1')

def generate_business_csv_report(data):
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write summary
    writer.writerow(['Profit & Loss Statement'])
    writer.writerow(['Period', f"{data['start_date']} to {data['end_date']}"])
    writer.writerow([]) # Spacer
    writer.writerow(['Metric', 'Amount (Rs.)'])
    writer.writerow(['Total Revenue', data['total_revenue']])
    writer.writerow(['Total Expenses', data['total_expenses']])
    writer.writerow(['Net Profit / (Loss)', data['net_profit']])
    writer.writerow([]) # Spacer
    
    # Write expense breakdown
    writer.writerow(['Expense Breakdown by Category'])
    writer.writerow(['Category', 'Amount (Rs.)'])
    for category, amount in data['expenses_by_category'].items():
        writer.writerow([category, amount])
        
    return output.getvalue()

@app.route('/business/reports', methods=['GET', 'POST'])
@login_required
def business_reports():
    if request.method == 'POST':
        try:
            start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
            end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d').date()
            report_format = request.form.get('format')

            if start_date > end_date:
                flash('Start date cannot be after end date.', 'danger')
                return redirect(url_for('business_reports'))
            
            # Query transactions in the date range
            transactions = BusinessTransaction.query.filter(
                BusinessTransaction.user_id == current_user.id,
                BusinessTransaction.date.between(start_date, end_date)
            ).all()

            # Calculate metrics
            total_revenue = sum(t.amount for t in transactions if t.type == 'revenue')
            total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
            
            expenses_by_category = {}
            for t in transactions:
                if t.type == 'expense':
                    expenses_by_category[t.category.name] = expenses_by_category.get(t.category.name, 0) + t.amount

            report_data = {
                'start_date': start_date.strftime('%d %b %Y'),
                'end_date': end_date.strftime('%d %b %Y'),
                'total_revenue': total_revenue,
                'total_expenses': total_expenses,
                'net_profit': total_revenue - total_expenses,
                'expenses_by_category': expenses_by_category
            }

            if report_format == 'pdf':
                pdf_data = generate_business_pdf_report(report_data)
                return Response(pdf_data, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=business_report.pdf'})
            elif report_format == 'csv':
                csv_data = generate_business_csv_report(report_data)
                return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=business_report.csv'})

        except Exception as e:
            flash(f'An error occurred while generating the report: {e}', 'danger')
    
    return render_template('business/reports.html', today=date.today().strftime('%Y-%m-%d'))

@app.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    description = request.form.get('description')
    amount = float(request.form.get('amount'))
    ttype = request.form.get('type')
    # Get category_id from the form
    category_id = request.form.get('category_id') 
    
    # Create some default categories if the user has none
    if not current_user.categories:
        default_categories = [
            ('Food', 'expense'), ('Transport', 'expense'), ('Utilities', 'expense'),
            ('Salary', 'income'), ('Other', 'expense')
        ]
        for cat_name, cat_type in default_categories:
            db.session.add(Category(name=cat_name, type=cat_type, user_id=current_user.id))
        db.session.commit()

    # If no category is selected, try to assign a default
    if not category_id:
        uncategorized = Category.query.filter_by(user_id=current_user.id, name='Other').first()
        if not uncategorized:
            uncategorized = Category(name='Other', user_id=current_user.id)
            db.session.add(uncategorized)
            db.session.commit()
        category_id = uncategorized.id

    new_transaction = Transaction(description=description, amount=amount, type=ttype, date=datetime.utcnow().date(), user_id=current_user.id, category_id=category_id)
    db.session.add(new_transaction)
    db.session.commit()
    flash('Transaction added successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/transactions')
@login_required
def view_transactions():
    # --- NEW: Search and Pagination Logic ---
    search_query = request.args.get('q', '')  # Get search query from URL
    page = request.args.get('page', 1, type=int) # Get page number, default to 1

    # Base query
    query = Transaction.query.filter_by(user_id=current_user.id)

    # If there's a search query, filter the results by description
    if search_query:
        query = query.filter(Transaction.description.ilike(f'%{search_query}%'))
    
    # Order by date and paginate the results (e.g., 10 items per page)
    transactions_pagination = query.order_by(Transaction.date.desc()).paginate(page=page, per_page=10, error_out=False)

    return render_template(
        'transactions.html', 
        transactions_pagination=transactions_pagination,
        search_query=search_query
    )

@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
@login_required
def delete_transaction(transaction_id):
    transaction = db.session.get(Transaction, transaction_id)
    if not transaction or transaction.user_id != current_user.id: flash('Not authorized.', 'error'); return redirect(url_for('view_transactions'))
    db.session.delete(transaction); db.session.commit()
    flash('Transaction deleted.', 'success'); return redirect(url_for('view_transactions'))

@app.route('/schemes')
@login_required
def schemes():
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    schemes_with_details = []
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed < 0: years_elapsed = 0
        tenure_years = scheme.tenure_months / 12
        maturity_amount = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** tenure_years)
        current_value = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed)
        penalized_rate = scheme.interest_rate - scheme.penalty_rate
        if penalized_rate < 0: penalized_rate = 0
        early_withdrawal_value = scheme.principal_amount * ((1 + (penalized_rate / 100)) ** years_elapsed)
        maturity_date = scheme.start_date + relativedelta(months=+scheme.tenure_months)
        schemes_with_details.append({'scheme': scheme, 'maturity_date': maturity_date, 'maturity_amount': maturity_amount, 'current_value': current_value, 'early_withdrawal_value': early_withdrawal_value})
    return render_template('schemes.html', schemes_data=schemes_with_details)

@app.route('/add_scheme', methods=['POST'])
@login_required
def add_scheme():
    scheme_name = request.form.get('scheme_name'); principal = float(request.form.get('principal_amount')); rate = float(request.form.get('interest_rate')); tenure = int(request.form.get('tenure_months')); start_date_str = request.form.get('start_date'); penalty = float(request.form.get('penalty_rate'))
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    new_scheme = FixedScheme(scheme_name=scheme_name, principal_amount=principal, interest_rate=rate, tenure_months=tenure, start_date=start_date, penalty_rate=penalty, user_id=current_user.id)
    db.session.add(new_scheme); db.session.commit()
    flash(f'Scheme "{scheme_name}" added successfully!', 'success'); return redirect(url_for('schemes'))

@app.route('/delete_scheme/<int:scheme_id>', methods=['POST'])
@login_required
def delete_scheme(scheme_id):
    scheme = db.session.get(FixedScheme, scheme_id)
    if not scheme or scheme.user_id != current_user.id: flash('Not authorized.', 'error'); return redirect(url_for('schemes'))
    db.session.delete(scheme); db.session.commit()
    flash('Scheme deleted.', 'success'); return redirect(url_for('schemes'))

@app.route('/salary_manager', methods=['GET', 'POST'])
@login_required
def salary_manager():
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    if request.method == 'POST':
        monthly_gross = float(request.form.get('monthly_gross'))
        deductions_80c = float(request.form.get('deductions_80c'))
        hra_exemption = float(request.form.get('hra_exemption'))
        if salary_details:
            salary_details.monthly_gross = monthly_gross; salary_details.deductions_80c = deductions_80c; salary_details.hra_exemption = hra_exemption
            flash('Salary details updated successfully!', 'success')
        else:
            salary_details = Salary(monthly_gross=monthly_gross, deductions_80c=deductions_80c, hra_exemption=hra_exemption, user_id=current_user.id)
            db.session.add(salary_details)
            flash('Salary details saved successfully!', 'success')
        db.session.commit()
        return redirect(url_for('salary_manager'))
    return render_template('salary_manager.html', salary=salary_details)

@app.route('/investments')
@login_required
def investments():
    sold_investments = SoldInvestment.query.filter_by(user_id=current_user.id).order_by(SoldInvestment.sell_date.desc()).all()
    return render_template('investments.html', sales=sold_investments)

@app.route('/add_investment', methods=['POST'])
@login_required
def add_investment():
    asset_type = request.form.get('asset_type')
    ticker = request.form.get('ticker_symbol').lower() if asset_type == 'Crypto' else request.form.get('ticker_symbol').upper()
    quantity = float(request.form.get('quantity'))
    price = float(request.form.get('purchase_price'))
    currency = request.form.get('purchase_currency')
    purchase_date_str = request.form.get('purchase_date')
    purchase_date = datetime.strptime(purchase_date_str, '%Y-%m-%d').date()
    new_investment = Investment(asset_type=asset_type, ticker_symbol=ticker, quantity=quantity, purchase_price=price, purchase_currency=currency, purchase_date=purchase_date, user_id=current_user.id)
    db.session.add(new_investment); db.session.commit()
    flash(f'{asset_type} "{ticker}" added to your portfolio!', 'success'); return redirect(url_for('investments'))

@app.route('/delete_investment/<int:investment_id>', methods=['POST'])
@login_required
def delete_investment(investment_id):
    investment = db.session.get(Investment, investment_id)
    if not investment or investment.user_id != current_user.id:
        flash('Not authorized to delete this investment.', 'error'); return redirect(url_for('investments'))
    db.session.delete(investment); db.session.commit()
    flash('Investment removed from portfolio.', 'success'); return redirect(url_for('investments'))

@app.route('/refresh_prices')
@login_required
def refresh_prices():
    user_investments = Investment.query.filter_by(user_id=current_user.id).all()
    refreshed_data = []
    cg = CoinGeckoAPI()
    
    # Set a default fallback exchange rate first
    usd_to_inr_rate = 83.5 
    try:
        # Fetch the latest USD to INR exchange rate from CoinGecko
        rates = cg.get_price(ids='tether', vs_currencies='inr')
        if rates and 'tether' in rates and 'inr' in rates['tether']:
            usd_to_inr_rate = rates['tether']['inr']
    except Exception as e:
        print(f"Could not fetch exchange rate, using fallback rate of {usd_to_inr_rate}. Error: {e}")

    for investment in user_investments:
        current_price_display = 0
        total_value_inr = 0
        profit_loss_display = 0
        
        try:
            if investment.asset_type == 'Stock':
                stock = yf.Ticker(investment.ticker_symbol)
                todays_data = stock.history(period='1d')
                if not todays_data.empty:
                    current_price_display = todays_data['Close'].iloc[-1]
                    total_value_inr = investment.quantity * current_price_display
                    investment_cost_inr = investment.purchase_price * investment.quantity
                    profit_loss_display = total_value_inr - investment_cost_inr
            
            # --- CORRECTED CRYPTO SECTION START ---
            elif investment.asset_type == 'Crypto':
                crypto_id = get_coingecko_id(investment.ticker_symbol)
                price_data = cg.get_price(ids=crypto_id, vs_currencies='usd')
                if price_data and price_data.get(crypto_id):
                    current_price_usd = price_data[crypto_id].get('usd', 0)
                    
                    # --- All final calculations are now consistently in INR ---
                    
                    # 1. Calculate the Total CURRENT Value in INR
                    total_value_inr = (investment.quantity * current_price_usd) * usd_to_inr_rate

                    # 2. Calculate the original Investment COST in INR
                    investment_cost_inr = 0
                    if investment.purchase_currency == 'INR':
                        investment_cost_inr = investment.purchase_price * investment.quantity
                    else:  # Assuming the purchase currency was 'USD'
                        investment_cost_inr = (investment.purchase_price * investment.quantity) * usd_to_inr_rate

                    # 3. Calculate the final Profit or Loss in INR
                    profit_loss_display = total_value_inr - investment_cost_inr
                    
                    # We still display the current price in USD for user reference
                    current_price_display = current_price_usd
            # --- CORRECTED CRYPTO SECTION END ---

        except Exception as e:
            print(f"Could not fetch price for {investment.ticker_symbol}: {e}")
            
        refreshed_data.append({
            'investment': {
                'id': investment.id,
                'ticker_symbol': investment.ticker_symbol.upper(),
                'asset_type': investment.asset_type,
                'quantity': investment.quantity,
                'purchase_price': investment.purchase_price,
                'purchase_currency': investment.purchase_currency
            },
            'current_price_display': current_price_display,
            'total_value_inr': total_value_inr,
            'profit_loss_display': profit_loss_display
        })
        
    return jsonify({'data': refreshed_data, 'exchange_rate': usd_to_inr_rate})

@app.route('/net_worth')
@login_required
def net_worth():
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    cash_balance = sum(t.amount for t in transactions if t.type == 'income') - sum(t.amount for t in transactions if t.type == 'expense')
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    total_schemes_value = 0
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed > 0:
            total_schemes_value += scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed)
    user_investments = Investment.query.filter_by(user_id=current_user.id).all()
    total_investments_value = 0
    cg = CoinGeckoAPI()
    try:
        rates = cg.get_price(ids='tether', vs_currencies='inr')
        usd_to_inr_rate = rates['tether']['inr']
    except Exception:
        usd_to_inr_rate = 83.5
    for investment in user_investments:
        try:
            if investment.asset_type == 'Stock':
                stock = yf.Ticker(investment.ticker_symbol)
                todays_data = stock.history(period='1d')
                if not todays_data.empty:
                    total_investments_value += investment.quantity * todays_data['Close'].iloc[-1]
            elif investment.asset_type == 'Crypto':
                crypto_id = get_coingecko_id(investment.ticker_symbol)
                price_data = cg.get_price(ids=crypto_id, vs_currencies='usd')
                if price_data and price_data.get(crypto_id):
                    current_price_usd = price_data[crypto_id].get('usd', 0)
                    total_investments_value += (investment.quantity * current_price_usd) * usd_to_inr_rate
        except Exception as e:
            print(f"Net Worth: Could not fetch price for {investment.ticker_symbol}: {e}")
    total_assets = cash_balance + total_schemes_value + total_investments_value
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    total_liabilities = 0
    loans_with_details = []
    for loan in user_loans:
        r = (loan.interest_rate / 12) / 100
        n = loan.tenure_months
        payments_made = (date.today().year - loan.start_date.year) * 12 + (date.today().month - loan.start_date.month)
        if payments_made < 0: payments_made = 0
        if payments_made > n: payments_made = n
        outstanding_balance = loan.principal * (((1 + r)**n) - ((1 + r)**payments_made)) / (((1 + r)**n) - 1) if (((1 + r)**n) - 1) != 0 else 0
        total_liabilities += outstanding_balance
        loans_with_details.append({'loan_name': loan.loan_name, 'outstanding': outstanding_balance})
    total_net_worth = total_assets - total_liabilities
    chart_data = {'labels': ['Cash', 'Fixed Schemes', 'Investments'], 'data': [cash_balance, total_schemes_value, total_investments_value]}
    return render_template('net_worth.html', 
                           net_worth=total_net_worth, 
                           assets=total_assets,
                           liabilities=total_liabilities,
                           cash=cash_balance, 
                           schemes=total_schemes_value, 
                           investments=total_investments_value,
                           loans=loans_with_details,
                           chart_data=json.dumps(chart_data))

@app.route('/tax_estimator')
@login_required
def tax_estimator():
    age = 0
    if current_user.dob:
        today = date.today()
        age = today.year - current_user.dob.year - ((today.month, today.day) < (current_user.dob.month, current_user.dob.day))
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    gross_salary_income = 0
    deductions = 0
    if salary_details:
        gross_salary_income = salary_details.monthly_gross * 12
        deductions = salary_details.deductions_80c + salary_details.hra_exemption
    user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
    total_interest_income = 0
    for scheme in user_schemes:
        years_elapsed = (date.today() - scheme.start_date).days / 365.25
        if years_elapsed > 0:
            interest = scheme.principal_amount * ((1 + (scheme.interest_rate / 100)) ** years_elapsed) - scheme.principal_amount
            total_interest_income += interest
    sold_investments = SoldInvestment.query.filter_by(user_id=current_user.id).all()
    stcg_stocks = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Stock' and s.gain_type == 'STCG')
    ltcg_stocks = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Stock' and s.gain_type == 'LTCG')
    crypto_gains = sum(s.capital_gain for s in sold_investments if s.asset_type == 'Crypto')
    stcg_tax = stcg_stocks * 0.15
    ltcg_taxable = max(0, ltcg_stocks - 100000)
    ltcg_tax = ltcg_taxable * 0.10
    crypto_tax = crypto_gains * 0.30
    total_capital_gains_tax = stcg_tax + ltcg_tax + crypto_tax
    total_regular_income = gross_salary_income + total_interest_income
    new_regime_details = calculate_new_regime_tax(total_regular_income)
    old_regime_details = calculate_old_regime_tax(total_regular_income, deductions, age)
    new_regime_details['total_tax'] += total_capital_gains_tax
    old_regime_details['total_tax'] += total_capital_gains_tax
    capital_gains_summary = {
        'stcg_stocks': stcg_stocks, 'stcg_tax': stcg_tax,
        'ltcg_stocks': ltcg_stocks, 'ltcg_tax': ltcg_tax,
        'crypto_gains': crypto_gains, 'crypto_tax': crypto_tax,
        'total_tax': total_capital_gains_tax
    }
    return render_template('tax_estimator.html', 
                           new_regime=new_regime_details, 
                           old_regime=old_regime_details, 
                           salary_setup=(salary_details is not None), 
                           user_age=age,
                           capital_gains=capital_gains_summary,
                           interest_income=total_interest_income)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        dob_str = request.form.get('dob')
        if dob_str:
            current_user.dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
            db.session.commit()
            flash('Your profile has been updated.', 'success')
            return redirect(url_for('profile'))
    return render_template('profile.html', user=current_user)

@app.route('/sell_investment/<int:investment_id>', methods=['POST'])
@login_required
def sell_investment(investment_id):
    investment = db.session.get(Investment, investment_id)
    if not investment or investment.user_id != current_user.id:
        flash('Investment not found or not authorized.', 'error')
        return redirect(url_for('investments'))

    sell_price = float(request.form.get('sell_price'))
    sell_quantity = float(request.form.get('sell_quantity'))
    sell_date = datetime.strptime(request.form.get('sell_date'), '%Y-%m-%d').date()

    if sell_quantity > investment.quantity or sell_quantity <= 0:
        flash('Invalid quantity to sell.', 'error')
        return redirect(url_for('investments'))

    holding_period_days = (sell_date - investment.purchase_date).days
    
    gain_type = 'STCG'
    if investment.asset_type == 'Stock' and holding_period_days > 365:
        gain_type = 'LTCG'

    purchase_cost = investment.purchase_price * sell_quantity
    sell_value = sell_price * sell_quantity
    capital_gain = sell_value - purchase_cost
    
    new_sale = SoldInvestment(
        asset_type=investment.asset_type,
        ticker_symbol=investment.ticker_symbol,
        quantity=sell_quantity,
        purchase_price=investment.purchase_price,
        purchase_date=investment.purchase_date,
        sell_price=sell_price,
        sell_date=sell_date,
        capital_gain=capital_gain,
        gain_type=gain_type,
        user_id=current_user.id
    )
    db.session.add(new_sale)

    investment.quantity -= sell_quantity
    if investment.quantity <= 0.000001:
        db.session.delete(investment)
    
    db.session.commit()
    flash(f'Successfully sold {sell_quantity} units of {investment.ticker_symbol.upper()}.', 'success')
    return redirect(url_for('investments'))

@app.route('/loans')
@login_required
def loans():
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    return render_template('loans.html', loans=user_loans)

@app.route('/add_loan', methods=['POST'])
@login_required
def add_loan():
    loan_name = request.form.get('loan_name')
    principal = float(request.form.get('principal'))
    rate = float(request.form.get('interest_rate'))
    tenure = int(request.form.get('tenure_months'))
    start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()

    r = (rate / 12) / 100
    emi = (principal * r * (1 + r)**tenure) / ((1 + r)**tenure - 1)

    new_loan = Loan(
        loan_name=loan_name,
        principal=principal,
        interest_rate=rate,
        tenure_months=tenure,
        emi_amount=emi,
        start_date=start_date,
        user_id=current_user.id
    )
    db.session.add(new_loan)
    db.session.commit()
    flash(f'Loan "{loan_name}" added successfully with a calculated EMI of ₹{emi:.2f}.', 'success')
    return redirect(url_for('loans'))

@app.route('/delete_loan/<int:loan_id>', methods=['POST'])
@login_required
def delete_loan(loan_id):
    loan = db.session.get(Loan, loan_id)
    if not loan or loan.user_id != current_user.id:
        flash('Loan not found or not authorized.', 'error')
        return redirect(url_for('loans'))
    db.session.delete(loan)
    db.session.commit()
    flash('Loan deleted successfully.', 'success')
    return redirect(url_for('loans'))

@app.route('/sold_investments')
@login_required
def sold_investments():
    sales = SoldInvestment.query.filter_by(user_id=current_user.id).order_by(SoldInvestment.sell_date.desc()).all()
    return render_template('sold_investments.html', sales=sales)

@app.route('/edit_transaction/<int:transaction_id>', methods=['GET', 'POST'])
@login_required
def edit_transaction(transaction_id):
    transaction = db.session.get(Transaction, transaction_id)
    if not transaction or transaction.user_id != current_user.id:
        flash('Transaction not found or not authorized.', 'error')
        return redirect(url_for('view_transactions'))
    
    if request.method == 'POST':
        transaction.description = request.form.get('description')
        transaction.amount = float(request.form.get('amount'))
        transaction.type = request.form.get('type')
        transaction.category_id = request.form.get('category_id')
        transaction.date = datetime.strptime(request.form.get('date'), '%Y-%m-%d').date()
        db.session.commit()
        flash('Transaction updated successfully!', 'success')
        return redirect(url_for('view_transactions'))
        
    return render_template('edit_transaction.html', transaction=transaction)

@app.route('/edit_scheme/<int:scheme_id>', methods=['GET', 'POST'])
@login_required
def edit_scheme(scheme_id):
    scheme = db.session.get(FixedScheme, scheme_id)
    if not scheme or scheme.user_id != current_user.id:
        flash('Scheme not found or not authorized.', 'error')
        return redirect(url_for('schemes'))

    if request.method == 'POST':
        scheme.scheme_name = request.form.get('scheme_name')
        scheme.principal_amount = float(request.form.get('principal_amount'))
        scheme.interest_rate = float(request.form.get('interest_rate'))
        scheme.tenure_months = int(request.form.get('tenure_months'))
        scheme.start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
        scheme.penalty_rate = float(request.form.get('penalty_rate'))
        db.session.commit()
        flash('Scheme updated successfully!', 'success')
        return redirect(url_for('schemes'))

    return render_template('edit_scheme.html', scheme=scheme)

@app.route('/edit_loan/<int:loan_id>', methods=['GET', 'POST'])
@login_required
def edit_loan(loan_id):
    loan = db.session.get(Loan, loan_id)
    if not loan or loan.user_id != current_user.id:
        flash('Loan not found or not authorized.', 'error')
        return redirect(url_for('loans'))

    if request.method == 'POST':
        loan.loan_name = request.form.get('loan_name')
        loan.principal = float(request.form.get('principal'))
        loan.interest_rate = float(request.form.get('interest_rate'))
        loan.tenure_months = int(request.form.get('tenure_months'))
        loan.start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
        
        # Recalculate EMI
        r = (loan.interest_rate / 12) / 100
        tenure = loan.tenure_months
        loan.emi_amount = (loan.principal * r * (1 + r)**tenure) / ((1 + r)**tenure - 1)
        
        db.session.commit()
        flash('Loan updated successfully!', 'success')
        return redirect(url_for('loans'))

    return render_template('edit_loan.html', loan=loan)

@app.route('/ai_insights')
@login_required
def ai_insights():
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.asc()).all()
    if not transactions:
        return jsonify({'insights': ["No data available to generate insights."]})

    df = pd.DataFrame(
        [(t.date, t.amount if t.type == 'income' else -t.amount, t.category) for t in transactions],
        columns=['date', 'amount', 'category']
    )

    insights = []

    # --- 1. Anomaly Detection (Expenses Only) ---
    try:
        expense_data = df[df['amount'] < 0]
        if len(expense_data) > 10:
            model = IsolationForest(contamination=0.15, random_state=42)
            expense_data['score'] = model.fit_predict(expense_data[['amount']])
            anomalies = expense_data[expense_data['score'] == -1]
            if not anomalies.empty:
                latest_anomaly = anomalies.iloc[-1]
                insights.append(
                    f"Alert: Unusual spending detected on {latest_anomaly['date'].strftime('%d %b')} "
                    f"in {latest_anomaly['category']} (₹{abs(latest_anomaly['amount']):.2f})."
                )
    except Exception as e:
        insights.append("Anomaly detection could not be performed.")

    # --- 2. Savings Goal Predictor ---
    try:
        balance = df['amount'].sum()
        target = 100000  # example savings goal
        daily_avg = df['amount'].mean()
        if daily_avg > 0:
            days_needed = (target - balance) / daily_avg
            if days_needed > 0:
                reach_date = datetime.now() + timedelta(days=int(days_needed))
                insights.append(f"At current rate, you may reach ₹{target:,.0f} savings by {reach_date.strftime('%d %b %Y')}.")
    except:
        insights.append("Could not predict savings goal timeline.")

    # --- 3. Category-wise Spending Forecast ---
    try:
        expense_categories = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()

        # Fetch EMI total from Loan model (correct field name: emi_amount)
        from models import Loan
        loans = Loan.query.filter_by(user_id=current_user.id).all()
        emi_total = sum([loan.emi_amount for loan in loans])
        if emi_total > 0:
            expense_categories['EMI'] = emi_total

        if not expense_categories.empty:
            top_category = expense_categories.sort_values(ascending=False).index[0]
            top_value = expense_categories.max()
            insights.append(f"Next month's largest expense is likely to be '{top_category}' at around ₹{top_value:.2f}.")
    except Exception as e:
        insights.append("Spending forecast unavailable.")

    # --- 4. Financial Health Indicator ---
    try:
        income_total = df[df['amount'] > 0]['amount'].sum()
        expense_total = abs(df[df['amount'] < 0]['amount'].sum())
        expense_ratio = expense_total / (income_total + 1e-5)
        if expense_ratio < 0.5:
            health = "Good"
        elif expense_ratio < 0.8:
            health = "Moderate"
        else:
            health = "Risky"
        insights.append(f"AI Financial Health: {health}.")
    except:
        insights.append("Could not assess financial health.")

    return jsonify({'insights': insights})

def generate_personal_pdf_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 10, 'Income & Expense Report', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"For the period: {data['start_date']} to {data['end_date']}", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Summary', 0, 1)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(95, 10, 'Total Income:', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['total_income']:.2f}", 1, 1, 'R')
    pdf.cell(95, 10, 'Total Expense:', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['total_expense']:.2f}", 1, 1, 'R')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(95, 10, 'Net Savings:', 1, 0)
    pdf.cell(95, 10, f"Rs. {data['net_savings']:.2f}", 1, 1, 'R')
    pdf.ln(10)
    if data['expenses_by_category']:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Expense Breakdown by Category', 0, 1)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(95, 10, 'Category', 1, 0)
        pdf.cell(95, 10, 'Amount', 1, 1, 'C')
        pdf.set_font('Helvetica', '', 12)
        for category, amount in data['expenses_by_category'].items():
            pdf.cell(95, 10, f"  {category}", 1, 0)
            pdf.cell(95, 10, f"Rs. {amount:.2f}", 1, 1, 'R')
    return pdf.output(dest='S').encode('latin-1')

def generate_personal_csv_report(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Income & Expense Report'])
    writer.writerow(['Period', f"{data['start_date']} to {data['end_date']}"])
    writer.writerow([]); writer.writerow(['Metric', 'Amount (Rs.)'])
    writer.writerow(['Total Income', data['total_income']])
    writer.writerow(['Total Expense', data['total_expense']])
    writer.writerow(['Net Savings', data['net_savings']])
    writer.writerow([]); writer.writerow(['Expense Breakdown by Category'])
    writer.writerow(['Category', 'Amount (Rs.)'])
    for category, amount in data['expenses_by_category'].items():
        writer.writerow([category, amount])
    return output.getvalue()


# app.py (REPLACE your old /reports route)
@app.route('/reports', methods=['GET', 'POST'])
@login_required
def reports():
    if request.method == 'POST':
        try:
            start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d').date()
            end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d').date()
            report_format = request.form.get('format')

            transactions = Transaction.query.filter(
                Transaction.user_id == current_user.id,
                Transaction.date.between(start_date, end_date)
            ).all()

            total_income = sum(t.amount for t in transactions if t.type == 'income')
            total_expense = sum(t.amount for t in transactions if t.type == 'expense')
            expenses_by_category = {}
            for t in transactions:
                if t.type == 'expense':
                    expenses_by_category[t.category.name] = expenses_by_category.get(t.category.name, 0) + t.amount

            report_data = {
                'start_date': start_date.strftime('%d %b %Y'), 'end_date': end_date.strftime('%d %b %Y'),
                'total_income': total_income, 'total_expense': total_expense,
                'net_savings': total_income - total_expense, 'expenses_by_category': expenses_by_category
            }

            if report_format == 'pdf':
                pdf_data = generate_personal_pdf_report(report_data)
                return Response(pdf_data, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=personal_report.pdf'})
            elif report_format == 'csv':
                csv_data = generate_personal_csv_report(report_data)
                return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=personal_report.csv'})

        except Exception as e:
            flash(f'An error occurred: {e}', 'danger')

    return render_template('reports.html', today=date.today().strftime('%Y-%m-%d'))

# --- AI/ML Routes ---

@app.route('/train_model')
@login_required
def train_model():
    transactions = Transaction.query.filter_by(user_id=current_user.id, type='expense').all()
    if len(transactions) < 10: # Need enough data to train
        return jsonify({'status': 'not_enough_data'})

    df = pd.DataFrame([(t.description, t.category) for t in transactions], columns=['description', 'category'])
    
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['description'], df['category'])
    
    # Save the trained model for the user
    joblib.dump(model, f'user_{current_user.id}_category_model.pkl')
    
    return jsonify({'status': 'success'})

@app.route('/predict_category', methods=['POST'])
@login_required
def predict_category():
    description = request.json['description']
    try:
        model = joblib.load(f'user_{current_user.id}_category_model.pkl')
        prediction_name = model.predict([description])[0]
        
        # New logic: Find the category ID for the predicted name
        predicted_category = Category.query.filter_by(user_id=current_user.id, name=prediction_name).first()
        
        if predicted_category:
            return jsonify({'category_id': predicted_category.id})
        else:
            # Fallback if the predicted category name doesn't exist for the user
            return jsonify({'category_id': None})
            
    except FileNotFoundError:
        return jsonify({'category_id': None}) # Default if model doesn't exist

@app.route('/predict_balance')
@login_required
def predict_balance():
    try:
        # Fetch user's transaction data
        transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.asc()).all()

        if not transactions:
            return jsonify({"prediction": [], "model": "No data available"})

        # Prepare DataFrame
        df = pd.DataFrame([{
            "date": t.date,
            "balance": t.amount if t.type == 'income' else -t.amount
        } for t in transactions])

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.groupby('date').sum().reset_index().dropna(subset=['date'])

        prediction = []
        used_model = "Insufficient data"

        # --- RandomForest Forecast (lightweight, free-tier compatible) ---
        if len(df) >= 2:
            df['days'] = (df['date'] - df['date'].min()).dt.days
            X = df[['days']]
            y = df['balance']
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            future_days = np.arange(df['days'].max() + 1, df['days'].max() + 31).reshape(-1, 1)
            prediction = rf.predict(future_days).tolist()
            used_model = "RandomForest"

        return jsonify({"prediction": prediction, "model": used_model})

    except Exception as e:
        return jsonify({"prediction": [], "model": f"Error: {str(e)}"})


# --- SPA React Redirect Handler ---
@app.before_request
def redirect_to_react():
    # If the requested file actually exists in the React static folder, let Flask serve it
    if request.path != "" and os.path.exists(os.path.join(app.static_folder, 'dist', request.path.lstrip('/'))):
        return
        
    # If it is a GET request, accepts HTML, and is not an API call or uploads folder
    if request.method == 'GET' and \
       request.accept_mimetypes.accept_html and \
       not request.path.startswith('/api/') and \
       not request.path.startswith('/uploads/'):
        
        index_path = os.path.join(app.static_folder, 'dist', 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(os.path.join(app.static_folder, 'dist'), 'index.html')


# --- SPA React Catch-All Route ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, 'dist', path)):
        return send_from_directory(os.path.join(app.static_folder, 'dist'), path)
    
    if path.startswith('api/') or path in ['login', 'logout', 'register', 'healthz']:
        return jsonify({'error': 'Not Found'}), 404
        
    index_path = os.path.join(app.static_folder, 'dist', 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(os.path.join(app.static_folder, 'dist'), 'index.html')
    else:
        return "React Frontend is building... Please run 'npm run build' inside the 'frontend' folder, then refresh.", 200

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
