from flask import Blueprint, jsonify, request, session, current_app, Response
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import func
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np

from models import (
    db, User, Transaction, FixedScheme, Salary, Investment, SoldInvestment, Loan,
    BusinessTransaction, BusinessClient, BusinessInvestment, BusinessLoan,
    Category, Budget
)

api_bp = Blueprint('api', __name__, url_prefix='/api')

# --- Serialization Helpers ---
def serialize_category(c):
    if not c: return None
    return {'id': c.id, 'name': c.name, 'type': c.type}

def serialize_transaction(t):
    return {
        'id': t.id,
        'description': t.description,
        'amount': t.amount,
        'type': t.type,
        'date': t.date.strftime('%Y-%m-%d'),
        'category': serialize_category(t.category)
    }

def serialize_investment(i):
    return {
        'id': i.id,
        'asset_type': i.asset_type,
        'ticker_symbol': i.ticker_symbol,
        'quantity': i.quantity,
        'purchase_price': i.purchase_price,
        'purchase_currency': i.purchase_currency,
        'purchase_date': i.purchase_date.strftime('%Y-%m-%d')
    }

def serialize_sold_investment(s):
    return {
        'id': s.id,
        'asset_type': s.asset_type,
        'ticker_symbol': s.ticker_symbol,
        'quantity': s.quantity,
        'purchase_price': s.purchase_price,
        'purchase_date': s.purchase_date.strftime('%Y-%m-%d'),
        'sell_price': s.sell_price,
        'sell_date': s.sell_date.strftime('%Y-%m-%d'),
        'capital_gain': s.capital_gain,
        'gain_type': s.gain_type
    }

def serialize_loan(l):
    return {
        'id': l.id,
        'loan_name': l.loan_name,
        'principal': l.principal,
        'interest_rate': l.interest_rate,
        'tenure_months': l.tenure_months,
        'emi_amount': l.emi_amount,
        'start_date': l.start_date.strftime('%Y-%m-%d')
    }

def serialize_scheme(s):
    return {
        'id': s.id,
        'scheme_name': s.scheme_name,
        'principal_amount': s.principal_amount,
        'interest_rate': s.interest_rate,
        'tenure_months': s.tenure_months,
        'start_date': s.start_date.strftime('%Y-%m-%d'),
        'penalty_rate': s.penalty_rate
    }

def serialize_client(c):
    return {
        'id': c.id,
        'name': c.name,
        'email': c.email,
        'phone': c.phone,
        'company': c.company
    }

def serialize_business_transaction(t):
    return {
        'id': t.id,
        'description': t.description,
        'amount': t.amount,
        'type': t.type,
        'date': t.date.strftime('%Y-%m-%d'),
        'category': serialize_category(t.category),
        'client': serialize_client(t.client) if t.client else None,
        'invoice_status': t.invoice_status,
        'due_date': t.due_date.strftime('%Y-%m-%d') if t.due_date else None,
        'receipt_filename': t.receipt_filename
    }

def serialize_business_investment(i):
    return {
        'id': i.id,
        'investment_name': i.investment_name,
        'investment_type': i.investment_type,
        'amount_invested': i.amount_invested,
        'purchase_date': i.purchase_date.strftime('%Y-%m-%d') if i.purchase_date else None,
        'useful_life_years': i.useful_life_years,
        'current_value': i.current_value,
        'roi': i.roi
    }

def serialize_business_loan(l):
    return {
        'id': l.id,
        'loan_name': l.loan_name,
        'principal_amount': l.principal_amount,
        'interest_rate': l.interest_rate,
        'tenure_months': l.tenure_months,
        'start_date': l.start_date.strftime('%Y-%m-%d') if l.start_date else None,
        'emi': l.emi,
        'remaining_balance': l.remaining_balance
    }

# --- Auth APIs ---

@api_bp.route('/auth/status', methods=['GET'])
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'role': current_user.role.strip() if current_user.role else 'personal'
            }
        })
    return jsonify({'authenticated': False})

@api_bp.route('/auth/login', methods=['POST'])
def auth_login():
    data = request.get_json() or request.form
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password, password):
        login_user(user)
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'role': user.role.strip() if user.role else 'personal'
            }
        })
    return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401

@api_bp.route('/auth/register', methods=['POST'])
def auth_register():
    data = request.get_json() or request.form
    username = data.get('username')
    password = data.get('password')
    dob_str = data.get('dob')
    role = data.get('role', 'personal')

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists.'}), 400

    try:
        dob = datetime.strptime(dob_str, '%Y-%m-%d').date()
    except Exception:
        dob = date.today()

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    new_user = User(username=username, password=hashed_password, dob=dob, role=role)
    db.session.add(new_user)
    db.session.commit()
    
    # Auto-login after registration
    login_user(new_user)
    return jsonify({
        'success': True,
        'user': {
            'id': new_user.id,
            'username': new_user.username,
            'role': new_user.role
        }
    })

@api_bp.route('/auth/logout', methods=['GET', 'POST'])
@login_required
def auth_logout():
    logout_user()
    return jsonify({'success': True})

# --- Personal Dashboard API ---

@api_bp.route('/dashboard', methods=['GET'])
@login_required
def dashboard_data():
    today = date.today()
    start_of_month = today.replace(day=1)
    end_of_month = start_of_month + relativedelta(months=1)
    
    # 1. Salary Auto-Credit
    salary_details = Salary.query.filter_by(user_id=current_user.id).first()
    if salary_details and salary_details.monthly_gross > 0:
        salary_credited = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.description == "Monthly Salary",
            Transaction.date >= start_of_month,
            Transaction.date < end_of_month
        ).first()

        if not salary_credited:
            salary_category = Category.query.filter_by(user_id=current_user.id, name='Salary').first()
            if not salary_category:
                salary_category = Category(name='Salary', type='income', user_id=current_user.id)
                db.session.add(salary_category)
                db.session.commit()

            salary_transaction = Transaction(
                description="Monthly Salary",
                amount=salary_details.monthly_gross,
                type="income",
                category_id=salary_category.id,
                date=start_of_month,
                user_id=current_user.id
            )
            db.session.add(salary_transaction)
            db.session.commit()

    # 2. Loan EMI Auto-Debit
    user_loans = Loan.query.filter_by(user_id=current_user.id).all()
    if user_loans:
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
                    category_id=emi_category.id,
                    date=start_of_month,
                    user_id=current_user.id
                ))
                db.session.commit()

    # 3. Current month stats
    monthly_transactions = Transaction.query.filter(
        Transaction.user_id == current_user.id,
        Transaction.date >= start_of_month,
        Transaction.date < end_of_month
    ).all()
    monthly_income  = sum(float(t.amount or 0) for t in monthly_transactions if t.type == 'income')
    monthly_expense = sum(float(t.amount or 0) for t in monthly_transactions if t.type == 'expense')

    all_transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    balance = sum(float(t.amount or 0) for t in all_transactions if t.type == 'income') \
            - sum(float(t.amount or 0) for t in all_transactions if t.type == 'expense')

    recent_txs = Transaction.query.filter_by(
        user_id=current_user.id
    ).order_by(Transaction.date.desc()).limit(5).all()

    # 4. Real 6-month cash flow history (actual DB data — not faked multipliers)
    MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_history = []
    for i in range(5, -1, -1):
        target  = today - relativedelta(months=i)
        m_start = target.replace(day=1)
        m_end   = m_start + relativedelta(months=1)
        m_txs   = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date >= m_start,
            Transaction.date < m_end
        ).all()
        monthly_history.append({
            'name':    MONTH_LABELS[m_start.month - 1],
            'income':  round(sum(float(t.amount or 0) for t in m_txs if t.type == 'income'),  2),
            'expense': round(sum(float(t.amount or 0) for t in m_txs if t.type == 'expense'), 2),
        })

    # 5. Budget — only count expense from budgeted categories (so % is meaningful)
    current_month_num = today.month
    current_year      = today.year
    budgets = Budget.query.filter_by(
        user_id=current_user.id,
        month=current_month_num,
        year=current_year
    ).all()
    total_budgeted   = sum(float(b.amount or 0) for b in budgets)
    budgeted_cat_ids = [b.category_id for b in budgets]

    if budgeted_cat_ids:
        budgeted_expense = db.session.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == current_user.id,
            Transaction.type == 'expense',
            Transaction.category_id.in_(budgeted_cat_ids),
            func.extract('month', Transaction.date) == current_month_num,
            func.extract('year',  Transaction.date) == current_year
        ).scalar() or 0.0
    else:
        budgeted_expense = 0.0

    raw_pct         = (float(budgeted_expense) / total_budgeted * 100) if total_budgeted > 0 else 0
    is_over_budget  = raw_pct > 100
    percentage_ring = min(int(raw_pct), 100)  # capped for SVG arc — label shows real value

    return jsonify({
        'balance':         round(balance, 2),
        'monthly_income':  round(monthly_income, 2),
        'monthly_expense': round(monthly_expense, 2),
        'monthly_history': monthly_history,
        'recent_transactions': [serialize_transaction(t) for t in recent_txs],
        'budget': {
            'total_budgeted':   round(total_budgeted, 2),
            'total_spent':      round(float(budgeted_expense), 2),
            'total_expense':    round(monthly_expense, 2),
            'percentage_spent': percentage_ring,
            'raw_percentage':   round(raw_pct, 1),
            'is_over_budget':   is_over_budget,
            'has_budgets':      len(budgets) > 0,
        }
    })

# --- Transactions CRUD ---

@api_bp.route('/transactions', methods=['GET', 'POST'])
@login_required
def manage_transactions():
    if request.method == 'GET':
        txs = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()
        categories = Category.query.filter_by(user_id=current_user.id).all()
        return jsonify({
            'transactions': [serialize_transaction(t) for t in txs],
            'categories': [serialize_category(c) for c in categories]
        })
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        description = data.get('description')
        amount = float(data.get('amount', 0))
        tx_type = data.get('type')
        category_id = data.get('category_id')
        date_str = data.get('date')
        
        try:
            tx_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            tx_date = date.today()

        new_tx = Transaction(
            description=description,
            amount=amount,
            type=tx_type,
            category_id=category_id,
            date=tx_date,
            user_id=current_user.id
        )
        db.session.add(new_tx)
        db.session.commit()
        return jsonify({'success': True, 'transaction': serialize_transaction(new_tx)})

@api_bp.route('/transactions/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def edit_delete_transaction(id):
    tx = Transaction.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    
    if request.method == 'DELETE':
        db.session.delete(tx)
        db.session.commit()
        return jsonify({'success': True})
        
    elif request.method == 'PUT':
        data = request.get_json()
        tx.description = data.get('description', tx.description)
        tx.amount = float(data.get('amount', tx.amount))
        tx.type = data.get('type', tx.type)
        tx.category_id = data.get('category_id', tx.category_id)
        if data.get('date'):
            try:
                tx.date = datetime.strptime(data.get('date'), '%Y-%m-%d').date()
            except Exception:
                pass
        db.session.commit()
        return jsonify({'success': True, 'transaction': serialize_transaction(tx)})

# --- Categories CRUD ---

@api_bp.route('/categories', methods=['GET', 'POST'])
@login_required
def manage_categories():
    if request.method == 'GET':
        categories = Category.query.filter_by(user_id=current_user.id).all()
        return jsonify({'categories': [serialize_category(c) for c in categories]})
        
    elif request.method == 'POST':
        data = request.get_json()
        name = data.get('name')
        c_type = data.get('type', 'expense')
        
        existing = Category.query.filter_by(user_id=current_user.id, name=name).first()
        if existing:
            return jsonify({'success': False, 'message': 'Category already exists.'}), 400
            
        new_cat = Category(name=name, type=c_type, user_id=current_user.id)
        db.session.add(new_cat)
        db.session.commit()
        return jsonify({'success': True, 'category': serialize_category(new_cat)})

@api_bp.route('/categories/<int:id>', methods=['DELETE'])
@login_required
def delete_category(id):
    c = Category.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    # Unlink transactions
    Transaction.query.filter_by(category_id=id).update({Transaction.category_id: None})
    db.session.delete(c)
    db.session.commit()
    return jsonify({'success': True})

# --- Budgets ---

@api_bp.route('/budgets', methods=['GET', 'POST'])
@login_required
def manage_budgets():
    today = date.today()
    month = int(request.args.get('month', today.month))
    year = int(request.args.get('year', today.year))
    
    if request.method == 'GET':
        budgets = Budget.query.filter_by(user_id=current_user.id, month=month, year=year).all()
        categories = Category.query.filter_by(user_id=current_user.id, type='expense').all()
        
        # Calculate spent for each category
        spent_map = {}
        for cat in categories:
            spent = db.session.query(func.sum(Transaction.amount)).filter(
                Transaction.user_id == current_user.id,
                Transaction.category_id == cat.id,
                Transaction.type == 'expense',
                func.extract('month', Transaction.date) == month,
                func.extract('year', Transaction.date) == year
            ).scalar() or 0.0
            spent_map[cat.id] = spent

        budget_list = []
        for b in budgets:
            budget_list.append({
                'id': b.id,
                'category_id': b.category_id,
                'category_name': b.category.name if b.category else 'Unknown',
                'amount': b.amount,
                'spent': spent_map.get(b.category_id, 0.0)
            })

        # Add entries for categories without set budgets
        budgeted_cat_ids = [b.category_id for b in budgets]
        for cat in categories:
            if cat.id not in budgeted_cat_ids:
                budget_list.append({
                    'id': None,
                    'category_id': cat.id,
                    'category_name': cat.name,
                    'amount': 0,
                    'spent': spent_map.get(cat.id, 0.0)
                })

        return jsonify({'budgets': budget_list})
        
    elif request.method == 'POST':
        data = request.get_json()
        category_id = data.get('category_id')
        amount = float(data.get('amount', 0))
        
        b = Budget.query.filter_by(user_id=current_user.id, category_id=category_id, month=month, year=year).first()
        if b:
            b.amount = amount
        else:
            b = Budget(category_id=category_id, amount=amount, month=month, year=year, user_id=current_user.id)
            db.session.add(b)
            
        db.session.commit()
        return jsonify({'success': True})

# --- Net Worth & Live Prices API ---

@api_bp.route('/net_worth', methods=['GET'])
@login_required
def net_worth():
    import concurrent.futures

    try:
        from app import get_coingecko_id

        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        cash_balance = sum(float(t.amount or 0) for t in transactions if t.type == 'income') \
                     - sum(float(t.amount or 0) for t in transactions if t.type == 'expense')

        # ── Fixed Schemes (fast, no network) ──────────────────────────────────
        user_schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
        total_schemes_value = 0.0
        schemes_detailed = []
        for scheme in user_schemes:
            years_elapsed = 0.0
            if scheme.start_date:
                years_elapsed = (date.today() - scheme.start_date).days / 365.25
            principal = float(scheme.principal_amount or 0)
            interest_rate = float(scheme.interest_rate or 0)
            cur_val = principal
            if years_elapsed > 0:
                cur_val = principal * ((1 + (interest_rate / 100)) ** years_elapsed)
            total_schemes_value += cur_val
            schemes_detailed.append({
                'id': scheme.id,
                'name': scheme.scheme_name,
                'principal': principal,
                'current_value': cur_val
            })

        # ── USD→INR rate (5-second timeout) ───────────────────────────────────
        def fetch_usd_inr():
            try:
                cg = CoinGeckoAPI()
                rates = cg.get_price(ids='tether', vs_currencies='inr')
                return float(rates['tether']['inr'])
            except Exception:
                return 83.5

        usd_to_inr_rate = 83.5
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fetch_usd_inr)
            try:
                usd_to_inr_rate = fut.result(timeout=5)
            except Exception:
                pass

        # ── Live prices per investment (5-second timeout each) ────────────────
        user_investments = Investment.query.filter_by(user_id=current_user.id).all()
        total_stocks_value = 0.0
        total_crypto_value = 0.0
        investments_detailed = []

        def fetch_price(inv):
            """Returns (current_price_in_native, total_val_inr, asset_category)"""
            qty = float(inv.quantity or 0)
            purchase_price = float(inv.purchase_price or 0)
            currency = inv.purchase_currency or 'INR'
            current_price = 0.0
            total_val_inr = 0.0
            category = 'stocks' if inv.asset_type == 'Stock' else 'crypto'
            try:
                if inv.asset_type == 'Stock':
                    stock = yf.Ticker(inv.ticker_symbol)
                    hist = stock.history(period='1d')
                    if not hist.empty:
                        raw_price = float(hist['Close'].iloc[-1])
                        current_price = raw_price
                        total_val_inr = (qty * raw_price * usd_to_inr_rate) if currency == 'USD' else (qty * raw_price)
                elif inv.asset_type == 'Crypto':
                    cg_inner = CoinGeckoAPI()
                    crypto_id = get_coingecko_id(inv.ticker_symbol)
                    price_data = cg_inner.get_price(ids=crypto_id, vs_currencies='usd')
                    if price_data and price_data.get(crypto_id):
                        usd_price = float(price_data[crypto_id].get('usd', 0))
                        current_price = usd_price
                        total_val_inr = qty * usd_price * usd_to_inr_rate
            except Exception:
                pass
            return {
                'id': inv.id,
                'ticker': inv.ticker_symbol.upper(),
                'asset_type': inv.asset_type,
                'quantity': qty,
                'purchase_price': purchase_price,
                'purchase_currency': currency,
                'current_price': current_price,
                'current_value_inr': total_val_inr,
                '_category': category,
                '_val_inr': total_val_inr
            }

        if user_investments:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
                futures = {ex.submit(fetch_price, inv): inv for inv in user_investments}
                for fut in concurrent.futures.as_completed(futures, timeout=15):
                    try:
                        result = fut.result(timeout=1)
                        investments_detailed.append(result)
                        if result['_category'] == 'stocks':
                            total_stocks_value += result['_val_inr']
                        else:
                            total_crypto_value += result['_val_inr']
                    except Exception:
                        inv = futures[fut]
                        investments_detailed.append({
                            'id': inv.id,
                            'ticker': inv.ticker_symbol.upper(),
                            'asset_type': inv.asset_type,
                            'quantity': float(inv.quantity or 0),
                            'purchase_price': float(inv.purchase_price or 0),
                            'purchase_currency': inv.purchase_currency or 'INR',
                            'current_price': 0.0,
                            'current_value_inr': 0.0,
                        })

        # Clean internal keys before serializing
        for item in investments_detailed:
            item.pop('_category', None)
            item.pop('_val_inr', None)

        total_assets = cash_balance + total_schemes_value + total_stocks_value + total_crypto_value

        user_loans = Loan.query.filter_by(user_id=current_user.id).all()
        total_liabilities = sum(float(l.principal or 0) for l in user_loans)

        return jsonify({
            'net_worth': total_assets - total_liabilities,
            'usd_to_inr_rate': usd_to_inr_rate,
            'assets': {
                'cash': cash_balance,
                'schemes': total_schemes_value,
                'stocks': total_stocks_value,
                'crypto': total_crypto_value,
                'total': total_assets
            },
            'liabilities': {
                'loans': total_liabilities,
                'total': total_liabilities
            },
            'details': {
                'schemes': schemes_detailed,
                'investments': investments_detailed
            }
        })

    except Exception as e:
        print(f"Error in net_worth: {e}")
        return jsonify({
            'net_worth': 0.0,
            'usd_to_inr_rate': 83.5,
            'assets': {'cash': 0.0, 'schemes': 0.0, 'stocks': 0.0, 'crypto': 0.0, 'total': 0.0},
            'liabilities': {'loans': 0.0, 'total': 0.0},
            'details': {'schemes': [], 'investments': []},
            'error': str(e)
        })

# --- Schemes APIs ---

@api_bp.route('/schemes', methods=['GET', 'POST'])
@login_required
def manage_schemes():
    if request.method == 'GET':
        schemes = FixedScheme.query.filter_by(user_id=current_user.id).all()
        return jsonify({'schemes': [serialize_scheme(s) for s in schemes]})
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        name = data.get('scheme_name')
        principal = float(data.get('principal_amount', 0))
        rate = float(data.get('interest_rate', 0))
        tenure = int(data.get('tenure_months', 0))
        date_str = data.get('start_date')
        penalty = float(data.get('penalty_rate', 0))
        
        try:
            start_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            start_date = date.today()

        new_scheme = FixedScheme(
            scheme_name=name,
            principal_amount=principal,
            interest_rate=rate,
            tenure_months=tenure,
            start_date=start_date,
            penalty_rate=penalty,
            user_id=current_user.id
        )
        db.session.add(new_scheme)
        db.session.commit()
        return jsonify({'success': True, 'scheme': serialize_scheme(new_scheme)})

@api_bp.route('/schemes/<int:id>', methods=['DELETE'])
@login_required
def delete_scheme(id):
    s = FixedScheme.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    db.session.delete(s)
    db.session.commit()
    return jsonify({'success': True})

# --- Loans APIs ---

@api_bp.route('/loans', methods=['GET', 'POST'])
@login_required
def manage_loans():
    if request.method == 'GET':
        loans = Loan.query.filter_by(user_id=current_user.id).all()
        return jsonify({'loans': [serialize_loan(l) for l in loans]})
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        name = data.get('loan_name')
        principal = float(data.get('principal', 0))
        rate = float(data.get('interest_rate', 0))
        tenure = int(data.get('tenure_months', 0))
        date_str = data.get('start_date')
        
        try:
            start_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            start_date = date.today()

        # EMI calculation
        r = (rate / 12) / 100
        n = tenure
        emi = principal * r * ((1 + r) ** n) / (((1 + r) ** n) - 1) if r > 0 else principal / n

        new_loan = Loan(
            loan_name=name,
            principal=principal,
            interest_rate=rate,
            tenure_months=tenure,
            emi_amount=emi,
            start_date=start_date,
            user_id=current_user.id
        )
        db.session.add(new_loan)
        db.session.commit()
        return jsonify({'success': True, 'loan': serialize_loan(new_loan)})

@api_bp.route('/loans/<int:id>', methods=['DELETE'])
@login_required
def delete_loan(id):
    l = Loan.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    db.session.delete(l)
    db.session.commit()
    return jsonify({'success': True})

# --- Investments APIs ---

@api_bp.route('/investments', methods=['GET', 'POST'])
@login_required
def manage_investments():
    if request.method == 'GET':
        invs = Investment.query.filter_by(user_id=current_user.id).all()
        sales = SoldInvestment.query.filter_by(user_id=current_user.id).order_by(SoldInvestment.sell_date.desc()).all()
        return jsonify({
            'investments': [serialize_investment(i) for i in invs],
            'sales': [serialize_sold_investment(s) for s in sales]
        })
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        asset_type = data.get('asset_type')
        ticker = data.get('ticker_symbol').lower() if asset_type == 'Crypto' else data.get('ticker_symbol').upper()
        quantity = float(data.get('quantity', 0))
        price = float(data.get('purchase_price', 0))
        currency = data.get('purchase_currency', 'INR')
        date_str = data.get('purchase_date')
        
        try:
            p_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            p_date = date.today()

        new_inv = Investment(
            asset_type=asset_type,
            ticker_symbol=ticker,
            quantity=quantity,
            purchase_price=price,
            purchase_currency=currency,
            purchase_date=p_date,
            user_id=current_user.id
        )
        db.session.add(new_inv)
        db.session.commit()
        return jsonify({'success': True, 'investment': serialize_investment(new_inv)})

@api_bp.route('/investments/<int:id>', methods=['DELETE'])
@login_required
def delete_investment(id):
    i = Investment.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    db.session.delete(i)
    db.session.commit()
    return jsonify({'success': True})

@api_bp.route('/investments/sell/<int:id>', methods=['POST'])
@login_required
def sell_investment(id):
    data = request.get_json() or request.form
    sell_qty = float(data.get('sell_quantity', 0))
    sell_price = float(data.get('sell_price', 0))
    sell_date_str = data.get('sell_date')
    
    try:
        sell_date = datetime.strptime(sell_date_str, '%Y-%m-%d').date()
    except Exception:
        sell_date = date.today()

    inv = Investment.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    
    if sell_qty > inv.quantity:
        return jsonify({'success': False, 'message': 'Sell quantity exceeds holdings.'}), 400
        
    capital_gain = (sell_price - inv.purchase_price) * sell_qty
    holding_days = (sell_date - inv.purchase_date).days
    gain_type = "LTCG" if holding_days > 365 else "STCG"

    sold_inv = SoldInvestment(
        asset_type=inv.asset_type,
        ticker_symbol=inv.ticker_symbol,
        quantity=sell_qty,
        purchase_price=inv.purchase_price,
        purchase_date=inv.purchase_date,
        sell_price=sell_price,
        sell_date=sell_date,
        capital_gain=capital_gain,
        gain_type=gain_type,
        user_id=current_user.id
    )
    db.session.add(sold_inv)

    if sell_qty == inv.quantity:
        db.session.delete(inv)
    else:
        inv.quantity -= sell_qty
        
    db.session.commit()
    return jsonify({'success': True})

# --- Salary APIs ---

@api_bp.route('/salary', methods=['GET', 'POST'])
@login_required
def manage_salary():
    salary = Salary.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'GET':
        return jsonify({
            'monthly_gross': salary.monthly_gross if salary else 0.0,
            'deductions_80c': salary.deductions_80c if salary else 0.0,
            'hra_exemption': salary.hra_exemption if salary else 0.0
        })
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        monthly_gross = float(data.get('monthly_gross', 0))
        deductions_80c = float(data.get('deductions_80c', 0))
        hra_exemption = float(data.get('hra_exemption', 0))
        
        if salary:
            salary.monthly_gross = monthly_gross
            salary.deductions_80c = deductions_80c
            salary.hra_exemption = hra_exemption
        else:
            salary = Salary(
                monthly_gross=monthly_gross,
                deductions_80c=deductions_80c,
                hra_exemption=hra_exemption,
                user_id=current_user.id
            )
            db.session.add(salary)
            
        db.session.commit()
        return jsonify({'success': True})

# --- Tax Estimator API ---

@api_bp.route('/tax', methods=['GET'])
@login_required
def tax_estimator():
    from app import calculate_new_regime_tax, calculate_old_regime_tax
    salary = Salary.query.filter_by(user_id=current_user.id).first()
    
    if not salary or salary.monthly_gross <= 0:
        return jsonify({'error': 'Please set up salary details first.'}), 400
        
    gross_annual = salary.monthly_gross * 12
    # Standard deduction + 80C + HRA
    total_deductions = salary.deductions_80c + salary.hra_exemption
    
    # Calculate age from DOB
    age = 30
    if current_user.dob:
        age = (date.today() - current_user.dob).days // 365
        
    new_regime = calculate_new_regime_tax(gross_annual)
    old_regime = calculate_old_regime_tax(gross_annual, total_deductions, age)
    
    return jsonify({
        'gross_annual': gross_annual,
        'new_regime': new_regime,
        'old_regime': old_regime,
        'recommendation': 'New Regime' if new_regime['total_tax'] <= old_regime['total_tax'] else 'Old Regime'
    })

# --- Business Client & Invoices APIs ---

@api_bp.route('/business/dashboard', methods=['GET'])
@login_required
def business_dashboard():
    # Clients count
    clients_count = BusinessClient.query.filter_by(user_id=current_user.id).count()
    # Income/Expense for Business
    txs = BusinessTransaction.query.filter_by(user_id=current_user.id).all()
    total_revenue = sum(t.amount for t in txs if t.type == 'revenue')
    total_expense = sum(t.amount for t in txs if t.type == 'expense')
    net_profit = total_revenue - total_expense
    
    recent_txs = BusinessTransaction.query.filter_by(
        user_id=current_user.id
    ).order_by(BusinessTransaction.date.desc()).limit(5).all()
    
    # Pending invoice metrics
    pending_invoices = BusinessTransaction.query.filter_by(
        user_id=current_user.id,
        invoice_status='unpaid'
    ).all()
    total_outstanding = sum(inv.amount for inv in pending_invoices)

    return jsonify({
        'clients_count': clients_count,
        'total_revenue': total_revenue,
        'total_expense': total_expense,
        'net_profit': net_profit,
        'total_outstanding': total_outstanding,
        'recent_transactions': [serialize_business_transaction(t) for t in recent_txs]
    })

@api_bp.route('/business/clients', methods=['GET', 'POST'])
@login_required
def manage_clients():
    if request.method == 'GET':
        clients = BusinessClient.query.filter_by(user_id=current_user.id).all()
        return jsonify({'clients': [serialize_client(c) for c in clients]})
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        name = data.get('name')
        email = data.get('email')
        phone = data.get('phone')
        company = data.get('company')
        
        new_client = BusinessClient(
            name=name, email=email, phone=phone, company=company, user_id=current_user.id
        )
        db.session.add(new_client)
        db.session.commit()
        return jsonify({'success': True, 'client': serialize_client(new_client)})

@api_bp.route('/business/clients/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def edit_delete_client(id):
    client = BusinessClient.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    
    if request.method == 'DELETE':
        db.session.delete(client)
        db.session.commit()
        return jsonify({'success': True})
        
    elif request.method == 'PUT':
        data = request.get_json()
        client.name = data.get('name', client.name)
        client.email = data.get('email', client.email)
        client.phone = data.get('phone', client.phone)
        client.company = data.get('company', client.company)
        db.session.commit()
        return jsonify({'success': True, 'client': serialize_client(client)})

@api_bp.route('/business/transactions', methods=['GET', 'POST'])
@login_required
def manage_business_transactions():
    if request.method == 'GET':
        txs = BusinessTransaction.query.filter_by(user_id=current_user.id).order_by(BusinessTransaction.date.desc()).all()
        clients = BusinessClient.query.filter_by(user_id=current_user.id).all()
        categories = Category.query.filter_by(user_id=current_user.id).all()
        return jsonify({
            'transactions': [serialize_business_transaction(t) for t in txs],
            'clients': [serialize_client(c) for c in clients],
            'categories': [serialize_category(c) for c in categories]
        })
        
    elif request.method == 'POST':
        data = request.get_json() or request.form
        description = data.get('description')
        amount = float(data.get('amount', 0))
        tx_type = data.get('type')
        category_id = data.get('category_id')
        client_id = data.get('client_id')
        invoice_status = data.get('invoice_status', 'paid')
        date_str = data.get('date')
        due_date_str = data.get('due_date')
        
        try:
            tx_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except Exception:
            tx_date = date.today()

        due_date = None
        if due_date_str:
            try:
                due_date = datetime.strptime(due_date_str, '%Y-%m-%d').date()
            except Exception:
                pass

        new_tx = BusinessTransaction(
            description=description,
            amount=amount,
            type=tx_type,
            category_id=category_id,
            client_id=client_id if client_id else None,
            invoice_status=invoice_status,
            date=tx_date,
            due_date=due_date,
            user_id=current_user.id
        )
        db.session.add(new_tx)
        db.session.commit()
        return jsonify({'success': True, 'transaction': serialize_business_transaction(new_tx)})

@api_bp.route('/business/transactions/<int:id>', methods=['PUT', 'DELETE'])
@login_required
def edit_delete_business_transaction(id):
    tx = BusinessTransaction.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    
    if request.method == 'DELETE':
        db.session.delete(tx)
        db.session.commit()
        return jsonify({'success': True})
        
    elif request.method == 'PUT':
        data = request.get_json()
        tx.description = data.get('description', tx.description)
        tx.amount = float(data.get('amount', tx.amount))
        tx.type = data.get('type', tx.type)
        tx.category_id = data.get('category_id', tx.category_id)
        tx.client_id = data.get('client_id', tx.client_id)
        tx.invoice_status = data.get('invoice_status', tx.invoice_status)
        if data.get('date'):
            try:
                tx.date = datetime.strptime(data.get('date'), '%Y-%m-%d').date()
            except Exception:
                pass
        if data.get('due_date'):
            try:
                tx.due_date = datetime.strptime(data.get('due_date'), '%Y-%m-%d').date()
            except Exception:
                pass
        db.session.commit()
        return jsonify({'success': True, 'transaction': serialize_business_transaction(tx)})

# --- AI Insights API ---

@api_bp.route('/ai_insights', methods=['GET'])
@login_required
def ai_insights_api():
    from app import ai_insights
    return ai_insights()

@api_bp.route('/predict_balance', methods=['GET'])
@login_required
def predict_balance_api():
    from app import predict_balance
    return predict_balance()

@api_bp.route('/predict_category', methods=['POST'])
@login_required
def predict_category_api():
    from app import predict_category
    return predict_category()

# --- Profile API ---

@api_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile_api():
    if request.method == 'GET':
        return jsonify({
            'username': current_user.username,
            'dob': current_user.dob.strftime('%Y-%m-%d') if current_user.dob else None,
            'role': current_user.role.strip() if current_user.role else 'personal',
        })
    data = request.get_json() or {}
    if data.get('dob'):
        try:
            from datetime import datetime as _dt
            current_user.dob = _dt.strptime(data['dob'], '%Y-%m-%d').date()
        except Exception:
            pass
    if data.get('role'):
        current_user.role = data['role']
    db.session.commit()
    return jsonify({'success': True})

# --- Reports API ---

@api_bp.route('/reports', methods=['POST'])
@login_required
def reports_api():
    data = request.get_json() or {}
    try:
        from datetime import datetime as _dt
        start_date = _dt.strptime(data.get('start_date'), '%Y-%m-%d').date()
        end_date = _dt.strptime(data.get('end_date'), '%Y-%m-%d').date()

        transactions = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            Transaction.date.between(start_date, end_date)
        ).all()

        total_income = sum(t.amount for t in transactions if t.type == 'income')
        total_expense = sum(t.amount for t in transactions if t.type == 'expense')
        expenses_by_category = {}
        for t in transactions:
            if t.type == 'expense':
                cat_name = t.category.name if t.category else 'Uncategorized'
                expenses_by_category[cat_name] = expenses_by_category.get(cat_name, 0) + t.amount

        return jsonify({
            'start_date': start_date.strftime('%d %b %Y'),
            'end_date': end_date.strftime('%d %b %Y'),
            'total_income': total_income,
            'total_expense': total_expense,
            'net_savings': total_income - total_expense,
            'expenses_by_category': expenses_by_category,
            'transactions': [serialize_transaction(t) for t in transactions]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Business Financials, Investments, Loans, Reports, and Insights APIs ---

@api_bp.route('/business/financials', methods=['GET'])
@login_required
def business_financials_api():
    try:
        today = date.today()
        start_of_month = today.replace(day=1)
        end_of_month = start_of_month + relativedelta(months=1)

        transactions_this_month = BusinessTransaction.query.filter(
            BusinessTransaction.user_id == current_user.id,
            BusinessTransaction.date >= start_of_month,
            BusinessTransaction.date < end_of_month
        ).all()

        revenue = sum(float(t.amount or 0) for t in transactions_this_month if t.type == 'revenue')
        expenses = sum(float(t.amount or 0) for t in transactions_this_month if t.type == 'expense')
        net_profit = revenue - expenses
        profit_margin = (net_profit / revenue) * 100 if revenue > 0 else 0

        # Overall summary
        clients_count = BusinessClient.query.filter_by(user_id=current_user.id).count()
        
        user_investments = BusinessInvestment.query.filter_by(user_id=current_user.id).all()
        total_investment_value = sum(float(inv.current_value or 0) for inv in user_investments)
        
        user_loans = BusinessLoan.query.filter_by(user_id=current_user.id).all()
        total_outstanding_loans = sum(float(l.remaining_balance or 0) for l in user_loans)

        return jsonify({
            'monthly_revenue': revenue,
            'monthly_expenses': expenses,
            'net_profit': net_profit,
            'profit_margin': profit_margin,
            'active_clients': clients_count,
            'total_investment_value': total_investment_value,
            'total_outstanding_loans': total_outstanding_loans
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/business/investments', methods=['GET', 'POST'])
@login_required
def business_investments_api():
    if request.method == 'GET':
        try:
            investments = BusinessInvestment.query.filter_by(user_id=current_user.id).all()
            return jsonify({
                'investments': [serialize_business_investment(inv) for inv in investments],
                'today': date.today().strftime('%Y-%m-%d')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif request.method == 'POST':
        try:
            data = request.get_json() or request.form
            name = data.get('investment_name')
            inv_type = data.get('investment_type')
            amount = float(data.get('amount_invested', 0))
            
            p_date_str = data.get('purchase_date')
            try:
                p_date = datetime.strptime(p_date_str, '%Y-%m-%d').date()
            except Exception:
                p_date = date.today()
                
            life_years = data.get('useful_life_years')
            if life_years is not None and life_years != '':
                life_years = int(life_years)
            else:
                life_years = None

            if not name or not inv_type or amount <= 0:
                return jsonify({'error': 'Valid name, type, and positive amount are required.'}), 400

            new_inv = BusinessInvestment(
                user_id=current_user.id,
                investment_name=name,
                investment_type=inv_type,
                amount_invested=amount,
                purchase_date=p_date,
                useful_life_years=life_years
            )
            db.session.add(new_inv)
            db.session.commit()
            return jsonify({'success': True, 'investment': serialize_business_investment(new_inv)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@api_bp.route('/business/investments/<int:id>', methods=['DELETE'])
@login_required
def delete_business_investment_api(id):
    try:
        inv = BusinessInvestment.query.filter_by(id=id, user_id=current_user.id).first_or_404()
        db.session.delete(inv)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/business/loans', methods=['GET', 'POST'])
@login_required
def business_loans_api():
    if request.method == 'GET':
        try:
            loans = BusinessLoan.query.filter_by(user_id=current_user.id).all()
            return jsonify({
                'loans': [serialize_business_loan(l) for l in loans],
                'today': date.today().strftime('%Y-%m-%d')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    elif request.method == 'POST':
        try:
            data = request.get_json() or request.form
            name = data.get('loan_name')
            principal = float(data.get('principal_amount', 0))
            rate = float(data.get('interest_rate', 0))
            tenure = int(data.get('tenure_months', 0))
            
            s_date_str = data.get('start_date')
            try:
                s_date = datetime.strptime(s_date_str, '%Y-%m-%d').date()
            except Exception:
                s_date = date.today()

            if not name or principal <= 0 or rate <= 0 or tenure <= 0:
                return jsonify({'error': 'All fields are required with positive numbers.'}), 400

            # Calculate EMI
            r = (rate / 12) / 100
            emi = (principal * r * (1 + r)**tenure) / (((1 + r)**tenure) - 1)
            
            new_loan = BusinessLoan(
                user_id=current_user.id,
                loan_name=name,
                principal_amount=principal,
                interest_rate=rate,
                tenure_months=tenure,
                start_date=s_date,
                emi=emi
            )
            db.session.add(new_loan)
            db.session.commit()
            return jsonify({'success': True, 'loan': serialize_business_loan(new_loan)})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@api_bp.route('/business/loans/<int:id>', methods=['DELETE'])
@login_required
def delete_business_loan_api(id):
    try:
        loan = BusinessLoan.query.filter_by(id=id, user_id=current_user.id).first_or_404()
        db.session.delete(loan)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/business/reports', methods=['POST'])
@login_required
def business_reports_api():
    try:
        from app import generate_business_pdf_report, generate_business_csv_report
        
        data = request.get_json() or {}
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        report_format = data.get('format', 'pdf')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

        if start_date > end_date:
            return jsonify({'error': 'Start date cannot be after end date.'}), 400
        
        # Query transactions in the date range
        transactions = BusinessTransaction.query.filter(
            BusinessTransaction.user_id == current_user.id,
            BusinessTransaction.date.between(start_date, end_date)
        ).all()

        # Calculate metrics
        total_revenue = sum(float(t.amount or 0) for t in transactions if t.type == 'revenue')
        total_expenses = sum(float(t.amount or 0) for t in transactions if t.type == 'expense')
        
        expenses_by_category = {}
        for t in transactions:
            if t.type == 'expense':
                cat_name = t.category.name if t.category else 'Uncategorized'
                expenses_by_category[cat_name] = expenses_by_category.get(cat_name, 0) + float(t.amount or 0)

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
            return Response(
                pdf_data,
                mimetype='application/pdf',
                headers={'Content-Disposition': 'attachment;filename=business_report.pdf'}
            )
        elif report_format == 'csv':
            csv_data = generate_business_csv_report(report_data)
            return Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment;filename=business_report.csv'}
            )
        else:
            return jsonify({'error': 'Invalid report format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/business/insights', methods=['POST'])
@login_required
def business_insights_api():
    try:
        from huggingface_hub import InferenceClient
        data = request.get_json() or {}
        month = int(data.get('month', date.today().month))
        year = int(data.get('year', date.today().year))

        start_date = date(year, month, 1)
        if month == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month + 1, 1) - timedelta(days=1)

        transactions = BusinessTransaction.query.filter(
            BusinessTransaction.user_id == current_user.id,
            BusinessTransaction.date.between(start_date, end_date)
        ).all()

        if not transactions:
            return jsonify({'insights': 'No transactions found for the selected period.'})

        # Aggregate data for the AI
        total_revenue = sum(float(t.amount or 0) for t in transactions if t.type == 'revenue')
        total_expenses = sum(float(t.amount or 0) for t in transactions if t.type == 'expense')
        net_profit = total_revenue - total_expenses
        
        expenses_by_category = {}
        for t in transactions:
            if t.type == 'expense':
                cat_name = t.category.name if t.category else 'Uncategorized'
                expenses_by_category[cat_name] = expenses_by_category.get(cat_name, 0) + float(t.amount or 0)
        
        top_expense_category = max(expenses_by_category, key=expenses_by_category.get) if expenses_by_category else "N/A"

        # --- AI Prompt Engineering ---
        prompt = f"""
        As a friendly financial analyst, analyze the following monthly data for a small business owner and provide a concise, easy-to-understand summary in bullet points. Focus on key takeaways and recommendations.

        DATA FOR {start_date.strftime('%B %Y')}:
        - Total Revenue: {total_revenue:.2f}
        - Total Expenses: {total_expenses:.2f}
        - Net Profit/Loss: {net_profit:.2f}
        - Total Number of Transactions: {len(transactions)}
        - Top Expense Category: {top_expense_category} with an amount of {expenses_by_category.get(top_expense_category, 0):.2f}

        Based on this data, generate a short summary.
        """

        insights = None
        
        from app import _gemini_client
        if _gemini_client:
            try:
                response = _gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if response and response.text:
                    insights = response.text
            except Exception as e:
                print(f"Gemini error in business insights: {e}")
                
        if not insights:
            try:
                client = InferenceClient()
                insights = client.text_generation(prompt, model="mistralai/Mistral-7B-Instruct-v0.2", max_new_tokens=250)
            except Exception as e:
                print(f"HuggingFace error in business insights: {e}")
                
        if not insights:
            insights = f"**Summary of {start_date.strftime('%B %Y')}**:\n"
            insights += f"- Total Revenue was INR {total_revenue:,.2f}.\n"
            insights += f"- Total Expenses were INR {total_expenses:,.2f}.\n"
            insights += f"- Net Cash flow is INR {net_profit:,.2f}.\n"
            if top_expense_category != "N/A":
                insights += f"- Your top spending area was **{top_expense_category}** (INR {expenses_by_category.get(top_expense_category, 0):,.2f}).\n"
            insights += "\n*Note: AI generated insights are currently unavailable, displaying auto-generated fallback calculations.*"

        return jsonify({'insights': insights})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


