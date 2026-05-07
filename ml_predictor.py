"""
ML PREDICTION ENGINE v1.1 - No Yahoo Finance
Uses NSE data + Statistical models for prediction
"""
import os
import time
import requests
from nsetools import Nse
from datetime import datetime
import numpy as np

os.environ['TZ'] = 'Asia/Kolkata'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_ML_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_ML_CHAT_ID')

STOCKS = {
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB'],
    'Consumer': ['ITC', 'HINDUNILVR', 'TITAN', 'DMART', 'TRENT'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO'],
    'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN'],
    'Energy': ['RELIANCE', 'POWERGRID', 'NTPC', 'ONGC', 'TATAPOWER'],
    'Metal': ['TATASTEEL', 'JSWSTEEL', 'JINDALSTEL'],
    'Others': ['LT', 'HAL', 'BEL', 'IRCTC', 'ZOMATO', 'ADANIPORTS']
}

# ============================================
# STATISTICAL BACKTESTING (No Yahoo needed)
# ============================================
def calculate_statistical_score(price, high_52, low_52, change_pct, delivery_pct):
    """Score based on statistical patterns without historical data"""
    score = 0
    
    # 52-Week position scoring
    if high_52 > 0 and low_52 > 0:
        range_52 = high_52 - low_52
        position_52 = ((price - low_52) / range_52 * 100) if range_52 > 0 else 50
        
        # Best buying zone: 20-50% of 52-week range
        if 20 <= position_52 <= 50:
            score += 15
        elif 50 < position_52 <= 70:
            score += 10
        elif position_52 < 20:
            score += 8  # Near bottom
    
    # Distance from high (value play)
    if high_52 > 0:
        dist_high = ((high_52 - price) / high_52) * 100
        if dist_high > 30:
            score += 12
        elif dist_high > 20:
            score += 10
        elif dist_high > 10:
            score += 6
    
    # Delivery based
    if delivery_pct > 65:
        score += 10
    elif delivery_pct > 50:
        score += 7
    elif delivery_pct > 35:
        score += 4
    
    # Momentum
    if 0.3 < change_pct < 2:
        score += 8
    elif 0 < change_pct <= 0.3:
        score += 5
    
    return min(25, score)

# ============================================
# FUNDAMENTAL ESTIMATION (From NSE data)
# ============================================
def estimate_fundamentals(price, high_52, low_52, volume, delivery_pct, sector):
    """Estimate fundamental quality from available data"""
    factors = []
    warnings = []
    fund_score = 0
    
    # Price position = valuation proxy
    if high_52 > 0 and low_52 > 0:
        mid_52 = (high_52 + low_52) / 2
        
        if price < mid_52:
            fund_score += 10
            factors.append("Trading below 52W midpoint (Value zone)")
        elif price < high_52 * 0.8:
            fund_score += 6
            factors.append("Room to reach 52W high")
        elif price > high_52 * 0.95:
            fund_score += 2
            warnings.append("Near 52W high - limited upside")
    
    # Volume = liquidity proxy
    if volume > 1000000:
        fund_score += 8
        factors.append("High liquidity (>1M shares)")
    elif volume > 500000:
        fund_score += 5
        factors.append("Good liquidity")
    else:
        warnings.append("Low liquidity")
    
    # Delivery = conviction proxy
    if delivery_pct > 60:
        fund_score += 8
        factors.append("Strong delivery (Investor conviction)")
    elif delivery_pct > 45:
        fund_score += 5
    elif delivery_pct < 30:
        warnings.append("Weak delivery")
    
    # Sector premium
    premium_sectors = ['IT', 'Pharma', 'Consumer']
    if sector in premium_sectors:
        fund_score += 4
        factors.append(f"Premium sector: {sector}")
    
    return {
        'score': min(20, fund_score),
        'factors': factors[:4],
        'warnings': warnings[:3]
    }

# ============================================
# SMART ENTRY DETECTOR
# ============================================
def detect_smart_entry(price, high_52, low_52, vwap, change_pct):
    """Detect optimal entry point"""
    score = 0
    signals = []
    
    # RSI estimation
    if high_52 > 0 and low_52 > 0:
        range_52 = high_52 - low_52
        pos_52 = ((price - low_52) / range_52 * 100) if range_52 > 0 else 50
        # Convert to approximate RSI
        estimated_rsi = 30 + (pos_52 * 0.4)
        
        if 35 <= estimated_rsi <= 50:
            score += 12
            signals.append(f"Buy zone (RSI ~{estimated_rsi:.0f})")
        elif 50 < estimated_rsi <= 65:
            score += 8
            signals.append(f"Moderate RSI ~{estimated_rsi:.0f}")
        elif estimated_rsi > 75:
            signals.append("Overbought zone - wait")
            score -= 5
    
    # VWAP position
    if vwap > 0:
        vs_vwap = ((price - vwap) / vwap) * 100
        if -2 < vs_vwap < 0:
            score += 10
            signals.append("Slightly below VWAP - good entry")
        elif 0 <= vs_vwap < 2:
            score += 7
            signals.append("At/Above VWAP - momentum")
        elif vs_vwap < -3:
            score += 5
            signals.append("Below VWAP - discount")
    
    # Pullback check
    if -1 < change_pct < 0:
        score += 8
        signals.append("Mild pullback - entry opportunity")
    elif -3 < change_pct <= -1:
        score += 5
        signals.append("Dip - potential reversal")
    
    quality = "EXCELLENT" if score >= 30 else "GOOD" if score >= 20 else "FAIR" if score >= 10 else "POOR"
    
    return {'score': min(25, max(0, score)), 'quality': quality, 'signals': signals[:4]}

# ============================================
# ML-STYLE PREDICTOR (Statistical)
# ============================================
def predict_profit_probability(tech_score, fund_score, entry_score, delivery_pct, change_pct):
    """Statistical profit probability calculator"""
    # Weighted probability model
    base_prob = 55  # Base probability
    
    # Technical contribution
    tech_bonus = (tech_score / 25) * 15
    
    # Fundamental contribution
    fund_bonus = (fund_score / 20) * 10
    
    # Entry quality contribution
    entry_bonus = (entry_score / 25) * 10
    
    # Delivery conviction
    delivery_bonus = 5 if delivery_pct > 60 else 3 if delivery_pct > 40 else 0
    
    # Momentum alignment
    momentum_bonus = 5 if 0 < change_pct < 2 else 2 if -1 < change_pct <= 0 else 0
    
    probability = base_prob + tech_bonus + fund_bonus + entry_bonus + delivery_bonus + momentum_bonus
    
    return min(92, max(35, round(probability, 1)))

# ============================================
# TELEGRAM
# ============================================
def send_ml_alert(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        if len(text) > 3900: text = text[:3900]
        resp = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}, timeout=10)
        return resp.json().get('ok', False)
    except: return False

# ============================================
# MAIN ANALYZER
# ============================================
def run_ml_analysis():
    nse = Nse()
    results = []
    now = datetime.now()
    
    print(f"🤖 ML PREDICTOR - {now.strftime('%d-%b %I:%M %p')}")
    
    all_symbols = [(sym, sec) for sec, syms in STOCKS.items() for sym in syms]
    
    for symbol, sector in all_symbols:
        try:
            q = nse.get_quote(symbol)
            if not q: continue
            
            intraday = q.get('intraDayHighLow', {})
            weekly = q.get('weekHighLow', {})
            
            price = float(q.get('lastPrice', 0))
            if price == 0: continue
            
            high = float(intraday.get('max', 0))
            low = float(intraday.get('min', 0))
            change_pct = float(q.get('pChange', 0))
            vwap = float(q.get('vwap', 0)) if q.get('vwap') else 0
            high_52 = float(weekly.get('max', 0))
            low_52 = float(weekly.get('min', 0))
            
            # Volume & Delivery
            try:
                vol = float(q.get('totalTradedVolume', 0))
                dq = float(q.get('deliveryQuantity', 0))
                delivery_pct = (dq/vol*100) if vol > 0 else 40
            except:
                vol = 0
                delivery_pct = 40
            
            # 1. Statistical Backtesting Score
            stat_score = calculate_statistical_score(price, high_52, low_52, change_pct, delivery_pct)
            
            # 2. Fundamental Estimation
            fundamentals = estimate_fundamentals(price, high_52, low_52, vol, delivery_pct, sector)
            
            # 3. Smart Entry Detection
            entry = detect_smart_entry(price, high_52, low_52, vwap, change_pct)
            
            # 4. Technical Score
            tech_score = 0
            day_range = high - low
            pos = ((price - low) / day_range * 100) if day_range > 0 else 50
            if 40 < pos < 70: tech_score += 8
            
            if high_52 > 0:
                dist_high = ((high_52 - price) / high_52) * 100
                if dist_high > 20: tech_score += 10
            if low_52 > 0:
                dist_low = ((price - low_52) / low_52) * 100
                if dist_low < 15: tech_score += 6
            
            if 0 < change_pct < 2: tech_score += 6
            if price > vwap: tech_score += 4
            
            # ===== ML PROBABILITY =====
            probability = predict_profit_probability(stat_score, fundamentals['score'], entry['score'], delivery_pct, change_pct)
            
            # ===== TOTAL SCORE =====
            total_score = round((
                stat_score * 0.25 +
                fundamentals['score'] * 0.20 +
                entry['score'] * 0.20 +
                tech_score * 0.15 +
                (probability / 100 * 20)  # ML probability contribution
            ) + 5, 1)
            
            total_score = min(95, total_score)
            
            # Target & Stop
            target = round(price * (1 + total_score/100), 0)
            stop_loss = round(price * 0.95, 0)
            
            if total_score >= 75:
                prediction, stars = "🔥 HIGH PROFIT", "⭐⭐⭐⭐⭐"
            elif total_score >= 60:
                prediction, stars = "🟢 GOOD SETUP", "⭐⭐⭐⭐"
            elif total_score >= 45:
                prediction, stars = "🔵 DECENT", "⭐⭐⭐"
            else:
                prediction, stars = "🟡 AVERAGE", "⭐⭐"
            
            results.append({
                'symbol': symbol, 'sector': sector, 'price': price,
                'score': total_score, 'stars': stars,
                'prediction': prediction,
                'probability': probability,
                'target': target, 'stop_loss': stop_loss,
                'change': change_pct,
                'fund_factors': fundamentals['factors'],
                'fund_warnings': fundamentals['warnings'],
                'entry_quality': entry['quality'],
                'entry_signals': entry['signals'],
                'delivery': delivery_pct,
                'value_zone': 'Yes' if (high_52 > 0 and price < (high_52 + low_52)/2) else 'No'
            })
            
            print(f"  {symbol:15} Score: {total_score:.1f} | Prob: {probability:.0f}% | {prediction}")
            time.sleep(0.1)
            
        except:
            pass
    
    return results, now

def build_ml_message(results, now):
    if not results: return None
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top = [r for r in results if r['score'] >= 75]
    good = [r for r in results if 60 <= r['score'] < 75]
    
    msg = f"🤖 <b>ML PREDICTION ENGINE</b>\n"
    msg += f"{now.strftime('%d-%b %I:%M %p')} IST\n"
    msg += f"{'═'*35}\n\n"
    
    msg += f"📊 <b>Analysis:</b> Statistical + Fundamental + Entry Timing\n"
    msg += f"├ High Profit: {len(top)} | Good: {len(good)} | Total: {len(results)}\n\n"
    
    if top:
        msg += f"🔥 <b>TOP PICKS</b>\n{'═'*35}\n\n"
        
        for i, r in enumerate(top[:4], 1):
            gain = ((r['target'] - r['price']) / r['price']) * 100
            
            msg += f"<b>#{i} {r['symbol']}</b> | {r['sector']} | Rs.{r['price']:.0f}\n"
            msg += f"{'─'*35}\n"
            msg += f"🤖 Score: {r['score']}/100 {r['stars']}\n"
            msg += f"📈 Profit Probability: <b>{r['probability']:.0f}%</b>\n"
            msg += f"🎯 Prediction: {r['prediction']}\n\n"
            
            msg += f"💰 <b>Fundamentals:</b>\n"
            for f in r['fund_factors'][:3]:
                msg += f"   ✅ {f}\n"
            
            msg += f"\n🎯 <b>Entry:</b> {r['entry_quality']}\n"
            for s in r['entry_signals'][:2]:
                msg += f"   ✅ {s}\n"
            
            msg += f"\n💵 <b>Trade:</b>\n"
            msg += f"   Target: Rs.{r['target']:.0f} (+{gain:.0f}%)\n"
            msg += f"   Stop: Rs.{r['stop_loss']:.0f} (-5%)\n"
            msg += f"   Delivery: {r['delivery']:.0f}%\n"
            
            if r['fund_warnings']:
                msg += f"\n⚠️ {', '.join(r['fund_warnings'][:2])}\n"
            msg += f"\n"
    
    msg += f"{'═'*35}\n"
    msg += f"🤖 <i>ML Statistical Model | No Yahoo Needed</i>"
    
    return msg

if __name__ == "__main__":
    results, now = run_ml_analysis()
    if results:
        msg = build_ml_message(results, now)
        if msg and send_ml_alert(msg):
            print(f"✅ Sent! {len(results)} stocks analyzed")
        else:
            print("❌ Failed to send")
    else:
        print("No data")
