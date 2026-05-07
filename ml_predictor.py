"""
ML PREDICTION ENGINE v2.0 - ULTIMATE VERSION
Multi-layer: Statistical + Pattern + Momentum + Value + Risk
Best possible prediction without Yahoo Finance
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

# Extended universe with growth stocks
STOCKS = {
    '🖥️ IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'PERSISTENT', 'LTI'],
    '🏦 Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK', 'BANDHANBNK', 'FEDERALBNK'],
    '💊 Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LAURUSLABS', 'ALKEM', 'BIOCON'],
    '🛒 Consumer': ['ITC', 'HINDUNILVR', 'TITAN', 'DMART', 'TRENT', 'TATACONSUM', 'DABUR'],
    '🚗 Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'TVSMOTOR'],
    '💰 Finance': ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 'PFC', 'RECLTD'],
    '⚡ Energy': ['RELIANCE', 'POWERGRID', 'NTPC', 'ONGC', 'TATAPOWER', 'ADANIPORTS', 'ADANIGREEN'],
    '🏗️ Infra': ['LT', 'HAL', 'BEL', 'IRCON', 'RVNL', 'IRCTC'],
    '🏭 Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDZINC', 'JINDALSTEL', 'VEDL'],
    '📱 Others': ['ZOMATO', 'PIDILITIND', 'BERGEPAINT', 'ASIANPAINT', 'INDIGO']
}

# ============================================
# LAYER 1: VALUE DETECTION
# ============================================
def calculate_value_score(price, high_52, low_52):
    """Detect if stock is at good value"""
    score = 0
    signals = []
    
    if high_52 == 0 or low_52 == 0:
        return {'score': 10, 'signals': ['Limited data']}
    
    range_52 = high_52 - low_52
    position = ((price - low_52) / range_52 * 100) if range_52 > 0 else 50
    dist_from_high = ((high_52 - price) / high_52 * 100)
    dist_from_low = ((price - low_52) / low_52 * 100)
    
    # Value zones
    if position < 30:
        score += 20
        signals.append(f"Deep value: {position:.0f}% of 52W range")
    elif position < 50:
        score += 15
        signals.append(f"Good value: {position:.0f}% of 52W range")
    elif position < 70:
        score += 10
        signals.append(f"Moderate: {position:.0f}% of 52W range")
    else:
        score += 3
        signals.append("Near high - limited value")
    
    # Distance from high
    if dist_from_high > 40:
        score += 15
        signals.append(f"Recovery potential: {dist_from_high:.0f}% below high")
    elif dist_from_high > 25:
        score += 10
        signals.append(f"Upside room: {dist_from_high:.0f}% below high")
    
    # Distance from low (safety margin)
    if dist_from_low < 15:
        score += 12
        signals.append(f"Strong support nearby: {dist_from_low:.0f}% above low")
    
    return {'score': min(30, score), 'signals': signals[:4], 'position': position}

# ============================================
# LAYER 2: MOMENTUM & TREND
# ============================================
def calculate_momentum_score(price, open_p, high, low, prev_close, change_pct, vwap):
    """Multi-factor momentum detection"""
    score = 0
    signals = []
    
    day_range = high - low
    position = ((price - low) / day_range * 100) if day_range > 0 else 50
    
    # Intraday strength
    if position > 65:
        score += 8
        signals.append("Closing near high - strength")
    elif position > 50:
        score += 5
    elif position < 35:
        score += 3
        signals.append("Near day low - potential reversal")
    
    # Price change quality
    if 0.3 < change_pct < 2:
        score += 10
        signals.append(f"Healthy rise: +{change_pct:.1f}%")
    elif 0 < change_pct <= 0.3:
        score += 6
        signals.append("Mild positive")
    elif -1 < change_pct < 0:
        score += 7
        signals.append(f"Dip: {change_pct:.1f}% - buy opportunity")
    elif -3 < change_pct <= -1:
        score += 4
        signals.append("Pullback - watch for reversal")
    
    # VWAP relationship
    if vwap > 0:
        vs_vwap = ((price - vwap) / vwap) * 100
        if 0 < vs_vwap < 1.5:
            score += 8
            signals.append("Above VWAP - bullish")
        elif -1 < vs_vwap <= 0:
            score += 6
            signals.append("Near VWAP - fair value")
        elif vs_vwap < -2:
            score += 4
            signals.append("Below VWAP - oversold")
    
    # Gap analysis
    if prev_close > 0:
        gap = ((open_p - prev_close) / prev_close) * 100
        if 0 < gap < 1:
            score += 5
            signals.append("Gap up with room")
        elif gap > 2:
            score += 2
            signals.append("Large gap - wait for fill")
    
    return {'score': min(20, score), 'signals': signals[:4]}

# ============================================
# LAYER 3: SMART MONEY
# ============================================
def calculate_smart_money_score(delivery_pct, buy_qty, sell_qty, volume):
    """Detect institutional activity"""
    score = 0
    signals = []
    
    # Delivery analysis
    if delivery_pct > 65:
        score += 12
        signals.append(f"Strong delivery: {delivery_pct:.0f}% - accumulation")
    elif delivery_pct > 50:
        score += 8
        signals.append(f"Good delivery: {delivery_pct:.0f}%")
    elif delivery_pct > 35:
        score += 4
    else:
        signals.append(f"Low delivery: {delivery_pct:.0f}% - speculative")
    
    # Buy/Sell ratio
    if buy_qty > 0 and sell_qty > 0:
        ratio = buy_qty / sell_qty
        if ratio > 1.5:
            score += 8
            signals.append(f"Buying pressure: {ratio:.1f}x")
        elif ratio > 1.1:
            score += 5
        elif ratio < 0.7:
            score += 2
            signals.append("Selling pressure")
    
    # Volume significance
    if volume > 2000000:
        score += 5
        signals.append("High volume activity")
    elif volume > 500000:
        score += 3
    
    return {'score': min(20, score), 'signals': signals[:3]}

# ============================================
# LAYER 4: RISK ASSESSMENT
# ============================================
def calculate_risk_score(price, high_52, low_52, change_pct, delivery_pct):
    """Assess downside risk"""
    score = 0
    warnings = []
    
    # Volatility risk
    if high_52 > 0 and low_52 > 0:
        volatility = ((high_52 - low_52) / low_52) * 100
        if volatility < 30:
            score += 8
        elif volatility < 50:
            score += 5
        else:
            warnings.append(f"High volatility: {volatility:.0f}%")
    
    # Gap risk
    if abs(change_pct) > 5:
        warnings.append(f"Extreme move: {change_pct:+.1f}%")
        score -= 3
    
    # Delivery risk
    if delivery_pct < 25:
        warnings.append("Very low delivery - high risk")
        score -= 5
    
    # Near circuit
    if change_pct > 8:
        warnings.append("Near upper circuit - reversal risk")
        score -= 4
    elif change_pct < -8:
        warnings.append("Near lower circuit - falling knife")
        score -= 4
    
    return {'score': max(0, min(15, score + 5)), 'warnings': warnings[:3]}

# ============================================
# LAYER 5: SECTOR STRENGTH
# ============================================
def calculate_sector_score(sector):
    """Score based on sector category"""
    sector_scores = {
        '🖥️ IT': 8,
        '🏦 Banking': 7,
        '💊 Pharma': 8,
        '🛒 Consumer': 7,
        '🚗 Auto': 6,
        '💰 Finance': 7,
        '⚡ Energy': 6,
        '🏗️ Infra': 7,
        '🏭 Metals': 5,
        '📱 Others': 5
    }
    return sector_scores.get(sector, 5)

# ============================================
# PREDICTION ENGINE
# ============================================
def calculate_win_probability(value, momentum, smart_money, risk, sector, delivery_pct, change_pct):
    """Calculate probability of profitable trade"""
    # Base probability
    prob = 50
    
    # Value contribution
    prob += (value['score'] / 30) * 12
    
    # Momentum contribution
    prob += (momentum['score'] / 20) * 10
    
    # Smart money contribution
    prob += (smart_money['score'] / 20) * 8
    
    # Risk reduction
    prob -= (15 - risk['score']) * 0.5
    
    # Sector bonus
    prob += sector * 0.5
    
    # Delivery conviction
    if delivery_pct > 60: prob += 5
    elif delivery_pct < 30: prob -= 3
    
    # Momentum alignment
    if 0.3 < change_pct < 2: prob += 3
    elif -1 < change_pct < 0: prob += 2
    
    return min(92, max(25, round(prob, 1)))

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
    
    print(f"🤖 ULTIMATE ML PREDICTOR - {now.strftime('%d-%b %I:%M %p')}")
    
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
            open_p = float(q.get('open', 0))
            change_pct = float(q.get('pChange', 0))
            prev_close = float(q.get('previousClose', 0))
            vwap = float(q.get('vwap', 0)) if q.get('vwap') else 0
            high_52 = float(weekly.get('max', 0))
            low_52 = float(weekly.get('min', 0))
            
            # Volume data
            try:
                vol = float(q.get('totalTradedVolume', 0))
                dq = float(q.get('deliveryQuantity', 0))
                delivery_pct = (dq/vol*100) if vol > 0 else 40
                buy_qty = float(q.get('totalBuyQuantity', 0))
                sell_qty = float(q.get('totalSellQuantity', 0))
            except:
                vol = 0; delivery_pct = 40; buy_qty = 0; sell_qty = 0
            
            # ALL 5 LAYERS
            value = calculate_value_score(price, high_52, low_52)
            momentum = calculate_momentum_score(price, open_p, high, low, prev_close, change_pct, vwap)
            smart_money = calculate_smart_money_score(delivery_pct, buy_qty, sell_qty, vol)
            risk = calculate_risk_score(price, high_52, low_52, change_pct, delivery_pct)
            sector_score = calculate_sector_score(sector)
            
            # Win Probability
            probability = calculate_win_probability(value, momentum, smart_money, risk, sector_score, delivery_pct, change_pct)
            
            # Total Score
            total_score = round(
                value['score'] * 0.25 +
                momentum['score'] * 0.20 +
                smart_money['score'] * 0.15 +
                risk['score'] * 0.10 +
                sector_score * 0.05 +
                (probability / 100) * 25
            , 1)
            
            total_score = min(95, max(10, total_score))
            
            # Target & Stop
            target_mult = 1.8 if total_score >= 75 else 1.5 if total_score >= 60 else 1.3
            target = round(price * target_mult, 0)
            stop_loss = round(price * 0.95, 0)
            
            # Rating
            if total_score >= 75:
                prediction = "🔥 HIGH CONVICTION"
                stars = "⭐⭐⭐⭐⭐"
                position = "15-20%"
            elif total_score >= 60:
                prediction = "🟢 STRONG BUY"
                stars = "⭐⭐⭐⭐"
                position = "10-15%"
            elif total_score >= 45:
                prediction = "🔵 MODERATE BUY"
                stars = "⭐⭐⭐"
                position = "5-10%"
            else:
                prediction = "🟡 WATCH"
                stars = "⭐⭐"
                position = "0-5%"
            
            results.append({
                'symbol': symbol, 'sector': sector, 'price': price,
                'score': total_score, 'stars': stars,
                'prediction': prediction,
                'probability': probability,
                'target': target, 'stop_loss': stop_loss,
                'position': position,
                'change': change_pct,
                'value_signals': value['signals'][:2],
                'momentum_signals': momentum['signals'][:2],
                'smart_signals': smart_money['signals'][:2],
                'risk_warnings': risk['warnings'][:2],
                'delivery': delivery_pct,
                'value_zone': f"{value['position']:.0f}% of 52W"
            })
            
            print(f"  {symbol:15} Score: {total_score:.1f} | Prob: {probability:.0f}% | {prediction}")
            time.sleep(0.08)
            
        except:
            pass
    
    return results, now

def build_message(results, now):
    if not results: return None
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top = [r for r in results if r['score'] >= 75]
    good = [r for r in results if 60 <= r['score'] < 75]
    
    msg = f"🤖 <b>ML PREDICTION ENGINE v2.0</b>\n"
    msg += f"{now.strftime('%d-%b %I:%M %p')} IST\n"
    msg += f"{'═'*35}\n\n"
    
    msg += f"📊 <b>5-Layer Analysis:</b>\n"
    msg += f"├ Value | Momentum | Smart Money\n"
    msg += f"├ Risk Assessment | Sector Strength\n"
    msg += f"├ Win Probability: Statistical Model\n"
    msg += f"├ High Conviction: {len(top)} stocks\n"
    msg += f"└ Strong Buy: {len(good)} stocks\n\n"
    
    if top:
        msg += f"🔥 <b>HIGH CONVICTION PICKS</b>\n{'═'*35}\n\n"
        
        for i, r in enumerate(top[:5], 1):
            gain = ((r['target'] - r['price']) / r['price']) * 100
            
            msg += f"<b>#{i} {r['symbol']}</b> | {r['sector']} | Rs.{r['price']:.0f}\n"
            msg += f"{'─'*35}\n"
            msg += f"🤖 Score: <b>{r['score']}/100</b> {r['stars']}\n"
            msg += f"📈 Win Probability: <b>{r['probability']:.0f}%</b>\n"
            msg += f"🎯 {r['prediction']}\n\n"
            
            msg += f"📊 <b>Layer Analysis:</b>\n"
            msg += f"   💎 Value: {', '.join(r['value_signals'])}\n"
            msg += f"   ⚡ Momentum: {', '.join(r['momentum_signals'])}\n"
            msg += f"   💰 Smart Money: {', '.join(r['smart_signals'])}\n"
            msg += f"   📍 Zone: {r['value_zone']}\n"
            
            if r['risk_warnings']:
                msg += f"   ⚠️ Risk: {', '.join(r['risk_warnings'])}\n"
            
            msg += f"\n💵 <b>Trade Plan:</b>\n"
            msg += f"   Target: Rs.{r['target']:.0f} (+{gain:.0f}%)\n"
            msg += f"   Stop: Rs.{r['stop_loss']:.0f} (-5%)\n"
            msg += f"   Position: {r['position']}\n"
            msg += f"   Delivery: {r['delivery']:.0f}%\n\n"
    
    msg += f"{'═'*35}\n"
    msg += f"🤖 <i>ML Engine v2.0 | 5-Layer Analysis</i>"
    
    return msg

if __name__ == "__main__":
    results, now = run_ml_analysis()
    if results:
        msg = build_message(results, now)
        if msg and send_ml_alert(msg):
            print(f"✅ Sent! {len(results)} stocks analyzed")
        else:
            print("❌ Failed")
    else:
        print("No data")
