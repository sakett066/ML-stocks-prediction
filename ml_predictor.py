"""
ML PREDICTION ENGINE v2.2 - Complete with SL, T1, T2, Holding Period
Multi-layer: Value + Momentum + Smart Money + Risk + Sector
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
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'PERSISTENT', 'LTI'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK', 'BANDHANBNK', 'FEDERALBNK'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'LAURUSLABS', 'ALKEM', 'BIOCON'],
    'Consumer': ['ITC', 'HINDUNILVR', 'TITAN', 'DMART', 'TRENT', 'TATACONSUM', 'DABUR'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'EICHERMOT', 'TVSMOTOR'],
    'Finance': ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 'PFC', 'RECLTD'],
    'Energy': ['RELIANCE', 'POWERGRID', 'NTPC', 'ONGC', 'TATAPOWER', 'ADANIPORTS', 'ADANIGREEN'],
    'Infra': ['LT', 'HAL', 'BEL', 'IRCON', 'RVNL', 'IRCTC'],
    'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDZINC', 'JINDALSTEL', 'VEDL'],
    'Others': ['ZOMATO', 'PIDILITIND', 'BERGEPAINT', 'ASIANPAINT', 'INDIGO']
}

def calculate_value_score(price, high_52, low_52):
    score = 0; signals = []
    if high_52 == 0 or low_52 == 0:
        return {'score': 10, 'signals': ['Limited data'], 'position': 50}
    range_52 = high_52 - low_52
    position = ((price - low_52) / range_52 * 100) if range_52 > 0 else 50
    dist_from_high = ((high_52 - price) / high_52 * 100)
    dist_from_low = ((price - low_52) / low_52 * 100)
    if position < 30: score += 20; signals.append(f"Deep value: {position:.0f}% of 52W")
    elif position < 50: score += 15; signals.append(f"Good value: {position:.0f}% of 52W")
    elif position < 70: score += 10; signals.append(f"Moderate: {position:.0f}% of 52W")
    else: score += 3; signals.append("Near high")
    if dist_from_high > 40: score += 15; signals.append(f"Recovery: {dist_from_high:.0f}% below high")
    elif dist_from_high > 25: score += 10; signals.append(f"Upside: {dist_from_high:.0f}% below high")
    if dist_from_low < 15: score += 12; signals.append(f"Support: +{dist_from_low:.0f}% above low")
    return {'score': min(30, score), 'signals': signals[:4], 'position': position}

def calculate_momentum_score(price, open_p, high, low, prev_close, change_pct, vwap):
    score = 0; signals = []
    day_range = high - low
    position = ((price - low) / day_range * 100) if day_range > 0 else 50
    if position > 65: score += 8; signals.append("Closing near high")
    elif position > 50: score += 5
    elif position < 35: score += 3; signals.append("Near day low - reversal")
    if 0.3 < change_pct < 2: score += 10; signals.append(f"+{change_pct:.1f}% rise")
    elif 0 < change_pct <= 0.3: score += 6; signals.append("Mild positive")
    elif -1 < change_pct < 0: score += 7; signals.append(f"{change_pct:.1f}% dip - opportunity")
    elif -3 < change_pct <= -1: score += 4; signals.append("Pullback")
    if vwap > 0:
        vs_vwap = ((price - vwap) / vwap) * 100
        if 0 < vs_vwap < 1.5: score += 8; signals.append("Above VWAP")
        elif -1 < vs_vwap <= 0: score += 6; signals.append("Near VWAP")
        elif vs_vwap < -2: score += 4; signals.append("Below VWAP")
    if prev_close > 0:
        gap = ((open_p - prev_close) / prev_close) * 100
        if 0 < gap < 1: score += 5; signals.append("Gap up")
    return {'score': min(20, score), 'signals': signals[:4]}

def calculate_smart_money_score(delivery_pct, buy_qty, sell_qty, volume):
    score = 0; signals = []
    if delivery_pct > 65: score += 12; signals.append(f"Delivery: {delivery_pct:.0f}% Strong")
    elif delivery_pct > 50: score += 8; signals.append(f"Delivery: {delivery_pct:.0f}% Good")
    elif delivery_pct > 35: score += 4
    else: signals.append(f"Delivery: {delivery_pct:.0f}% Low")
    if buy_qty > 0 and sell_qty > 0:
        ratio = buy_qty / sell_qty
        if ratio > 1.5: score += 8; signals.append(f"Buy pressure: {ratio:.1f}x")
        elif ratio > 1.1: score += 5
        elif ratio < 0.7: score += 2; signals.append("Sell pressure")
    if volume > 2000000: score += 5; signals.append("High volume")
    elif volume > 500000: score += 3
    return {'score': min(20, score), 'signals': signals[:3]}

def calculate_risk_score(price, high_52, low_52, change_pct, delivery_pct):
    score = 0; warnings = []
    if high_52 > 0 and low_52 > 0:
        volatility = ((high_52 - low_52) / low_52) * 100
        if volatility < 30: score += 8
        elif volatility < 50: score += 5
        else: warnings.append(f"High vol: {volatility:.0f}%")
    if abs(change_pct) > 5: warnings.append(f"Extreme: {change_pct:+.1f}%"); score -= 3
    if delivery_pct < 25: warnings.append("Low delivery"); score -= 5
    return {'score': max(0, min(15, score + 5)), 'warnings': warnings[:3]}

def calculate_sector_score(sector):
    scores = {'IT':8,'Banking':7,'Pharma':8,'Consumer':7,'Auto':6,'Finance':7,'Energy':6,'Infra':7,'Metals':5,'Others':5}
    return scores.get(sector, 5)

def calculate_win_probability(value, momentum, smart_money, risk, sector, delivery_pct, change_pct):
    prob = 50
    prob += (value['score'] / 30) * 12
    prob += (momentum['score'] / 20) * 10
    prob += (smart_money['score'] / 20) * 8
    prob -= (15 - risk['score']) * 0.5
    prob += sector * 0.5
    if delivery_pct > 60: prob += 5
    elif delivery_pct < 30: prob -= 3
    if 0.3 < change_pct < 2: prob += 3
    elif -1 < change_pct < 0: prob += 2
    return min(92, max(25, round(prob, 1)))

def send_ml_alert(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        if len(text) > 3900: text = text[:3900]
        resp = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}, timeout=10)
        return resp.json().get('ok', False)
    except: return False

def run_ml_analysis():
    nse = Nse()
    results = []
    now = datetime.now()
    
    print(f"ML Predictor v2.2 - {now.strftime('%d-%b %I:%M %p')}")
    
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
            
            try:
                vol = float(q.get('totalTradedVolume', 0))
                dq = float(q.get('deliveryQuantity', 0))
                delivery_pct = (dq/vol*100) if vol > 0 else 40
                buy_qty = float(q.get('totalBuyQuantity', 0))
                sell_qty = float(q.get('totalSellQuantity', 0))
            except:
                vol = 0; delivery_pct = 40; buy_qty = 0; sell_qty = 0
            
            value = calculate_value_score(price, high_52, low_52)
            momentum = calculate_momentum_score(price, open_p, high, low, prev_close, change_pct, vwap)
            smart_money = calculate_smart_money_score(delivery_pct, buy_qty, sell_qty, vol)
            risk = calculate_risk_score(price, high_52, low_52, change_pct, delivery_pct)
            sector_score = calculate_sector_score(sector)
            
            probability = calculate_win_probability(value, momentum, smart_money, risk, sector_score, delivery_pct, change_pct)
            
            total_score = round(
                value['score'] * 0.25 +
                momentum['score'] * 0.20 +
                smart_money['score'] * 0.15 +
                risk['score'] * 0.10 +
                sector_score * 0.05 +
                (probability / 100) * 25
            , 1)
            
            total_score = total_score + 10
            total_score = min(95, max(15, total_score))
            
            stop_loss = round(price * 0.95, 0)
            
            # T1, T2 & Holding Period based on score
            if total_score >= 55:
                t1 = round(price * 1.20, 0); t2 = round(price * 1.40, 0)
                hold = "2-4 weeks"; position = "15-20%"
                prediction = "HIGH CONVICTION"; stars = "⭐⭐⭐⭐⭐"
            elif total_score >= 42:
                t1 = round(price * 1.12, 0); t2 = round(price * 1.22, 0)
                hold = "1-3 weeks"; position = "10-15%"
                prediction = "STRONG BUY"; stars = "⭐⭐⭐⭐"
            elif total_score >= 32:
                t1 = round(price * 1.07, 0); t2 = round(price * 1.14, 0)
                hold = "1-2 weeks"; position = "5-10%"
                prediction = "MODERATE BUY"; stars = "⭐⭐⭐"
            else:
                t1 = round(price * 1.04, 0); t2 = round(price * 1.08, 0)
                hold = "3-7 days"; position = "0-5%"
                prediction = "WATCH"; stars = "⭐⭐"
            
            t1_gain = round(((t1 - price) / price) * 100, 1)
            t2_gain = round(((t2 - price) / price) * 100, 1)
            
            results.append({
                'symbol': symbol, 'sector': sector, 'price': price,
                'score': total_score, 'stars': stars,
                'prediction': prediction, 'probability': probability,
                'stop_loss': stop_loss, 't1': t1, 't2': t2,
                't1_gain': t1_gain, 't2_gain': t2_gain,
                'hold': hold, 'position': position,
                'change': change_pct,
                'value_signals': value['signals'][:2],
                'momentum_signals': momentum['signals'][:2],
                'smart_signals': smart_money['signals'][:2],
                'risk_warnings': risk['warnings'][:2],
                'delivery': delivery_pct,
                'value_zone': f"{value['position']:.0f}% of 52W"
            })
            
            print(f"  {symbol:15} Score: {total_score:.1f} | T1:+{t1_gain}% | T2:+{t2_gain}% | {hold}")
            time.sleep(0.08)
            
        except:
            pass
    
    return results, now

def build_message(results, now):
    if not results: return None
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    top = [r for r in results if r['score'] >= 55]
    good = [r for r in results if 42 <= r['score'] < 55]
    moderate = [r for r in results if 32 <= r['score'] < 42]
    
    msg = f"<b>ML PREDICTION ENGINE v2.2</b>\n"
    msg += f"{now.strftime('%d-%b %I:%M %p')} IST\n"
    msg += f"{'═'*35}\n\n"
    
    msg += f"<b>Results:</b>\n"
    msg += f"High Conviction: {len(top)}\n"
    msg += f"Strong Buy: {len(good)}\n"
    msg += f"Moderate: {len(moderate)}\n"
    msg += f"Total: {len(results)} stocks\n\n"
    
    # Show all three categories
    all_shown = top[:4] + good[:2] + moderate[:1]
    
    if all_shown:
        msg += f"<b>TOP PICKS</b>\n{'═'*35}\n\n"
        
        for i, r in enumerate(all_shown[:6], 1):
            emoji = "🟢" if r['score'] >= 55 else "🔵" if r['score'] >= 42 else "🟡"
            
            msg += f"{emoji} <b>#{i} {r['symbol']}</b> | {r['sector']} | Rs.{r['price']:.0f}\n"
            msg += f"{'─'*35}\n"
            msg += f"Score: <b>{r['score']}/100</b> {r['stars']} | {r['prediction']}\n"
            msg += f"Win Probability: <b>{r['probability']:.0f}%</b>\n\n"
            
            msg += f"<b>Analysis:</b>\n"
            if r['value_signals']: msg += f"  Value: {', '.join(r['value_signals'])}\n"
            if r['momentum_signals']: msg += f"  Momentum: {', '.join(r['momentum_signals'])}\n"
            if r['smart_signals']: msg += f"  Smart $: {', '.join(r['smart_signals'])}\n"
            msg += f"  Zone: {r['value_zone']}\n"
            if r['risk_warnings']: msg += f"  Risk: {', '.join(r['risk_warnings'])}\n"
            
            msg += f"\n<b>Trade Plan:</b>\n"
            msg += f"  Entry: Rs.{r['price']:.0f}\n"
            msg += f"  SL: Rs.{r['stop_loss']:.0f} (-5%)\n"
            msg += f"  T1: Rs.{r['t1']:.0f} (+{r['t1_gain']}%) | T2: Rs.{r['t2']:.0f} (+{r['t2_gain']}%)\n"
            msg += f"  Hold: {r['hold']} | Position: {r['position']}\n"
            msg += f"  Delivery: {r['delivery']:.0f}%\n\n"
    
    msg += f"{'═'*35}\n"
    msg += f"<i>ML Engine v2.2 | SL + T1 + T2 + Hold Period</i>"
    
    return msg

if __name__ == "__main__":
    results, now = run_ml_analysis()
    if results:
        msg = build_message(results, now)
        if msg and send_ml_alert(msg):
            top_count = len([r for r in results if r['score'] >= 55])
            good_count = len([r for r in results if 42 <= r['score'] < 55])
            print(f"Sent! {len(results)} stocks | Top:{top_count} | Good:{good_count}")
        else:
            print("Failed to send")
    else:
        print("No data")
