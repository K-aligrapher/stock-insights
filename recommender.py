def recommendation(price_trend, sentiment):
    if price_trend > 0 and sentiment["positive"] > sentiment["negative"]:
        return "BUY ✅"
    elif price_trend < 0 and sentiment["negative"] > sentiment["positive"]:
        return "SELL ❌"
    else:
        return "HOLD ⚖️"
