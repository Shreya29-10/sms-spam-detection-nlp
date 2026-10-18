import re

URGENT = ["urgent", "immediately", "hurry", "act now", "call now", "limited time",
          "expires", "last chance", "final notice", "asap", "blocked",
          "suspended", "within 24 hours", "today only"]
PRIZE = ["free", "win", "won", "winner", "prize", "reward", "lottery", "bonus",
         "gift", "congratulations", "selected", "offer", "discount", "cashback"]
CTA = ["click", "claim", "reply", "verify", "login", "subscribe", "download",
       "call now", "text"]

def _find_words(text, words):
    low = text.lower()
    return [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", low)]

def analyze_signals(text):
    """Returns a list of dicts: name, found (bool), detail (str)."""
    text = str(text)
    out = []

    urls = re.findall(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|in|info|biz|xyz|co|ly|link)\b\S*)",
                      text, flags=re.I)
    out.append({"name": "Link / URL", "found": bool(urls),
                "detail": ", ".join(urls[:2])})

    phone = re.search(r"(?:\+?\d[\s-]?){10,}", text)
    out.append({"name": "Phone number", "found": bool(phone),
                "detail": phone.group().strip() if phone else ""})

    money = re.findall(r"[£$€₹]\s?\d[\d,]*|\b\d[\d,]*\s?(?:rs|inr|usd|cash|pounds|dollars)\b",
                       text, flags=re.I)
    out.append({"name": "Money / currency", "found": bool(money),
                "detail": ", ".join(money[:3])})

    prize = _find_words(text, PRIZE)
    out.append({"name": "Prize / free-offer words", "found": bool(prize),
                "detail": ", ".join(prize[:4])})

    urgent = _find_words(text, URGENT)
    out.append({"name": "Urgency words", "found": bool(urgent),
                "detail": ", ".join(urgent[:4])})

    cta = _find_words(text, CTA)
    out.append({"name": "Call-to-action", "found": bool(cta),
                "detail": ", ".join(cta[:4])})

    caps = re.findall(r"\b[A-Z]{3,}\b", text)
    out.append({"name": "ALL-CAPS words", "found": len(caps) >= 2,
                "detail": ", ".join(caps[:4])})

    n_excl = text.count("!")
    out.append({"name": "Many exclamation marks", "found": n_excl >= 2,
                "detail": f"{n_excl} found"})

    return out