# Local run — Brave + IPVanish + DeepSeek

## 1. Host
- Connect **IPVanish** (system VPN). Confirm IP at https://ipinfo.io
- Install **Brave**
- Python 3.11+

## 2. Setup
```bash
git clone https://github.com/alistra742-source/cs2.git
cd cs2
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Env
```bash
# Windows cmd
set CHROME_PATH=C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe
set PROXY_MODE=off
set TOR_FALLBACK=0
set HEADLESS=0
set PORT=8080

# DeepSeek USER token (from chat.deepseek.com — NOT platform API key)
set DEEPSEEK_USER_TOKEN=paste_token_here
```

Get token:
1. Log in https://chat.deepseek.com
2. F12 → Application → Local Storage → userToken → copy the `value` string
   OR Network → any /api/v0/ request → authorization header (Bearer ...)


## 4. Start
```bash
python app.py
```
Dashboard: http://127.0.0.1:8080 → Start

## Flow
1. Brave opens Discord register (headed)
2. Form fill
3. hCaptcha checkbox → challenge
4. Accessibility path + DeepSeek answer
5. Submit → continue toward token / mail verify
