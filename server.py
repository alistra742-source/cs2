import asyncio
import base64
import gc
import json
import os
import random
import re
import socket
import struct
import time
from typing import Optional

from browser_engine import async_playwright, ENGINE

from captcha_solver import (
    extract_hcaptcha_sitekey,
    extract_rqdata_from_body,
    read_hcaptcha_token,
)
from duckmail import TempMail
from vision_solver import OllamaVisionClient
from drag_solver import DragSolver
import hcaptcha_types as hct
import human_mouse as hm

# Mean per-tile CNN confidence required before the OFFLINE tile classifier
# is trusted for a grid round (below this gate the vision model answers).
_CNN_MIN_CONF = float(os.environ.get("SOLVER_CNN_MIN_CONF", "0.62"))


# ── Shared JS: robust login-link / back-to-login detection ──────────────
# Discord renders the "Already have an account?" control as a REAL
# <button type="submit"> inside the register form, and its label can carry
# non-breaking spaces / split spans — plain substring matching on raw
# textContent fails (that's exactly how runs end up on /login). This helper
# normalizes ALL whitespace (incl. \u00a0) across textContent + aria-label +
# title + value and tests the blacklist. Injected INSIDE each evaluate's
# arrow-function body (the engine wraps function-looking strings in parens
# and calls them, so a top-level const would be a syntax error).
_LOGIN_LINK_GUARD = r"""const __isLoginLink = (el) => {
    const raw = (el.textContent || '') + ' ' +
                (el.getAttribute('aria-label') || '') + ' ' +
                (el.getAttribute('title') || '') + ' ' +
                (el.value || '');
    const t = raw.toLowerCase().replace(/\s+/g, ' ').trim();
    // ALL locales: clicking the "Already have an account?" / back-to-login
    // control navigates to /login and silently kills the run, so the guard
    // has to recognize it in whatever language Discord is serving (Swedish
    // "Har du redan ett konto?", French "Déjà un compte ?", German
    // "Bereits ein Konto?", Dutch "Al een account?", Russian
    // "Уже есть аккаунт?"...).
    return /(already|have an account|have account|account\?|log ?in|sign ?in|signin|back to|forgot|login|einloggen|anmelden|logga in|logg inn|log ind|connexion|se connecter|connecte|iniciar sesi|acceder|entrar|conectar|accedi|inloggen|přihlásit|zaloguj|zalogować|войти|вход|войдите|로그인|ログイン|登录|登入|đăng nhập|giriş yap|kirjaudu|åter till|tillbaka|terug|retour|zurück|volver|indietro|tilbage|tilbake)/.test(t);
};"""

# Shared JS regex: the register form's submit button label in EVERY locale
# (German "Konto erstellen", French "Créer un compte", Spanish "Crear
# cuenta", Russian "Создать аккаунт", Korean "가입"...). Injected into the
# button-click evaluates with __SUBMIT_TEXT_RE__ so the Create Account click
# works no matter what language Discord serves.
_SUBMIT_TEXT_RE = (
    "create account|create an account|sign up|signup|continue|"
    "registrieren|konto erstellen|erstelle konto|créer un compte|s'inscrire|"
    "inscription|crear cuenta|registrarse|criar conta|cadastrar|cadastre|"
    "aanmelden|account aanmaken|registrera|skapa konto|opret konto|"
    "opret bruger|załóż konto|zarejestruj|создать аккаунт|зарегистрироваться|"
    "регистрация|tạo tài khoản|đăng ký|가입|회원가입|注册|创建|アカウント作成|登録|"
    "üye ol|kayıt ol|weiter|continuer|continuar|continua|"
    "volgende|fortsätt|fortsett|fortsæt|kontynuuj|продолжить|devam et|"
    "tiếp tục|계속|继续|続ける"
)

# ── Shared JS: find Discord's REQUIRED ToS checkbox (real controls only) ──
# Discord's register form has two checkboxes: the required Terms-of-Service
# agreement and an optional marketing/"email updates" box. Older code
# matched [class*="checkbox"], which ALSO hit styled container divs (double
# toggles) and the marketing box (the "wrong checkbox"). This targets ONLY
# real checkbox controls (native input / role=checkbox / data-state), skips
# the marketing box by its label, and returns the click point of the first
# unchecked ToS box (or null when none remains).
_TOS_TARGET_JS = r"""() => {
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    // The consent checkbox renders as several element types across Discord
    // builds/locales: a native input, a div[role=checkbox], a data-state
    // box, or a plain div with class*='checkbox'. Accept ALL of them;
    // never styled container divs (leaf-ish boxes only).
    const isRealBox = (el) => {
        if (el.tagName === 'INPUT' && el.type === 'checkbox') return true;
        if (el.getAttribute('role') === 'checkbox') return true;
        if (el.getAttribute('data-state')) return true;
        if (el.getAttribute('aria-checked') !== null) return true;
        const cls = (el.className || '').toString().toLowerCase();
        // Accept styled divs whose class signals the consent checkbox
        // (checkbox / agree / terms / tos / consent / accept). Discord's
        // class names are hashed (checkboxWrapper_f73e0c etc.), so any
        // of these signals + a small leaf-ish box is good enough.
        if (cls.includes('checkbox') || cls.includes('check-box')
                || cls.includes('agree') || cls.includes('terms')
                || cls.includes('tos') || cls.includes('consent')
                || cls.includes('accept')) {
            const r = el.getBoundingClientRect();
            return r.width >= 8 && r.height >= 8 && el.children.length <= 3;
        }
        return false;
    };
    const els = [];
    for (const el of document.querySelectorAll(
        'input[type="checkbox"], [role="checkbox"], [data-state], [aria-checked], ' +
        '[class*="checkbox" i], [class*="check-box" i], [class*="agree" i], ' +
        '[class*="terms" i], [class*="tos" i], [class*="consent" i], [class*="accept" i]')) {
        if (isRealBox(el)) els.push(el);
    }
    const candidates = [];
    const allUnchecked = [];
    for (const cb of els) {
        if (cb.checked || cb.getAttribute('aria-checked') === 'true'
            || cb.getAttribute('data-state') === 'checked') continue;
        // The real click target: the box itself when it has size, else the
        // first sized ancestor (the box's visible representation).
        let target = null;
        let r = cb.getBoundingClientRect();
        if (r && r.width >= 5 && r.height >= 5) target = cb;
        if (!target) {
            for (let p = cb.parentElement; p && p !== document.body; p = p.parentElement) {
                const pr = p.getBoundingClientRect();
                if (pr && pr.width >= 8 && pr.height >= 8) { target = p; break; }
            }
        }
        if (!target || target.offsetParent === null) continue;
        // Label text: closest <label>, else the DIRECT row (the box's own
        // row — an ancestor shared with the marketing box would mislabel
        // the ToS box), else the nearest labeled ancestor.
        let label = '';
        try {
            const lab = cb.closest('label');
            if (lab) label = lab.innerText || '';
        } catch (e) {}
        if (!label) {
            try {
                const row = cb.parentElement;
                if (row && row !== document.body) {
                    const rt = norm(row.innerText || '');
                    if (rt.length > 4 && rt.length <= 200) label = rt;
                }
            } catch (e) {}
        }
        if (!label) {
            try {
                let anc = cb.parentElement;
                for (let i = 0; anc && i < 3; i++) {
                    const t = norm(anc.innerText || '');
                    if (t.length > 4 && t.length <= 220) { label = t; break; }
                    anc = anc.parentElement;
                }
            } catch (e) {}
        }
        const lowL = low(label);
        const r2 = target.getBoundingClientRect();
        const entry = {
            x: r2.left + r2.width / 2,
            y: r2.top + r2.height / 2,
            tos: /terms|service|agreement|conditions|villkor|voorwaarden|condiciones|akzeptiere|accedo|aceito|godk|aksoord|conform|akkoord|gelezen|nous avons lu|acepto los t|принимаю|已阅读|同意|nutzungsbedingungen|datenschutzerklärung|datenschutzerklaerung|gelesen|datenschutz/.test(lowL),
            label: lowL,
            el: target
        };
        allUnchecked.push(entry);
        // Skip the optional marketing / email-updates box in ANY locale
        // (Dutch 'e-mails ontvangen'/'aanbiedingen', Swedish 'mejl'/'tips',
        // ...). Only the ToS agreement enables the button.
        if (/mejl|e-post|mail|email|marketing|updat|news|newsletter|promotion|exclusive|offers|subscribe|reklam|tips|erbjudande|aanbieding|optioneel/.test(lowL)) continue;
        candidates.push(entry);
    }
    // If every visible box looked like marketing (label detection failed),
    // fall back to the last unchecked box that does NOT look like marketing,
    // else the very last unchecked one — by layout the ToS box sits below
    // the marketing box.
    if (!candidates.length && allUnchecked.length) {
        const nonMkt = allUnchecked.filter(c => !/mejl|e-post|mail|email|marketing|updat|news|newsletter|promotion|exclusive|offers|subscribe|reklam|tips|erbjudande|aanbieding|optioneel/.test(c.label));
        const pick = nonMkt.length ? nonMkt[nonMkt.length - 1] : allUnchecked[allUnchecked.length - 1];
        candidates.push(pick);
    }
    if (!candidates.length) return null;
    // Prefer the box whose label signals ToS; among the rest prefer the LAST
    // unchecked visible box (the ToS box sits below the optional marketing
    // box).
    const tosOnes = candidates.filter(c => c.tos);
    const rest = candidates.filter(c => !c.tos);
    const ordered = tosOnes.concat(rest.reverse());
    const best = ordered[0];
    try { best.el.setAttribute('data-tos-target', '1'); } catch (e) {}
    // Discord toggles the box when its <label> / the row text is clicked too —
    // clicking the label (which often wraps or neighbours the box) is the most
    // reliable toggle and lands on a larger target. Prefer the label centre,
    // falling back to the box centre.
    let lx = best.x, ly = best.y;
    try {
        const lab = best.el.closest('label') || best.el.parentElement;
        if (lab) {
            const lr = lab.getBoundingClientRect();
            if (lr && lr.width >= 8 && lr.height >= 8) {
                lx = lr.left + lr.width / 2;
                ly = lr.top + lr.height / 2;
            }
        }
    } catch (e) {}
    return { x: best.x, y: best.y, lx: lx, ly: ly, tos: best.tos ? 1 : 0, tag: (best.el && best.el.tagName || '').toLowerCase() };
}"""

# JS-dispatch fallback for the ToS box: dispatches pointer/mouse events ON
# the box element itself, so it works even when a transparent overlay or a
# moving page swallowed the trusted click. Native inputs are additionally
# force-checked (prototype setter + input/change events).
_TOS_CLICK_JS = r"""() => {
    const el = document.querySelector('[data-tos-target]');
    if (!el) return null;
    try { el.scrollIntoView({ block: 'center' }); } catch (e) {}
    for (const type of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click']) {
        el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }));
    }
    if (el.tagName === 'INPUT') {
        try {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'checked').set;
            setter.call(el, true);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
            return 'input_forced';
        } catch (e) {
            return 'dispatched';
        }
    }
    return 'dispatched';
}"""


_TOS_FALLBACK_JS = r"""() => {
    // Position-based fallback: Discord ALWAYS renders the required ToS row
    // directly above the Create Account button. When the standard checkbox
    // selectors find nothing (some layouts render the box as a styled div
    // with no role/data-state), find the submit button and click the
    // box-like element sitting in the row right above it.
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const isBox = (el) => {
        if (!el || el.nodeType !== 1) return false;
        const r = el.getBoundingClientRect();
        if (!r || r.width < 8 || r.height < 8) return false;
        const cls = (el.className || '').toString().toLowerCase();
        const tag = el.tagName.toLowerCase();
        if (tag === 'input' && el.type === 'checkbox') return true;
        if (el.getAttribute('role') === 'checkbox') return true;
        if (el.getAttribute('data-state')) return true;
        if (el.getAttribute('aria-checked') !== null) return true;
        if (cls.includes('checkbox') || cls.includes('checkBox')) return true;
        if (cls.includes('circle') && cls.includes('button')) return true;
        // A square-ish leaf container (Discord's box wrapper)
        if (r.width <= 40 && r.height <= 40 && el.children.length <= 3) {
            const cs = getComputedStyle(el);
            if (cs.borderRadius && cs.cursor === 'pointer') return true;
        }
        return false;
    };
    const btns = Array.from(document.querySelectorAll('button, [role="button"]'));
    const submit = btns.filter(b => b.offsetParent !== null)
        .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top)[0];
    if (!submit) return null;
    const srect = submit.getBoundingClientRect();
    const candidates = [];
    // Scan the whole form for unchecked box-like elements ABOVE the button
    // (within 1.4x the button's height), closest to it first.
    for (const el of document.querySelectorAll('*')) {
        if (!isBox(el)) continue;
        if (el.checked || el.getAttribute('aria-checked') === 'true'
            || el.getAttribute('data-state') === 'checked') continue;
        const r = el.getBoundingClientRect();
        if (r.top >= srect.top || r.bottom <= srect.top - 260) continue;
        candidates.push(el);
    }
    if (!candidates.length) return null;
    candidates.sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top);
    const box = candidates[0];
    // The click target: the box itself if sized, else the nearest sized
    // ancestor (the box's visible representation).
    let target = box;
    let r = box.getBoundingClientRect();
    if (r.width < 5 || r.height < 5) {
        for (let p = box.parentElement; p && p !== document.body; p = p.parentElement) {
            const pr = p.getBoundingClientRect();
            if (pr.width >= 8 && pr.height >= 8) { target = p; r = pr; break; }
        }
    }
    if (!target || target.offsetParent === null) return null;
    try { target.setAttribute('data-tos-target', '1'); } catch (e) {}
    const txt = low(norm(box.closest('div') ? (box.parentElement ? box.parentElement.innerText : '') : ''));
    return { x: r.left + r.width / 2, y: r.top + r.height / 2, tag: (target.tagName || '').toLowerCase(), label: txt.slice(0, 90) };
}"""




# ── Discord rate-limit phrases — rotate the proxy the moment these show ──
# Discord localizes the 429 page to the region's language, so include the
# common spellings (German "zu viele Anfragen", French "trop de requêtes",
# Spanish "demasiadas solicitudes", Russian "слишком много запросов"...).
_RATE_LIMIT_KEYWORDS = (
    "the resource is being rate limited",
    "resource is being rate limited",
    "you are being rate limited",
    "rate limited",
    "ratelimited",
    "too many requests",
    "slowdown",
    "try again later",
    "zu viele anfragen",
    "trop de requêtes",
    "trop de demandes",
    "demasiadas solicitudes",
    "demasiadas peticiones",
    "muitas solicitações",
    "te veel verzoeken",
    "för många förfrågningar",
    "for mange forespørsler",
    "for mange anmodninger",
    "слишком много запросов",
    "zbyt wiele żądań",
    "çok fazla istek",
    "limite de débit",
)


# ── Discord register-page state ─────────────────────────────────────────
# The navigation poll reads the page through TWO independent channels:
#   1. JS evaluation (page.evaluate) — the engine falls back to the raw CDP
#      websocket when the reattached WebDriver session's JS context is stale
#      (the old white-screen bug: page loaded, title="Discord", but every
#      evaluate() returned None and the bot rotated a healthy session).
#   2. CDP DOM-presence checks (driver.cdp.is_element_present) — no JS
#      execution required at all.
_NAV_STATE_JS = r"""() => {
    const body = document.body;
    if (!body) return JSON.stringify({error: "no-body"});
    const text = body.innerText || "";
    const titleLow = (document.title || "").toLowerCase();
    const challenge = /just a moment|checking your browser|verify you are human|attention required/.test(titleLow + " " + text.toLowerCase().substring(0, 800)) || !!document.querySelector('iframe[src*="challenges.cloudflare.com"], #challenge-stage, #cf-challenge-running, #cf-chl');
    // Broad selectors — Discord uses aria-label, not name
    const email = document.querySelector('input[name="email"], input[type="email"], input[aria-label*="email" i], input[aria-label*="Email"], input[id*="email" i]');
    const username = document.querySelector('input[name="username"], input[aria-label*="username" i], input[aria-label*="display" i]');
    const password = document.querySelector('input[name="password"], input[type="password"], input[aria-label*="password" i]');
    // Age-gate + login detection is locale-agnostic: Discord localizes the
    // register page to the proxy region, so accept the common spellings
    // (Dutch "geboortedatum", French "date de naissance", German
    // "Geburtsdatum", Swedish "födelsedatum", Russian "дата рождения",
    // Korean "생년월일"...).
    const hasAge = /birthday|date of birth|born|how old|geboortedatum|date de naissance|geburtsdatum|fecha de nacimiento|data di nascita|data de nascimento|födelsedatum|fødselsdato|fødselsdato|data urodzenia|дата рождения|datum narození|doğum tarihi|tanggal lahir|생년월일|生年月日|出生日期/i.test(text.substring(0, 400));
    const hasMonth = document.querySelector('[class*="month" i], [aria-label*="month" i], [class*="maand" i], [class*="mois" i], [class*="monat" i], select');
    const isLogin = /login|sign in|welcome back|anmelden|einloggen|logga in|logg inn|log ind|connexion|se connecter|iniciar sesi|acceder|entrar|conectar|accedi|inloggen|přihlásit|zaloguj|войти|вход|로그인|ログイン|登录|đăng nhập|giriş yap|kirjaudu/i.test(text.substring(0, 400));
    const hasQR = document.querySelector('img[src*="qr" i], [class*="qr" i]');
    const continueBtn = document.querySelector('button[type="submit"], button[class*="continue" i]');
    return JSON.stringify({
        url: location.href,
        title: document.title || "",
        readyState: document.readyState || "",
        email: email !== null,
        username: username !== null,
        password: password !== null,
        ageGate: hasAge || hasMonth !== null,
        isLogin: isLogin,
        hasQR: hasQR,
        hasButton: continueBtn !== null,
        hasAppMount: document.querySelector("#app-mount") !== null,
        inputCount: document.querySelectorAll("input").length,
        buttonCount: document.querySelectorAll("button").length,
        cfClearance: document.cookie.indexOf("cf_clearance=") !== -1,
        challenge: challenge,
        textPreview: text.substring(0, 250)
    });
}"""

# CDP DOM-presence selectors used when JS evaluation is unavailable.
_CDP_NAV_SELECTORS = {
    "email": 'input[name="email"], input[type="email"], input[aria-label*="email" i], input[id*="email" i]',
    "username": 'input[name="username"], input[aria-label*="username" i], input[aria-label*="display" i]',
    "password": 'input[name="password"], input[type="password"], input[aria-label*="password" i]',
    "hasAppMount": "#app-mount",
    "challenge": 'iframe[src*="challenges.cloudflare.com"], #challenge-stage, #cf-challenge-running',
}


# ── Full-form readiness probe ─────────────────────────────────────────
# _goto_register() returns as soon as email+username (or the age gate) paint,
# but the SPA keeps hydrating: password, the three DOB dropdowns, ToS and the
# Continue button can appear a beat later. Filling before they exist — and
# before React has attached its value trackers — is what produced runs where
# the bot typed into a half-rendered page and the form ended up empty. This
# probe is the "is it actually all there yet?" gate _wait_for_form_ready polls.
_FORM_READY_JS = r"""() => {
    const vis = (el) => !!el && (el.offsetParent !== null || el.getClientRects().length > 0);
    const q = (sel) => document.querySelector(sel);
    const email = q('input[name="email"], input[type="email"], input[autocomplete="email"], input[aria-label*="email" i], input[placeholder*="email" i], input[id*="email" i]');
    const username = q('input[name="username"], input[autocomplete="username"], input[aria-label*="username" i], input[id*="username" i], input[placeholder*="username" i]');
    const password = q('input[name="password"], input[type="password"], input[autocomplete="new-password"], input[aria-label*="password" i]');
    // DOB controls: native <select> or React-Select combobox/container whose
    // label/placeholder/text mentions month/day/year in the page's locale
    // (Dutch "Dag/Maand/Jaar", French "Jour/Mois/Année", ...).
    const DOB_LABELS = __DOB_LABELS__;
    const seen = {};
    const controls = Array.from(document.querySelectorAll(
        'select, [role="combobox"], [role="listbox"], [role="button"], [class*="select" i], [class*="dropdown" i], [class*="control" i]'
    ));
    for (const el of controls) {
        if (!vis(el)) continue;
        const cls = typeof el.className === 'string' ? el.className : '';
        const acc = (cls + ' ' + (el.getAttribute('aria-label') || '') + ' ' +
                     (el.getAttribute('name') || '') + ' ' + (el.getAttribute('id') || '') + ' ' +
                     (el.getAttribute('placeholder') || '') + ' ' +
                     ((el.textContent || '').slice(0, 80))).toLowerCase();
        for (const key of Object.keys(DOB_LABELS)) {
            for (const al of DOB_LABELS[key]) {
                if (new RegExp('(^|[^a-z0-9])' + al + '([^a-z0-9]|$)').test(acc)) { seen[key] = true; break; }
            }
            if (seen[key]) break;
        }
    }
    const body = document.body ? document.body.innerText : '';
    return JSON.stringify({
        email: vis(email),
        username: vis(username),
        password: vis(password),
        dob: Object.keys(seen).length,
        dobText: /date of birth|birthday|geboortedatum|date de naissance|geburtsdatum|fecha de nacimiento|data di nascita|data de nascimento|födelsedatum|fødselsdato|data urodzenia|дата рождения|datum narození|doğum tarihi|tanggal lahir|생년월일|生年月日|出生日期/i.test(body),
        inputs: document.querySelectorAll('input').length,
        buttons: document.querySelectorAll('button').length,
        readyState: document.readyState || '',
    });
}"""


# Robust DOB (Month/Day/Year) setter. Discord's DOB control has changed
# across builds: native <select>, a React-Select combobox, or a custom div.
# This targets the control BY LABEL (aria-label / name / id / placeholder /
# class) and sets the matching option directly — never "first N inputs",
# never tab-roulette, never typing into whatever happens to have focus.
_DOB_FALLBACK_JS = r"""
async () => {
    const LABEL = __LABEL__;
    const OPT = __OPT__;
    const DOB_LABELS = __DOB_LABELS__;
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const monthIndex = ['january','february','march','april','may','june',
        'july','august','september','october','november','december']
        .indexOf(low(OPT)) + 1;
    // Localized month names (Dutch "januari", "maart", French "janvier",
    // "mars", ...) resolve to their numeric index so the option match works
    // in whatever locale Discord is serving.
    const MONTH_ALIASES = {
        'januari':1,'janvier':1,'januar':1,'enero':1,'gennaio':1,'styczeń':1,
        'январь':1,'січень':1,'януари':1,'tammikuu':1,'jaanuar':1,'janvāris':1,
        'sausis':1,'Ιανουάριος':1,'ocak':1,'január':1,'ianuarie':1,'يناير':1,
        'जनवरी':1,'1월':1,'1月':1,'มกราคม':1,
        'februari':2,'février':2,'fevrier':2,'februar':2,'febrero':2,'febbraio':2,
        'luty':2,'февраль':2,'лютий':2,'февруари':2,'helmikuu':2,'veebruar':2,
        'februāris':2,'vasaris':2,'Φεβρουάριος':2,'şubat':2,'február':2,'februarie':2,
        'فبراير':2,'फरवरी':2,'2월':2,'2月':2,'กุมภาพันธ์':2,
        'maart':3,'mars':3,'märz':3,'marts':3,'marzo':3,'março':3,'marzec':3,
        'март':3,'березень':3,'maaliskuu':3,'märts':3,'kovas':3,'Μάρτιος':3,
        'mart':3,'március':3,'martie':3,'مارس':3,'मार्च':3,'3월':3,'3月':3,'มีนาคม':3,
        'april':4,'avril':4,'abril':4,'kwiecień':4,'апрель':4,'квітень':4,
        'април':4,'huhtikuu':4,'aprill':4,'aprīlis':4,'balandis':4,'Απρίλιος':4,
        'nisan':4,'április':4,'aprilie':4,'أبريل':4,'अप्रैल':4,'4월':4,'4月':4,'เมษายน':4,
        'mei':5,'mai':5,'mayo':5,'maggio':5,'maj':5,'май':5,'травень':5,
        'toukokuu':5,'maijs':5,'gegužė':5,'Μάιος':5,'mayıs':5,'május':5,
        'مايو':5,'मई':5,'5월':5,'5月':5,'พฤษภาคม':5,
        'juni':6,'juin':6,'junio':6,'giugno':6,'июнь':6,'червень':6,'юни':6,
        'kesäkuu':6,'juuni':6,'jūnijs':6,'birželis':6,'Ιούνιος':6,'haziran':6,
        'június':6,'iunie':6,'يونيو':6,'जून':6,'6월':6,'6月':6,'มิถุนายน':6,
        'juli':7,'juillet':7,'julio':7,'luglio':7,'июль':7,'липень':7,'юли':7,
        'heinäkuu':7,'juuli':7,'jūlijs':7,'liepa':7,'Ιούλιος':7,'temmuz':7,
        'július':7,'iulie':7,'يوليو':7,'जुलाई':7,'7월':7,'7月':7,'กรกฎาคม':7,
        'augustus':8,'augusti':8,'august':8,'août':8,'aout':8,'agosto':8,
        'август':8,'серпень':8,'elokuu':8,'augusts':8,'rugpjūtis':8,'Αύγουστος':8,
        'ağustos':8,'augusztus':8,'أغسطس':8,'अगस्त':8,'8월':8,'8月':8,'สิงหาคม':8,
        'september':9,'septembre':9,'septiembre':9,'settembre':9,'сентябрь':9,
        'вересень':9,'септември':9,'syyskuu':9,'septembris':9,'rugsėjis':9,
        'Σεπτέμβριος':9,'eylül':9,'szeptember':9,'septembrie':9,'سبتمبر':9,
        'सितंबर':9,'9월':9,'9月':9,'กันยายน':9,
        'oktober':10,'octobre':10,'octubre':10,'ottobre':10,'октябрь':10,
        'жовтень':10,'октомври':10,'lokakuu':10,'oktoober':10,'oktobris':10,
        'spalis':10,'Οκτώβριος':10,'ekim':10,'október':10,'octombrie':10,'أكتوبر':10,
        'अक्टूबर':10,'10월':10,'10月':10,'ตุลาคม':10,
        'november':11,'novembre':11,'noviembre':11,'ноябрь':11,'листопад':11,
        'ноември':11,'marraskuu':11,'novembris':11,'lapkritis':11,'Νοέμβριος':11,
        'kasım':11,'noiembrie':11,'نوفمبر':11,'नवंबर':11,'11월':11,'11月':11,'พฤศจิกายน':11,
        'december':12,'décembre':12,'dezember':12,'diciembre':12,'dicembre':12,
        'desember':12,'декабрь':12,'грудень':12,'декември':12,'joulukuu':12,
        'detsember':12,'decembris':12,'gruodis':12,'Δεκέμβριος':12,'aralık':12,
        'decembrie':12,'ديسمبر':12,'दिसंबर':12,'12월':12,'12月':12,'ธันวาคม':12,
    };
    const wantNum = monthIndex || MONTH_ALIASES[low(OPT)] || (parseInt(OPT, 10) || 0);
    const wantStr = low(OPT);
    const optionMatches = (text, value) => {
        const t = low(text || ''); const v = low(value || '');
        if (!t && !v) return false;
        if (t === wantStr || v === wantStr) return true;
        if (MONTH_ALIASES[t] && MONTH_ALIASES[t] === wantNum) return true;
        if (!wantNum) return false;
        const n = String(wantNum);
        const p = n.length === 1 ? '0' + n : n;
        const toks = t.split(/[^a-z0-9]+/).filter(Boolean);
        return t === n || v === n || t === p || v === p || toks.indexOf(n) !== -1 || toks.indexOf(p) !== -1;
    };
    const labelHits = (el) => {
        const cls = (typeof el.className === 'string') ? el.className : '';
        const acc = norm(el.getAttribute('aria-label') || '') + ' ' +
                    norm(el.getAttribute('name') || '') + ' ' +
                    norm(el.getAttribute('id') || '') + ' ' +
                    norm(el.getAttribute('placeholder') || '') + ' ' +
                    norm(el.getAttribute('data-label') || '') + ' ' +
                    norm(cls);
        const a = low(acc);
        const labels = DOB_LABELS[LABEL] || [LABEL.toLowerCase()];
        for (const al of labels) {
            if (new RegExp('(^|[^a-z0-9])' + al + '([^a-z0-9]|$)').test(a)) return true;
        }
        return false;
    };
    let candidates = Array.from(document.querySelectorAll(
        'select, [role="combobox"], [role="listbox"], [class*="select" i], [class*="dropdown" i], [class*="control" i]'
    )).filter(labelHits);
    if (!candidates.length) {
        if (!document.body) return 'not_found';
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
        let node;
        const labels = DOB_LABELS[LABEL] || [LABEL.toLowerCase()];
        const re = new RegExp('(^|[^a-z0-9])(' + labels.join('|') + ')([^a-z0-9]|$)');
        while ((node = walker.nextNode())) {
            if (!re.test(low(norm(node.textContent)))) continue;
            let p = node.parentElement;
            for (let i = 0; p && i < 5; i++) {
                if (p.matches && p.matches('select, [role="combobox"], [class*="select" i], [class*="dropdown" i], [class*="control" i]')) {
                    candidates.push(p);
                    break;
                }
                p = p.parentElement;
            }
            if (candidates.length) break;
        }
    }
    for (const el of candidates) {
        const tag = el.tagName.toLowerCase();
        if (tag !== 'select' && el.offsetParent === null) continue;
        if (tag === 'select') {
            for (const opt of Array.from(el.options || [])) {
                if (optionMatches(opt.text || opt.label, opt.value)) {
                    el.value = opt.value;
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    return 'native:' + (opt.text || opt.value);
                }
            }
            continue;
        }
        el.scrollIntoView({ block: 'center' });
        el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
        el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        for (let attempt = 0; attempt < 8; attempt++) {
            if (attempt > 0) await new Promise(r => setTimeout(r, 250));
            const opts = document.querySelectorAll('[role="option"], [id*="option" i], [class*="option" i], ul li');
            for (const opt of opts) {
                const t = norm(opt.textContent || opt.getAttribute('aria-label') || '');
                if (!t) continue;
                if (optionMatches(t, opt.getAttribute('data-value') || opt.getAttribute('value') || t)) {
                    opt.scrollIntoView({ block: 'nearest' });
                    opt.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
                    opt.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
                    opt.dispatchEvent(new MouseEvent('click', { bubbles: true }));
                    return 'combo:' + t;
                }
            }
        }
    }
    return 'not_found';
}
"""

# Locale-aware DOB dropdown labels — Discord localizes the register form to
# the proxy's region ("Dag/Maand/Jaar", "Jour/Mois/Année", "Tag/Monat/Jahr",
# "день/месяц/год"...), so label matching accepts the common spellings across
# ALL languages, not just English. These feed _FORM_READY_JS, _DOB_LOCATE_JS
# and _DOB_FALLBACK_JS (via json.dumps), so the filler works on any form
# Discord serves.
# ── Credential field selectors (shared by fill AND verify) ──
# name-first; the username selector deliberately EXCLUDES
# input[autocomplete="username"]: Discord's email box can carry
# autocomplete="username", and querySelector resolves in document order, so
# that term made the username READ resolve to the EMAIL field — the
# "username reads back the email value" verify abort that killed runs
# before DOB was ever attempted (fields were actually filled; the read
# was lying). Fill and verify MUST use the same strings so they can never
# drift apart again.
_CRED_FIELD_SELECTORS = {
    "email": "input[name='email'], input[type='email'], input[autocomplete='email'], input[aria-label*='email' i], input[id*='email' i]",
    "display": "input[name='global_name'], input[aria-label*='display name' i], input[aria-label*='display' i]",
    "username": "input[name='username'], input[id*='username' i], input[aria-label*='username' i], input[placeholder*='username' i]",
    "password": "input[name='password'], input[type='password'], input[autocomplete='new-password'], input[aria-label*='password' i]",
}

_DOB_LABEL_ALIASES = {
    "Month": [
        "month", "maand", "mois", "monat", "mes", "mês", "mese",
        "miesiąc", "månad", "måned", "měsíc", "mesiac", "месяц", "місяць",
        "месец", "kuukausi", "kuu", "mēnesis", "mėnuo", "μήνας", "ay",
        "hónap", "lună", "bulan", "tháng", "월", "月", "شهر", "महीना", "เดือน",
    ],
    "Day": [
        "day", "dag", "jour", "tag", "día", "dia", "giorno", "dzień",
        "deň", "den", "день", "ден", "päivä", "päev", "diena", "ημέρα",
        "gün", "nap", "zi", "hari", "ngày", "일", "日", "يوم", "दिन", "วัน",
    ],
    "Year": [
        "year", "jaar", "année", "annee", "jahr", "año", "ano",
        "anno", "rok", "år", "год", "рік", "година", "vuosi", "aasta",
        "gads", "metai", "έτος", "yıl", "év", "tahun", "năm", "년", "年",
        "سنة", "साल", "ปี",
    ],
}
# Localized month names → numeric index, so the English month names the bot
# generates ("January"...) can be matched against the localized options
# Discord renders in ANY language (Dutch "Januari", French "janvier", German
# "März", Russian "март", Korean "3월", ...).
_MONTH_ALIASES = {
    "januari": 1, "janvier": 1, "januar": 1, "enero": 1, "gennaio": 1,
    "styczeń": 1, "январь": 1, "січень": 1, "януари": 1, "tammikuu": 1,
    "jaanuar": 1, "janvāris": 1, "sausis": 1, "Ιανουάριος": 1, "ocak": 1,
    "január": 1, "ianuarie": 1, "يناير": 1, "जनवरी": 1, "1월": 1, "1月": 1,
    "มกราคม": 1,
    "februari": 2, "février": 2, "fevrier": 2, "februar": 2, "febrero": 2, "febbraio": 2,
    "luty": 2, "февраль": 2, "лютий": 2, "февруари": 2, "helmikuu": 2,
    "veebruar": 2, "februāris": 2, "vasaris": 2, "Φεβρουάριος": 2, "şubat": 2,
    "február": 2, "februarie": 2, "فبراير": 2, "फरवरी": 2, "2월": 2, "2月": 2,
    "กุมภาพันธ์": 2,
    "maart": 3, "mars": 3, "märz": 3, "marts": 3, "marzo": 3, "março": 3, "marzec": 3,
    "март": 3, "березень": 3, "maaliskuu": 3, "märts": 3, "kovas": 3,
    "Μάρτιος": 3, "mart": 3, "március": 3, "martie": 3, "مارس": 3, "मार्च": 3,
    "3월": 3, "3月": 3, "มีนาคม": 3,
    "april": 4, "avril": 4, "abril": 4, "kwiecień": 4, "апрель": 4, "квітень": 4,
    "април": 4, "huhtikuu": 4, "aprill": 4, "aprīlis": 4, "balandis": 4,
    "Απρίλιος": 4, "nisan": 4, "április": 4, "aprilie": 4, "أبريل": 4, "अप्रैल": 4,
    "4월": 4, "4月": 4, "เมษายน": 4,
    "mei": 5, "mai": 5, "mayo": 5, "maggio": 5, "maj": 5,
    "май": 5, "травень": 5, "toukokuu": 5, "maijs": 5, "gegužė": 5,
    "Μάιος": 5, "mayıs": 5, "május": 5, "مايو": 5, "मई": 5,
    "5월": 5, "5月": 5, "พฤษภาคม": 5,
    "juni": 6, "juin": 6, "junio": 6, "giugno": 6,
    "июнь": 6, "червень": 6, "юни": 6, "kesäkuu": 6, "juuni": 6, "jūnijs": 6,
    "birželis": 6, "Ιούνιος": 6, "haziran": 6, "június": 6, "iunie": 6,
    "يونيو": 6, "जून": 6, "6월": 6, "6月": 6, "มิถุนายน": 6,
    "juli": 7, "juillet": 7, "julio": 7, "luglio": 7,
    "июль": 7, "липень": 7, "юли": 7, "heinäkuu": 7, "juuli": 7, "jūlijs": 7,
    "liepa": 7, "Ιούλιος": 7, "temmuz": 7, "július": 7, "iulie": 7,
    "يوليو": 7, "जुलाई": 7, "7월": 7, "7月": 7, "กรกฎาคม": 7,
    "augustus": 8, "augusti": 8, "august": 8, "août": 8, "aout": 8, "agosto": 8,
    "август": 8, "серпень": 8, "elokuu": 8, "augusts": 8, "rugpjūtis": 8,
    "Αύγουστος": 8, "ağustos": 8, "augusztus": 8, "أغسطس": 8, "अगस्त": 8,
    "8월": 8, "8月": 8, "สิงหาคม": 8,
    "september": 9, "septembre": 9, "septiembre": 9, "settembre": 9,
    "сентябрь": 9, "вересень": 9, "септември": 9, "syyskuu": 9, "septembris": 9,
    "rugsėjis": 9, "Σεπτέμβριος": 9, "eylül": 9, "szeptember": 9, "septembrie": 9,
    "سبتمبر": 9, "सितंबर": 9, "9월": 9, "9月": 9, "กันยายน": 9,
    "oktober": 10, "octobre": 10, "octubre": 10, "ottobre": 10,
    "октябрь": 10, "жовтень": 10, "октомври": 10, "lokakuu": 10, "oktoober": 10,
    "oktobris": 10, "spalis": 10, "Οκτώβριος": 10, "ekim": 10, "október": 10,
    "octombrie": 10, "أكتوبر": 10, "अक्टूबर": 10, "10월": 10, "10月": 10, "ตุลาคม": 10,
    "november": 11, "novembre": 11, "noviembre": 11,
    "ноябрь": 11, "листопад": 11, "ноември": 11, "marraskuu": 11, "novembris": 11,
    "lapkritis": 11, "Νοέμβριος": 11, "kasım": 11, "noiembrie": 11,
    "نوفمبر": 11, "नवंबर": 11, "11월": 11, "11月": 11, "พฤศจิกายน": 11,
    "december": 12, "décembre": 12, "dezember": 12, "diciembre": 12, "dicembre": 12,
    "desember": 12, "декабрь": 12, "грудень": 12, "декември": 12, "joulukuu": 12,
    "detsember": 12, "decembris": 12, "gruodis": 12, "Δεκέμβριος": 12, "aralık": 12,
    "decembrie": 12, "ديسمبر": 12, "दिसंबर": 12, "12월": 12, "12月": 12, "ธันวาคม": 12,
}

_MONTHS_EN = ("january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december")


def _month_index(name: str) -> int:
    """Numeric month index for an English or localized month name (0 = not a month)."""
    n = (name or "").strip().lower()
    if n in _MONTHS_EN:
        return _MONTHS_EN.index(n) + 1
    return _MONTH_ALIASES.get(n, 0)


def _dob_text_matches(text: str, option_text: str) -> bool:
    """True when a DOB control's current text represents `option_text` in the
    page's locale (e.g. the Dutch 'Januari' matches the English 'January').

    Discord renders the control's visible button text as 'value, value'
    (label + value duplicated, e.g. 'January, January'), so matching is
    TOKEN-based: the option text - or its localized/numeric equivalent -
    must appear as one whitespace/comma-separated token in the control text.
    """
    import re as _re
    t = (text or "").strip().lower()
    o = (option_text or "").strip().lower()
    if not t or not o:
        return False
    tokens = [tok for tok in _re.split(r"[^a-z0-9]+", t) if tok]
    if o in tokens:
        return True
    want = _month_index(o)
    if want:
        return any(_month_index(tok) == want for tok in tokens)
    return any(tok.lstrip("0") == o.lstrip("0") for tok in tokens if tok.isdigit())


# Locate a DOB dropdown control by its localized label ("Day"/"Month"/"Year"
# with the locale alias table — Dutch "Dag/Maand/Jaar", French
# "Jour/Mois/Année", ...). Scans control-like elements (role=button,
# combobox, select/dropdown/control classes, native select) so page body
# copy can never be mistaken for a label, with a text-walker fallback for
# controls that carry none of those markers. Picks the DEEPEST match — the
# individual control, never the DOB group container that holds all three
# labels. Marks the element with data-dob-target so the caller can drive it
# with trusted Playwright clicks. Also accepts the selected VALUE text
# (alias-aware) so a control whose placeholder was replaced by the value
# ("Januari" instead of "Maand") is still found for verification.
# Read the visible text of a DOB control by its LOCALIZED combobox
# aria-label (e.g. 'Month'/'Maand'/'Monat'/'mois'...). The data-dob-target
# marker can land on the field label after a React re-render (the label
# text like 'Month*' matches too), so the post-fill verify reads the
# combobox's select-field text directly as a fallback.
_DOB_VALUE_JS = r"""([label, aliases]) => {
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const labels = (aliases && aliases[label]) || [label.toLowerCase()];
    const re = new RegExp('(^|[^a-z0-9])(' + labels.join('|') + ')([^a-z0-9]|$)');
    for (const t of document.querySelectorAll('[role="combobox"]')) {
        const aria = norm(t.getAttribute('aria-label') || '');
        if (!re.test(aria)) continue;
        const wrap = t.closest('[class*="selectField" i]') || t.parentElement;
        if (!wrap) continue;
        const txt = norm(wrap.innerText || '');
        const lines = txt.split('\n').map(norm).filter(Boolean);
        return lines.length ? lines[lines.length - 1] : txt;
    }
    return '';
}"""

_DOB_LOCATE_JS = r"""([label, aliases, valueText, monthAliases]) => {
    try { document.querySelectorAll('[data-dob-target="' + label + '"]').forEach(e => e.removeAttribute('data-dob-target')); } catch(e) {}
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const labels = (aliases && aliases[label]) || [label.toLowerCase()];
    const re = new RegExp('(^|[^a-z0-9])(' + labels.join('|') + ')([^a-z0-9]|$)');
    const MONTHS = ['january','february','march','april','may','june','july',
        'august','september','october','november','december'];
    const want = valueText ? low(valueText) : null;
    const wantNum = want ? ((MONTHS.indexOf(want) + 1) || (monthAliases && monthAliases[want]) || (parseInt(valueText, 10) || 0)) : 0;
    // Discord renders the control text as 'value, value' (label + value
    // duplicated), so match by TOKEN, not the whole string.
    const valueHits = (t) => {
        if (!want || !t) return false;
        const toks = String(t).split(/[^a-z0-9]+/).filter(Boolean);
        if (toks.indexOf(want) !== -1) return true;
        if (wantNum) {
            const n = String(wantNum);
            const p = n.length === 1 ? '0' + n : n;
            for (const tok of toks) {
                if ((monthAliases && monthAliases[tok]) === wantNum) return true;
                if (MONTHS.indexOf(tok) + 1 === wantNum) return true;
                if (tok === n || tok === p) return true;
            }
        }
        return false;
    };
    const hits = [];
    const scan = document.querySelectorAll(
        '[role="button"], [role="combobox"], [class*="select" i], [class*="dropdown" i], [class*="control" i], select'
    );
    for (const el of scan) {
        if (!el.offsetParent) continue;
        // Skip zero-size a11y targets: Discord's combobox has a hidden
        // focusTarget div (role=combobox) with NO size - Playwright
        // refuses to click it. The visible selectButton div is the real
        // click target.
        const _r = el.getBoundingClientRect();
        if (!_r || _r.width < 5 || _r.height < 5) continue;
        const acc = low(norm((el.getAttribute('aria-label') || '') + ' ' +
                             (el.getAttribute('placeholder') || '') + ' ' +
                             (el.getAttribute('data-label') || '') + ' ' +
                             (el.textContent || '').slice(0, 80)));
        // Group containers span all three DOB controls ('Month, Month
        // Day, Day Year, Year'); an individual control shows ONE short
        // value/label. Cap the matched text so the container can never
        // be picked as the deepest 'match' (its text contains every
        // label, and depth counts ancestors, so it always sorted first).
        const _tt = norm(el.textContent || '');
        if (_tt.length > 40) continue;
        if (re.test(acc) || valueHits(low(_tt.slice(0, 80)))) {
            let depth = 0;
            let p = el.parentElement;
            while (p) { depth++; p = p.parentElement; }
            hits.push({ el: el, depth: depth });
        }
    }
    // Text-walker fallback: controls with none of the role/class markers
    // (their placeholder text still identifies them).
    if (!hits.length) {
        if (!document.body) return null;
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
        let node;
        while ((node = walker.nextNode())) {
            const t = low(norm(node.textContent));
            if (!re.test(t)) continue;
            let p = node.parentElement;
            for (let i = 0; p && i < 6; i++) {
                if (p.offsetParent !== null && (p.textContent || '').trim().length <= 40) {
                    const _pr = p.getBoundingClientRect();
                    if (_pr && _pr.width >= 5 && _pr.height >= 5) {
                        hits.push({ el: p, depth: 0 });
                        break;
                    }
                }
                p = p.parentElement;
            }
            if (hits.length) break;
        }
    }
    if (!hits.length) return null;
    hits.sort((a, b) => b.depth - a.depth);
    const target = hits[0].el;
    target.setAttribute('data-dob-target', label);
    return { tag: target.tagName.toLowerCase(), depth: hits[0].depth };
}"""

# Options of an open DOB menu (custom dropdowns and native <select>).
_DOB_OPTION_SEL = '[role="listbox"] [role="option"], [role="menu"] [role="option"], [class*="menu" i] [role="option"], [class*="popout" i] [role="option"], [class*="menu" i] [class*="option" i], [class*="popout" i] [class*="option" i], [role="option"], [id*="option" i], [class*="option" i], option, li, [role="menuitem"]'

# Scoped "did THIS control's menu actually open?" check. The old global
# check (any visible [class*=menu]/[class*=option] anywhere on the page)
# false-positived on the Month/Day menus' leftovers and on unrelated page
# chrome, so the Year step could proceed to pick an option from a menu that
# was never open and then fail every pick method. This requires a visible
# option/menu element that sits in the SAME ROW as the marked control and
# opens directly above or below it — i.e. the control's own menu.
_DOB_MENU_OPEN_JS = r"""(label) => {
    const ctrl = document.querySelector('[data-dob-target="' + label + '"]');
    if (!ctrl) return false;
    const cr = ctrl.getBoundingClientRect();
    if (!cr.width || !cr.height) return false;
    let n = 0;
    const cands = document.querySelectorAll(
        '[role="option"], [role="menuitem"], [class*="option" i], ' +
        '[class*="menu" i], [class*="listbox" i], [class*="popout" i]');
    for (const e of cands) {
        if (e === ctrl) continue;
        if (e.offsetParent !== null && e.getClientRects().length === 0) continue;
        const r = e.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        // Same horizontal band as the control (menus drop under it).
        const sameRow = (r.left < cr.right + 250) && (r.right > cr.left - 250);
        // Opening just below OR just above the control's edge.
        const adjacent =
            (r.top >= cr.bottom - 12 && r.top <= cr.bottom + 260) ||
            (r.bottom <= cr.top + 12 && r.bottom >= cr.top - 260);
        if (sameRow && adjacent) n++;
    }
    // >=2 so a single stray fragment can't count as an open menu.
    return n >= 2;
}"""

# Find the index (within _DOB_OPTION_SEL) of the option that represents
# `optionText` in the page's locale. Months resolve to their numeric index so
# the English "January" matches the Dutch "Januari" / French "janvier" / ...
# options Discord renders. Returns -1 when the menu isn't open or nothing
# matches.
_DOB_OPTION_INDEX_JS = r"""([optionText, monthAliases]) => {
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const MONTHS = ['january','february','march','april','may','june','july',
        'august','september','october','november','december'];
    const wantStr = low(optionText);
    const wantNum = (MONTHS.indexOf(wantStr) + 1) || (monthAliases && monthAliases[wantStr]) || (parseInt(optionText, 10) || 0);
    const matches = (t, v) => {
        const a = low(t || ''); const b = low(v || '');
        if (!a && !b) return false;
        if (a === wantStr || b === wantStr) return true;
        if (wantNum && (monthAliases && monthAliases[a] === wantNum || MONTHS.indexOf(a) + 1 === wantNum)) return true;
        if (wantNum) {
            const n = String(wantNum);
            const p = n.length === 1 ? '0' + n : n;
            const toksA = a.split(/[^a-z0-9]+/).filter(Boolean);
            const toksB = b.split(/[^a-z0-9]+/).filter(Boolean);
            if (toksA.some(tok => tok === n || tok === p) || toksB.some(tok => tok === n || tok === p)) return true;
        }
        return false;
    };
    const sel = __OPT_SEL__;
    const opts = Array.from(document.querySelectorAll(sel));
    let visIdx = 0;
    for (const el of opts) {
        // hidden li/option elements from other menus must NOT shift the
        // index - count visible options only.
        if (el.offsetParent === null) continue;
        const t = norm(el.textContent || el.getAttribute('aria-label') || '');
        const v = el.getAttribute('data-value') || el.getAttribute('value') || t;
        if (matches(t, v)) return visIdx;
        visIdx++;
    }
    return -1;
}"""

# Coordinates fallback for option selection: any visible element (leaf-ish)
# whose text represents `optionText` in the page's locale. Handles menus whose
# options carry none of the usual role/class markers. Returns viewport center
# coords for a trusted page.mouse.click.
_DOB_OPTION_POS_JS = r"""([optionText, monthAliases]) => {
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const MONTHS = ['january','february','march','april','may','june','july',
        'august','september','october','november','december'];
    const wantStr = low(optionText);
    const wantNum = (MONTHS.indexOf(wantStr) + 1) || (monthAliases && monthAliases[wantStr]) || (parseInt(optionText, 10) || 0);
    const matches = (t, v) => {
        const a = low(t || ''); const b = low(v || '');
        if (!a && !b) return false;
        if (a === wantStr || b === wantStr) return true;
        if (wantNum && (monthAliases && monthAliases[a] === wantNum || MONTHS.indexOf(a) + 1 === wantNum)) return true;
        if (wantNum) {
            const n = String(wantNum);
            const p = n.length === 1 ? '0' + n : n;
            const toksA = a.split(/[^a-z0-9]+/).filter(Boolean);
            const toksB = b.split(/[^a-z0-9]+/).filter(Boolean);
            if (toksA.some(tok => tok === n || tok === p) || toksB.some(tok => tok === n || tok === p)) return true;
        }
        return false;
    };
    const sel = '[role="listbox"] [role="option"], [role="menu"] [role="option"], [class*="menu" i] [role="option"], [class*="popout" i] [role="option"], [class*="menu" i] [class*="option" i], [class*="popout" i] [class*="option" i], [role="option"], [id*="option" i], [class*="option" i], [role="menuitem"], li';
    const all = document.querySelectorAll(sel);
    for (const el of all) {
        if (!el.offsetParent && el.getClientRects().length === 0) continue;
        const t = norm(el.textContent || el.getAttribute('aria-label') || '');
        const v = el.getAttribute('data-value') || el.getAttribute('value') || t;
        if (matches(t, v)) {
            // Bring the option fully inside the VIEWPORT, not just the menu
            // listbox: the Year menu drops past the 720px viewport bottom
            // and a 'nearest' scroll cannot fix that (the page has no room
            // to scroll). Center it, then correct with an explicit window
            // scroll if the page itself couldn't reach. Re-measure after.
            try { el.scrollIntoView({ block: 'center' }); } catch (e) {}
            let r = el.getBoundingClientRect();
            const vh = window.innerHeight || document.documentElement.clientHeight || 0;
            if (vh && r && (r.top < 0 || r.bottom > vh)) {
                const delta = (r.top < 0) ? (r.top - 24) : (r.bottom - vh + 24);
                try { window.scrollBy(0, delta); } catch (e) {}
                r = el.getBoundingClientRect();
            }
            if (!r || r.width < 3 || r.height < 3) continue;
            const vh2 = window.innerHeight || document.documentElement.clientHeight || 0;
            if (vh2 && (r.top < 0 || r.bottom > vh2 + 2)) continue; // still unreachable
            return { x: r.left + r.width / 2, y: r.top + r.height / 2, text: t.slice(0, 30) };
        }
    }
    return null;
}"""

# JS-dispatch fallback for option selection: same matcher as
# _DOB_OPTION_POS_JS but dispatches pointer/mouse events ON the option
# element itself, so it works even when the menu is covered by a transparent
# overlay (the events go to the target node, not the overlay).
_DOB_OPTION_DISPATCH_JS = r"""([optionText, monthAliases]) => {
    const norm = (s) => (s == null ? '' : String(s)).replace(/\s+/g, ' ').trim();
    const low = (s) => norm(s).toLowerCase();
    const MONTHS = ['january','february','march','april','may','june','july',
        'august','september','october','november','december'];
    const wantStr = low(optionText);
    const wantNum = (MONTHS.indexOf(wantStr) + 1) || (monthAliases && monthAliases[wantStr]) || (parseInt(optionText, 10) || 0);
    const matches = (t, v) => {
        const a = low(t || ''); const b = low(v || '');
        if (!a && !b) return false;
        if (a === wantStr || b === wantStr) return true;
        if (wantNum && (monthAliases && monthAliases[a] === wantNum || MONTHS.indexOf(a) + 1 === wantNum)) return true;
        if (wantNum) {
            const n = String(wantNum);
            const p = n.length === 1 ? '0' + n : n;
            const toksA = a.split(/[^a-z0-9]+/).filter(Boolean);
            const toksB = b.split(/[^a-z0-9]+/).filter(Boolean);
            if (toksA.some(tok => tok === n || tok === p) || toksB.some(tok => tok === n || tok === p)) return true;
        }
        return false;
    };
    const sel = '[role="listbox"] [role="option"], [role="menu"] [role="option"], [class*="menu" i] [role="option"], [class*="popout" i] [role="option"], [class*="menu" i] [class*="option" i], [class*="popout" i] [class*="option" i], [role="option"], [id*="option" i], [class*="option" i], [role="menuitem"], li';
    const all = document.querySelectorAll(sel);
    for (const el of all) {
        if (!el.offsetParent && el.getClientRects().length === 0) continue;
        const t = norm(el.textContent || el.getAttribute('aria-label') || '');
        const v = el.getAttribute('data-value') || el.getAttribute('value') || t;
        if (matches(t, v)) {
            try { el.scrollIntoView({ block: 'nearest' }); } catch (e) {}
            for (const type of ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click']) {
                el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }));
            }
            return t.slice(0, 30);
        }
    }
    return null;
}"""


# React-safe value write: native prototype setter (REPLACES the whole value —
# never appends to whatever is already in the field) + React value-tracker
# sync + real input/change events. Element-targeted: it writes to the resolved
# element directly and NEVER depends on focus or the global keyboard, so a
# stray keystroke can never land in another field. The old
# click + Control+A + press_sequentially fallback typed into WHATEVER held
# focus (Discord's register page keeps focus on the first input, the email
# box) — that is exactly how the username ended up concatenated inside the
# email field while the username input stayed empty.


def _human_typing_delay(ch: str) -> float:
    """Per-character typing delay (seconds) that mimics a real typist.

    Uppercase / symbols take longer (shift reach, then release), digits
    a touch slower than lowercase, and everything has jitter. Averages
    ~70ms per lowercase char — a fast human typist, not a machine gun
    and not a hunt-and-pecker.
    """
    if ch.isupper() or not ch.isascii():
        return random.uniform(0.09, 0.22)
    if ch.isdigit():
        return random.uniform(0.06, 0.16)
    if ch in "!@#$%&*_-.+":
        return random.uniform(0.10, 0.24)
    if ch.islower():
        return random.uniform(0.045, 0.13)
    return random.uniform(0.05, 0.15)


_REACT_SET_VALUE_JS = r"""([sel, value]) => {
    const el = document.querySelector(sel);
    if (!el) return false;
    const proto = (el instanceof HTMLTextAreaElement) ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = (Object.getOwnPropertyDescriptor(proto, 'value') || {}).set
                || (Object.getOwnPropertyDescriptor(el.__proto__, 'value') || {}).set;
    if (setter) {
        if (el._valueTracker && typeof el._valueTracker.setValue === 'function') {
            el._valueTracker.setValue(value === '' ? '__initial__' : '');
        }
        setter.call(el, value);
    } else {
        el.value = value;
    }
    el.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    el.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
    return true;
}"""


# ── Log verbosity ─────────────────────────────────────────
# Normal mode prints ONLY the essential signup events listed below (plus
# warnings / errors, which always print). Everything else — proxy sweeps,
# nav polls, fingerprint rotations, captcha retries, mail polling — only
# appears in the ALL logs: run with LOG_LEVEL=all to see it.
_LOG_ALL = os.environ.get("LOG_LEVEL", "").strip().lower() \
    in ("all", "debug", "verbose")

_ESSENTIAL_PREFIXES = (
    "[Nav] Navigating to ",                 # navigating to Discord
    "Using configured email:",              # email in use
    "[Mail] No email configured",           # creating an inbox
    "[Mail] [OK]",                          # inbox ready / verification link
    "Email: ",                              # filled email field
    "Display: ",                            # filled username + password fields
    "[Form] ToS",                           # ToS checkbox clicked
    "[Form] All fields + ToS verified OK",
    "[Form] Form filled - checking for hCaptcha",
    "Clicking Create Account",
    "[OK] Account button clicked",
    "[OK] Create Account submitted",
    "[Captcha] Checking for hCaptcha",
    "[Captcha] Waiting for hCaptcha to load",
    "[Captcha] Checkbox clicked",           # auto-clicked the hCaptcha checkbox
    "[Captcha] Clicking hCaptcha checkbox", # about to click the widget checkbox
    "[Captcha] [READY]",                    # hCaptcha rendered
    "[Captcha] [OK]",
    "[Captcha] Challenge round",
    "[Captcha] Clicking tiles:",
    "[Captcha] [OK] Vision",
    "[Captcha] Clicked Next",
    "[Captcha] Clicked Verify",
    "[Captcha] Offline",
    "[Captcha] Next challenge",
    "[Captcha] Offline tower",
    "[Captcha] Vision tower",
    "[Captcha] Tower wording",
    "[Captcha] Tower heuristic",
    "[Captcha] Tower last-resort",
)


def _log_essential(message: str) -> bool:
    """True when the message is one of the essential signup events."""
    if not any(message.startswith(p) for p in _ESSENTIAL_PREFIXES):
        return False
    return True


# ── TOR Control ───────────────────────────────────────────

def _tor_newnym():
    """Signal TOR to switch to a new identity (fresh exit node)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(15)
        s.connect(("127.0.0.1", 9051))
        s.recv(1024)
        s.sendall(b"AUTHENTICATE\r\n")
        auth_resp = s.recv(1024).decode().strip()
        if "250" not in auth_resp:
            s.close()
            print(f"[TOR] auth failed: {auth_resp}", flush=True)
            return False
        s.sendall(b"SIGNAL NEWNYM\r\n")
        resp = s.recv(1024).decode().strip()
        s.close()
        if "250" in resp:
            time.sleep(3)
            return True
        print(f"[TOR] newnym rejected: {resp}", flush=True)
    except Exception as e:
        print(f"[TOR] newnym error: {e}", flush=True)
    return False


def _tor_check():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(("127.0.0.1", 9050))
        s.close()
        return True
    except:
        return False


def _cgroup_oom_kills() -> int:
    """Total cgroup OOM kills for this container (0 when unreadable).

    cgroup v2 exposes the counter in memory.events ("oom_kill N"); v1 in
    memory.oom_control. A rising counter while a renderer "Page crashed"
    means the KILLER was the container memory limit — not a dead proxy —
    so rotating circuits cannot fix it.
    """
    for path in ("/sys/fs/cgroup/memory.events",
                 "/sys/fs/cgroup/memory/memory.oom_control"):
        try:
            with open(path, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2 and parts[0] == "oom_kill":
                        try:
                            return int(parts[1])
                        except ValueError:
                            return 0
        except Exception:
            pass
    return 0


PAST_CAPTCHA_KEYWORDS = ['/channels', '/verify', '/welcome', '@me', 'discord.com/app']

_BIO_POOL = [
    "just vibing",
    "professional sleeper",
    "i like turtles",
    "certified yapper",
    "caffeine powered",
    "music > everything",
    "gamer for life",
    "casually existing",
    "be nice or leave",
    "no thoughts, only vibes",
]

import stealth
from stealth import (
    apply_cdp_stealth,
    build_context_options,
    build_init_script,
    launch_args,
)

# Secondary navigations (verification link, token page). Halved from 60s:
# these pages are light and a hang this long only wastes a dead-session slot.
NAV_TIMEOUT_MS = 30000

# Hard cap on the /register render-wait.  A page can sit "loaded but no form"
# forever: Cloudflare serves a canned shell (title + #app-mount + the "You
# need to enable JavaScript" stub) to flagged IPs, JS bundles can drop, and
# half-dead circuits stall. Without a budget the worker polls indefinitely
# (the "insanely long" hang). Cloudflare managed challenges are EXEMPT — they
# auto-resolve and get unlimited time; everything else rotates to a fresh
# circuit once the budget is exhausted.
RENDER_WAIT_BUDGET_S = 75.0
LOW_MEMORY_MODE = (os.environ.get("LOW_MEMORY_MODE") or "1").strip().lower() not in ("0", "false", "no", "off")
LOW_MEMORY_VIEWPORT = {"width": 1280, "height": 720}

# ── LIVE camera frames (full browser view) ─────────────────────────
# A camera frame must be the BROWSER'S FULL VIEW — the whole window, with
# the page exactly as it is. The old full_page=True primary path (the
# engine maps it to CDP captureBeyondViewport) re-rendered the page with
# the viewport expanded to the full page size BEFORE the capture:
#   * Discord's React app listens for resize and re-lays out mid-capture,
#     so frames caught the form blank, shifted or half-rendered — the
#     "it looks like it hasn't loaded / the code is making it disappear"
#     symptom;
#   * content below the original viewport had never been painted (lazy
#     React), so the "full" image was blank exactly where the filled
#     form should have been;
#   * when the expanded capture missed its 20s budget (common in the
#     1 GB container — the re-render also allocates a huge paint buffer
#     and can OOM-kill the renderer), the engine silently fell back to a
#     viewport frame of just the top 720px — the "screenshot isn't full"
#     symptom.
# So the primary frame is now a plain FULL-VIEWPORT capture (the entire
# browser window: stable, zero page perturbation, no blank regions, no
# OOM), preceded by a conservative reveal step that scrolls the register
# form into view ONLY when it is entirely out of sight and no menu is
# open — the filled form is always in the Discord frame, and the camera
# can never fight the bot's own actions.
#
# FULLPAGE_SHOTS=1 opts into the WHOLE SCROLLABLE PAGE instead: the page
# is first scrolled through in viewport steps (forcing the lazy React
# content to actually render), then the full surface is captured with an
# explicit clip and the previous scroll position is restored. Off by
# default — the full browser view above is what the live feed shows.
FULLPAGE_SHOTS = (os.environ.get("FULLPAGE_SHOTS") or "0").strip().lower() not in ("0", "false", "no", "off")
FULLPAGE_MAX_PX = max(2000, int(os.environ.get("FULLPAGE_MAX_PX") or 8000))

# Scroll the register form back into view when it is ENTIRELY out of
# sight. Never touches the scroll while a menu/popout is open (a DOB
# dropdown mid-selection) or when the form is at least partially visible
# (the bot is working in it). Returns 'scrolled' when it moved the page.
_REVEAL_FORM_JS = r"""() => {
    try {
        const openMenu = document.querySelector(
            '[role="option"], [role="menuitem"], [class*="popout" i]');
        if (openMenu && openMenu.offsetParent !== null) return 'ok';
        const form = document.querySelector('form')
            || document.querySelector(
                'input[name="email"], input[type="email"], input[name="password"]');
        if (!form) return 'ok';
        const r = form.getBoundingClientRect();
        if (!r || (r.width === 0 && r.height === 0)) return 'ok';
        const vh = window.innerHeight || document.documentElement.clientHeight || 0;
        if (vh <= 0) return 'ok';
        if (!(r.bottom <= 0 || r.top >= vh)) return 'ok';
        form.scrollIntoView({ block: 'center', behavior: 'instant' });
        return 'scrolled';
    } catch (e) {
        return 'ok';
    }
}"""

# Live-camera reveal: still never fights an open menu, but if the register
# form is mostly below the fold we pull it up so the full browser window
# shows the form instead of a half-empty Discord chrome shot.
_REVEAL_FORM_LIVE_JS = r"""() => {
    try {
        const openMenu = document.querySelector(
            '[role="option"], [role="menuitem"], [class*="popout" i]');
        if (openMenu && openMenu.offsetParent !== null) return 'ok';
        const form = document.querySelector('form')
            || document.querySelector(
                'input[name="email"], input[type="email"], input[name="password"]');
        if (!form) return 'ok';
        const r = form.getBoundingClientRect();
        if (!r || (r.width === 0 && r.height === 0)) return 'ok';
        const vh = window.innerHeight || document.documentElement.clientHeight || 0;
        if (vh <= 0) return 'ok';
        const entirelyOut = (r.bottom <= 0 || r.top >= vh);
        const mostlyBelow = r.top > vh * 0.28 && r.bottom > vh;
        if (!entirelyOut && !mostlyBelow) return 'ok';
        form.scrollIntoView({ block: 'start', behavior: 'instant' });
        return 'scrolled';
    } catch (e) {
        return 'ok';
    }
}"""

PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"IEND\xaeB`\x82"


def png_is_complete(data) -> bool:
    """True when ``data`` is a finished PNG (signature + IEND), not a half frame."""
    if not isinstance(data, (bytes, bytearray)):
        return False
    if len(data) < 33 or bytes(data[:8]) != PNG_SIG:
        return False
    tail = bytes(data[-24:])
    return tail.endswith(PNG_IEND) or PNG_IEND in tail


def png_dimensions(data) -> tuple:
    """(width, height) from a PNG IHDR, or (0, 0) if unreadable."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 24:
        return (0, 0)
    if bytes(data[:8]) != PNG_SIG:
        return (0, 0)
    try:
        width, height = struct.unpack(">II", bytes(data[16:24]))
    except Exception:
        return (0, 0)
    if width <= 0 or height <= 0 or width > 20000 or height > 20000:
        return (0, 0)
    return (int(width), int(height))

# Full size of the scrollable document + the current viewport.
_PAGE_SIZE_JS = r"""() => {
    const de = document.documentElement;
    const b = document.body;
    const docH = Math.max(de ? de.scrollHeight : 0, b ? b.scrollHeight : 0);
    return {
        docH: docH,
        vw: window.innerWidth || (de ? de.clientWidth : 0),
        vh: window.innerHeight || (de ? de.clientHeight : 0),
    };
}"""


async def _full_page_shot(page, log=None,
                          fullpage_timeout: float = 20.0) -> bytes:
    """Whole scrollable surface of a lazy-rendering SPA, or b"".

    1. Measure the full size; skip when the page fits the viewport (the
       viewport frame IS the full page) or exceeds the OOM budget.
    2. Scroll through in viewport-sized steps with short pauses so React
       actually renders the lazy content below the fold (a bare
       beyond-viewport capture would paint that area blank).
    3. Capture the full surface with an explicit clip (one paint — no
       viewport resize storm), then restore the pre-capture scroll
       position so the frame never yanks the bot's working position.
    """
    try:
        size = await asyncio.wait_for(page.evaluate(_PAGE_SIZE_JS), timeout=2.5)
        if not isinstance(size, dict):
            return b""
        doc_h = float(size.get("docH") or 0)
        vw = float(size.get("vw") or 0)
        vh = float(size.get("vh") or 0)
        if vw <= 0 or vh <= 0 or doc_h <= vh * 1.05:
            return b""  # fits (roughly) in the viewport - viewport frame is full
        if doc_h > FULLPAGE_MAX_PX:
            if log:
                log(f"[Shot] page {int(doc_h)}px tall exceeds the "
                    f"{FULLPAGE_MAX_PX}px budget - viewport frame", level="info")
            return b""
        # Remember where the bot is working, then scroll through so the
        # lazy content renders (real scroll events, short settle pauses).
        try:
            scroll_y = float(await asyncio.wait_for(
                page.evaluate("window.scrollY || 0"), timeout=2.0))
        except Exception:
            scroll_y = 0.0
        step = vh * 0.9
        steps = max(1, int((doc_h - vh) / step)) + 1
        for i in range(1, steps + 1):
            y = min(step * i, doc_h - vh)
            try:
                await asyncio.wait_for(
                    page.evaluate(f"window.scrollTo(0, {int(y)})"), timeout=2.0)
            except Exception:
                break
            await asyncio.sleep(0.15)
        data = b""
        try:
            data = await asyncio.wait_for(
                page.screenshot(full_page=True,
                                clip={"x": 0, "y": 0,
                                      "width": int(vw), "height": int(doc_h)},
                                timeout=fullpage_timeout * 1000),
                timeout=fullpage_timeout)
        except Exception as e:
            if log:
                log(f"[Shot] full-page capture failed "
                    f"({type(e).__name__}) - viewport frame", level="warn")
        finally:
            try:
                await asyncio.wait_for(
                    page.evaluate(f"window.scrollTo(0, {int(scroll_y)})"),
                    timeout=2.0)
            except Exception:
                pass
        return data or b""
    except Exception:
        return b""


def _png_viewport_ok(data, min_w: int = 200, min_h: int = 200) -> bool:
    """True when ``data`` is a finished PNG large enough to be a full frame."""
    if not png_is_complete(data):
        return False
    width, height = png_dimensions(data)
    return width >= min_w and height >= min_h


async def capture_page_screenshot(page, log=None,
                                  fullpage_timeout: float = 20.0,
                                  viewport_timeout: float = 10.0,
                                  reveal: str = "safe") -> bytes:
    """Full browser-view frame, with the register form guaranteed in sight.

    Primary path (default): a conservative reveal step — scroll the
    register form into view only when it is ENTIRELY out of sight and no
    menu is open (see _REVEAL_FORM_JS), so the camera can never fight
    the bot's own actions — then a FULL-VIEWPORT capture: the entire
    browser window, zero page perturbation, no blank beyond-viewport
    regions, no OOM. This is the "full browser view" the feed shows.

    ``reveal="live"`` still never fights an open menu, but will also
    pull the register form up when it is mostly below the fold so the
    live camera shows the full Discord window instead of half chrome.

    Incomplete / truncated / tiny PNGs are rejected and retried. A
    glitched half-frame is never returned — callers keep their last
    good image instead.

    Opt-in path (FULLPAGE_SHOTS=1): the whole scrollable surface,
    rendered first by a scroll-through so lazy content is in the paint
    (see _full_page_shot), budgeted by FULLPAGE_MAX_PX.

    Every caller (worker camera loop, dashboard LIVE feed, the event
    captures in the signup flow) goes through here. Returns PNG bytes,
    or b"" when even the viewport capture fails (callers keep their last
    good frame).
    """
    if page is None:
        return b""

    async def _viewport(attempt_timeout: float) -> bytes:
        try:
            data = await asyncio.wait_for(
                page.screenshot(full_page=False, timeout=attempt_timeout * 1000), timeout=attempt_timeout)
            return data or b""
        except Exception:
            return b""

    live = str(reveal or "safe").strip().lower() == "live"
    reveal_js = _REVEAL_FORM_LIVE_JS if live else _REVEAL_FORM_JS
    try:
        if await asyncio.wait_for(page.evaluate(reveal_js), timeout=2.0) == "scrolled":
            await asyncio.sleep(0.25)  # let the scroll + repaint settle
    except Exception:
        pass
    # Two animation frames so Discord/React finish painting after a reveal.
    try:
        await asyncio.wait_for(page.evaluate(
            "() => new Promise((ok) => {"
            "requestAnimationFrame(() => requestAnimationFrame(ok));"
            "})"), timeout=1.5)
    except Exception:
        pass

    if FULLPAGE_SHOTS:
        try:
            shot = await _full_page_shot(page, log=log,
                                         fullpage_timeout=fullpage_timeout)
            if _png_viewport_ok(shot):
                return shot
        except Exception as e:
            if log:
                log(f"[Shot] full-page path failed ({type(e).__name__}) - viewport frame",
                    level="warn")

    # Primary frame: the full viewport (the entire browser window).
    # Live register retries a couple of times so a truncated/half PNG is
    # never published as the camera frame.
    tries = 3 if live else 2
    for attempt in range(tries):
        timeout = viewport_timeout if attempt == 0 else max(4.0, float(viewport_timeout) * 0.7)
        shot = await _viewport(timeout)
        if _png_viewport_ok(shot):
            return shot
        if attempt + 1 < tries:
            await asyncio.sleep(0.12)
    return b""


# ── Browser-error URL detection (engine-agnostic) ──────────────────────
# A dead proxy / DNS failure lands the tab on the browser's built-in error
# page. The engine is real Chrome, so the direct form is
# chrome-error://chromewebdata/; about:neterror:: is kept for robustness
# in case a future engine swap lands on Firefox. Both are DEFINITIVE
# "this circuit cannot reach Discord at all" signals — rotate.
_BROWSER_ERROR_URL_MARKERS = (
    "chrome-error://",
    "about:neterror",
    "neterror::",
)


def _is_browser_error_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return any(m in u for m in _BROWSER_ERROR_URL_MARKERS)


# ═══════════════════════════════════════════════════════════════
# Human Behavior Simulation
# ═══════════════════════════════════════════════════════════════

# Pointer realism is provided by human_mouse.py (manual cubic-bezier glide,
# settle, and human dwell on down/up) for the challenge interactions that
# need it, and by plain page.mouse clicks elsewhere. Those clicks ride on
# nodriver's CDP Input.dispatchMouseEvent — real Chrome input events.

class DiscordAutomation:
    def __init__(self, headless: bool = False, email: str = "",
                 proxy=None, worker_id: str = "B1", domain: str = "vibify.cc"):
        self.headless = headless
        self.worker_id = worker_id
        self._domain = (domain or "glasswhitehub.com").strip().lower() or "glasswhitehub.com"
        # proxy: dict {proto, host, port, username, password, key} or None
        self.proxy = proxy
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._ua = ""
        self._tor_enabled = False
        # True when the browser is running DIRECT (no proxy, no TOR) — the
        # last-resort transport when every residential session is dead and
        # TOR is unreachable. _build_context() honors it instead of raising.
        self._direct = False
        self._screenshots: list = []
        self._activity_log: list = []
        self._email = (email or os.environ.get("ACCOUNT_EMAIL", "")).strip()
        self._username = ""
        self._password = ""
        self._token = ""
        # Local Ollama vision model solves the hCaptcha image grid itself -
        # vision model solves the image grid directly (see vision_solver.py).
        self._vision = OllamaVisionClient(log=self._log)
        # Latest hCaptcha enterprise rqdata captured from the live getcaptcha
        # request (fresh per challenge, reset at the start of each attempt).
        self._rqdata = ""
        # JSON body of the last hCaptcha /getcaptcha RESPONSE — the challenge
        # payload carries request_type (which of the five challenge families
        # this round is), the prompt and the tile/reference URLs.
        self._challenge_payload = None
        # Offline CNN solvers (tile_classifier.py). Lazy: None until first
        # use, False when torch/weights are absent (vision-model fallback).
        self._cnn_tile = None
        self._cnn_point = None
        self._cnn_drag = None
        # duckmail.sbs client — created once per bot, reused across attempts.
        # (Lost in the cybertemp→duckmail switch, which silently killed every
        # inbox creation with a NoneType crash — see git log efb6f99.)
        self._mail: Optional[TempMail] = TempMail(log=self._log)
        self._user_id = ""
        self._avatar_data = ""
        self._bio = ""
        self._humanized = False
        self._exit_ip = ""
        # Set when Discord asks for phone verification after account creation
        # — the worker then rotates proxy + fingerprint + mail domain and retries.
        self.phone_verify_detected = False
        # True once this session actually rendered Discord's register page
        # (used by the worker to distinguish dead sessions from soft failures).
        self._nav_ok = False
        # True when the mail provider failed BEFORE Discord even loaded — the
        # worker uses this to retry the same proxy/fingerprint instead of
        # rotating (mail failures are not IP problems).
        self._mail_failed = False
        # Human-readable reason the last _goto_register() returned False —
        # surfaced in the worker's per-attempt summary so every failure is
        # self-explanatory ("TOR circuit blocked: page unresponsive after 9s").
        self._nav_error: str = ""
        # Set by the app when the user hits Stop — aborts an in-flight
        # navigation wait immediately so Stop actually stops (the browser is
        # then PARKED on Discord and reused on the next Start).
        self._stopped = asyncio.Event()
        # True once the browser's renderer process dies ("Page crashed").
        # A crashed tab is gone for good — the render-wait loop recovers it
        # in place once (fresh page, same circuit) then rotates instead of
        # burning 20 dead polls on a corpse.
        self._page_crashed = False
        self._last_browser_diag = 0.0
        # OOM tracking: cgroup OOM-kill counter at browser launch + crash
        # timestamps. A renderer crash accompanied by a rising counter is a
        # MEMORY kill — recovery must reclaim + retry the SAME transport
        # instead of rotating circuits (which cannot fix memory pressure).
        self._oom_kills_at_launch = 0
        self._oom_crash_times: list = []
        # Real Chrome owns the identity: a fresh temporary profile per launch
        # (nodriver mints a temp user-data-dir) — there is no bot-side
        # fingerprint to keep or randomize.
        self._fingerprint = {}

    def _log(self, message: str, level: str = "info") -> None:
        # The store keeps EVERYTHING so the dashboard's ALL LOGS toggle can
        # show the full detail; the console only prints essential events +
        # warnings/errors unless LOG_LEVEL=all.
        essential = level in ("warn", "error") or _log_essential(message)
        print_console = _LOG_ALL or essential
        tagged = f"[{self.worker_id}] {message}"
        entry = {
            "time": time.strftime("%H:%M:%S"),
            "timestamp": time.time(),
            "level": level,
            "essential": essential,
            "message": tagged
        }
        self._activity_log.append(entry)
        if len(self._activity_log) > 500:
            self._activity_log = self._activity_log[-500:]
        if print_console:
            print(f"[{entry['time']}] [{level.upper()}] {tagged}", flush=True)

    def _log_exception(self, message: str, exc: Exception) -> None:
        # Record the EXACT problem (exception class + full traceback) into
        # the activity log, so the dashboard's ALL LOGS toggle shows why a
        # step failed instead of only a stderr traceback it never sees.
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._log(f"{message} — {tb.rstrip()}", level="error")

    async def _log_browser_diagnostics(self, reason: str,
                                       exc: Optional[BaseException] = None,
                                       force: bool = False) -> None:
        """Add concise runtime evidence for browser crashes to ALL LOGS.

        This avoids guessing whether a repeated page crash came from a proxy,
        a stale browser build, a root runtime, or a cgroup memory/pid limit.
        Sensitive connection credentials and account data are deliberately
        excluded from the report.
        """
        now = time.time()
        if not force and now - self._last_browser_diag < 2.0:
            return
        self._last_browser_diag = now

        def _read(path: str, limit: int = 240) -> str:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read(limit).strip().replace("\n", "; ")
            except Exception as read_error:
                return f"unavailable:{type(read_error).__name__}"

        def _collect() -> dict:
            try:
                from importlib.metadata import version
                nodriver_version = version("nodriver")
            except Exception:
                nodriver_version = "unknown"
            cache_dir = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "nodriver")
            return {
                "reason": reason,
                "error_type": type(exc).__name__ if exc is not None else "",
                "error": str(exc)[:500] if exc is not None else "",
                "uid": os.getuid() if hasattr(os, "getuid") else None,
                "engine": ENGINE,
                "nodriver_package": nodriver_version,
                "browser_connected": bool(getattr(self._browser, "is_connected", False)) if self._browser else False,
                "page_present": self._page is not None,
                "transport": "proxy" if self.proxy else ("direct" if self._direct else "tor"),
                "home": os.environ.get("HOME", ""),
                "cache_exists": os.path.isdir(cache_dir),
                "cgroup_memory_current": _read("/sys/fs/cgroup/memory.current"),
                "cgroup_memory_max": _read("/sys/fs/cgroup/memory.max"),
                "cgroup_memory_events": _read("/sys/fs/cgroup/memory.events"),
                "cgroup_oom_kills": _cgroup_oom_kills(),
                "cgroup_pids_current": _read("/sys/fs/cgroup/pids.current"),
                "cgroup_pids_max": _read("/sys/fs/cgroup/pids.max"),
                "process_status": _read("/proc/self/status", 1200),
            }

        try:
            diagnostics = await asyncio.to_thread(_collect)
            self._log("[Diag] Browser failure " + json.dumps(diagnostics, sort_keys=True), level="error")
            # Make the #1 container killer impossible to miss in ALL LOGS:
            # a non-zero cgroup OOM-kill counter means the browser process
            # was killed by the container memory limit, NOT by a bad proxy
            # or a dead circuit — rotating cannot fix it.
            try:
                if int(diagnostics.get("cgroup_oom_kills") or 0) > 0:
                    self._log(
                        f"[Diag] cgroup OOM killer has fired {diagnostics.get('cgroup_oom_kills')}x "
                        f"in total — browser processes are being KILLED by the container memory "
                        f"limit (max={diagnostics.get('cgroup_memory_max')}). This is a memory "
                        "problem, not a proxy/circuit problem; raise the container memory limit.",
                        level="error")
            except Exception:
                pass
        except Exception as diag_error:
            self._log(f"[Diag] Browser failure diagnostics unavailable: {type(diag_error).__name__}: {diag_error}", level="error")

    def get_activity_log(self) -> list:
        return self._activity_log

    def _launch_proxy(self) -> Optional[dict]:
        """The proxy rides on browser launch (nodriver applies it as Chrome
        --proxy-server/--proxy-user/--proxy-pass launch flags — a
        context-level proxy would be rejected when the browser already has
        one). Returns the Playwright-style {server, username, password} dict
        (or None for TOR/direct)."""
        if not (self.proxy and isinstance(self.proxy, dict)):
            return None
        p = self.proxy
        proto = p.get("proto", "http")
        lp = {"server": f"{proto}://{p.get('host')}:{p.get('port')}"}
        if p.get("username"):
            lp["username"] = p.get("username")
            lp["password"] = p.get("password", "")
        return lp

    async def _relaunch_browser(self) -> None:
        """Close and relaunch the browser bound to self.proxy. The engine
        cannot change a running browser's proxy (it is a launch flag), so a
        proxy change requires a full relaunch."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        args = launch_args(headless=self.headless)
        # The engine pins the proxy at browser launch. When there is no sticky
        # residential session, relaunch must ride TOR exactly like initialize()
        # does — otherwise switch_proxy(None) silently goes DIRECT (the
        # context-level proxy is ignored by the engine) and Discord's
        # Cloudflare blocks the datacenter IP with a browser error page.
        launch_proxy = self._launch_proxy()
        if launch_proxy is None and not self._direct and _tor_check():
            launch_proxy = {"server": "socks5://127.0.0.1:9050"}
            self._tor_enabled = True
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless, args=args, proxy=launch_proxy)
        await self._build_context()

    _PROXY_IP_CACHE: dict = {}

    def _resolve_proxy_ip(self, host: str) -> str:
        """DNS-resolve the proxy host to an IP (best-effort, cached per host).
        Runs in a worker thread — never blocks the event loop."""
        if not host:
            return ""
        if host in self._PROXY_IP_CACHE:
            return self._PROXY_IP_CACHE[host]
        try:
            ip = socket.gethostbyname(host)
        except Exception:
            return ""
        if ip:
            self._PROXY_IP_CACHE[host] = ip
        return ip

    async def _log_proxy_exit_ip(self) -> None:
        """Best-effort: report the REAL egress IP of the current browser
        session (residential exit != gateway DNS IP). Bounded, never blocks."""
        page = self._page
        if page is None:
            return
        try:
            ip = await asyncio.wait_for(page.evaluate(
                "async () => { try { "
                "const r = await fetch('https://api.ipify.org?format=json', "
                "{cache: 'no-store'}); "
                "const d = await r.json(); return d.ip || ''; } "
                "catch (e) { return ''; } }"
            ), timeout=6)
        except Exception:
            return
        if ip:
            self._exit_ip = ip  # captured for the persistent proxy store
            label = "proxy session" if self.proxy else "TOR circuit"
            self._log(f"[Proxy] Exit IP ({label}): {ip}")

    async def initialize(self) -> None:
        self._playwright = await async_playwright().start()

        args = launch_args(headless=self.headless)
        self._log(f"[Engine] {ENGINE} launch args: {len(args)}")

        # Real Chrome IS the identity: genuine Chrome JS/TLS/HTTP2
        # fingerprints, real fonts, real capabilities — nothing for the bot
        # to pin, mint, or randomize.
        self._ua = ""
        self._fingerprint = {}
        self._log(f"[Engine] Fresh {ENGINE} context requested")

        # Launch the browser WITH the proxy. The engine applies it at
        # firefox launch — a proxy passed later to new_context() would be
        # rejected and traffic would go direct.
        launch_proxy = self._launch_proxy()
        if launch_proxy is None and not self._direct and _tor_check():
            launch_proxy = {"server": "socks5://127.0.0.1:9050"}
            self._tor_enabled = True
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless, args=args, proxy=launch_proxy)

        # Self-identifying engine line: if this is ABSENT, the deployed
        # image predates the nodriver/Chrome switch (or the OOM-aware
        # recovery) — rebuild before drawing conclusions from its OOMs.
        # Headless real-Chrome is the lightest engine we've shipped, and
        # cgroup OOM attribution (see _cgroup_oom_kills) is always on.
        self._log(
            f"[Engine] {ENGINE} on real Google Chrome (google-chrome-stable), "
            f"headless={self.headless}, LOW_MEMORY_MODE={'on' if LOW_MEMORY_MODE else 'off'}"
        )

        # Use a smaller renderer surface in a 1 GB container. This keeps
        # page paint and screenshot buffers materially below a 1920x1080
        # desktop surface while preserving a standard desktop layout.
        await self._build_context()

        # Done — context created by _build_context with full CDP evasion

    async def _build_context(self) -> None:
        """Build a fresh browser context with current self.proxy.
        Shared by initialize() and switch_proxy()."""
        vp = dict(LOW_MEMORY_VIEWPORT) if LOW_MEMORY_MODE else {'width': 1920, 'height': 1080}
        ctx_opts = build_context_options(
            self._fingerprint, self._ua, proxy=self.proxy, viewport=vp
        )
        if self.proxy and isinstance(self.proxy, dict):
            p = self.proxy
            server = f"{p.get('proto', 'http')}://{p.get('host')}:{p.get('port')}"
            host_ip = await asyncio.to_thread(self._resolve_proxy_ip, p.get("host", ""))
            ip_part = f" IP={host_ip}," if host_ip else ""
            self._log(f"Proxy: {server} ({ip_part} auth={'yes' if p.get('username') else 'no'})")
        elif getattr(self, "_direct", False):
            self._tor_enabled = False
            self._log("[Proxy] Direct connection - no proxy")
        elif _tor_check():
            self._tor_enabled = True
            self._log("[TOR] Using TOR SOCKS5 proxy...")
            if _tor_newnym():
                self._log("[TOR] New identity requested")
            # Chrome already rides the TOR proxy from browser launch — a
            # context-level proxy would be rejected when the browser was
            # launched with one.
            await asyncio.sleep(1)
        else:
            self._log("[TOR] [FATAL] TOR SOCKS5 (127.0.0.1:9050) NOT reachable - TOR-only mode requires TOR running on this instance", level="error")
            self._tor_enabled = False
            raise RuntimeError("TOR not available - TOR-only mode requires TOR on 127.0.0.1:9050")

        self._context = await self._browser.new_context(**ctx_opts)
        if self._ua:
            self._log(f"User-Agent: {self._ua[:60]}...")
        else:
            self._log("[Fingerprint] User-Agent: engine-owned identity")
        await self._context.add_init_script(
            build_init_script(self._fingerprint, self._ua)
        )
        # ENGLISH IS FORCED (operator request): spoof navigator.language /
        # languages so hCaptcha + Discord render English even when the
        # proxy region or site would otherwise localize them.
        await self._context.add_init_script(
            "() => {"
            "try {"
            "Object.defineProperty(navigator, 'language', {get: () => 'en-US'});"
            "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});"
            "Object.defineProperty(navigator, 'userLanguage', {get: () => 'en-US'});"
            "Object.defineProperty(navigator, 'browserLanguage', {get: () => 'en-US'});"
            "} catch (e) {}"
            "}"
        )
        self._page = await self._context.new_page()
        self._attach_rqdata_capture()
        self._attach_crash_listener()

        # Automation-tell stripping (incl. navigator.webdriver) is done by
        # nodriver itself at the CDP level — apply_cdp_stealth is a contract
        # no-op.
        await apply_cdp_stealth(self._context, self._page)

        # Baseline for OOM attribution: a renderer crash with a HIGHER cgroup
        # OOM-kill counter than this is a memory kill, not a dead proxy.
        self._oom_kills_at_launch = _cgroup_oom_kills()

        # Report the real egress IP of this session (bounded, never blocks).
        asyncio.create_task(self._log_proxy_exit_ip())

    def _attach_rqdata_capture(self) -> None:
        """Listen for hCaptcha's getcaptcha POST and stash its enterprise rqdata.

        Discord runs hCaptcha in enterprise mode: every token is bound to the
        per-challenge rqdata the page passes to the widget. That value is NOT
        reliably present in the static DOM (the widget renders from a minified
        bundle), but hCaptcha's own JS echoes it in the getcaptcha request body
        when the checkbox is clicked. Attaching here — at page creation, before
        any navigation — means we catch it whether it fires on widget init or
        on our checkbox click.
        """
        if self._page is None:
            return
        try:
            self._page.on("request", self._on_page_request)
        except Exception as e:
            self._log(f"[Captcha] Could not attach rqdata request capture: {e}",
                      level="warn")
        # Response side of the same exchange: the /getcaptcha RESPONSE body is
        # the challenge payload (request_type, prompt, tile URLs) — the most
        # reliable of the three family-classification tiers.
        try:
            self._page.on("response", self._on_page_response)
        except Exception as e:
            self._log(f"[Captcha] Could not attach payload response capture: {e}",
                      level="warn")

    def _attach_crash_listener(self) -> None:
        """Detect tab crashes (renderer died) the moment they happen instead
        of learning about them through 20 dead polls in the render-wait loop.

        A crashed page is gone for good — it never resurrects on its own. The
        flag lets _goto_register recover the tab in place (fresh page on the
        SAME circuit: renderer crashes are usually transient memory spikes,
        not dead proxies) and rotate only if it crashes again."""
        self._page_crashed = False
        if self._page is None:
            return
        try:
            self._page.on("crash", self._on_page_crash)
        except Exception as e:
            self._log(f"[Nav] Could not attach crash listener: {e}", level="warn")

    def _on_page_crash(self) -> None:
        self._page_crashed = True
        self._log("[Nav] Page crashed (browser tab died)", level="warn")
        try:
            asyncio.get_running_loop().create_task(
                self._log_browser_diagnostics("renderer-crash-event")
            )
        except RuntimeError:
            # The callback can run while the loop is closing; the navigation
            # exception path records the same diagnostics in that case.
            pass

    def _on_page_request(self, request) -> None:
        try:
            url = (request.url or "").lower()
            if "hcaptcha" not in url:
                return
            if "getcaptcha" not in url and "checkcaptcha" not in url:
                return
            rqdata = ""
            # Some enterprise builds pass rqdata as a getcaptcha QUERY param
            # instead of (or alongside) the POST body. The blob is
            # case-sensitive base64, so parse the ORIGINAL (unlowered) URL.
            try:
                from urllib.parse import parse_qs, urlsplit
                _q = parse_qs(urlsplit(request.url).query)
                for _k, _vv in _q.items():
                    if "rqdata" in _k.lower() and _vv:
                        _cand = _vv[0].strip()
                        if len(_cand) > 8:
                            rqdata = _cand
                            break
            except Exception:
                pass
            if not rqdata:
                body = None
                try:
                    body = getattr(request, "post_data_buffer", None)
                except Exception:
                    body = None
                if body is None:
                    try:
                        body = getattr(request, "post_data", None)
                    except Exception:
                        body = None
                if body is not None:
                    rqdata = extract_rqdata_from_body(body)
            if rqdata:
                self._rqdata = rqdata
                self._log(
                    f"[Captcha] Captured enterprise rqdata ({len(rqdata)} chars) "
                    f"from {request.url[-60:]}")
        except Exception as e:
            self._log(f"[Captcha] rqdata capture error: {e}", level="debug")

    async def _on_page_response(self, response) -> None:
        """Stash the /getcaptcha JSON (the challenge payload) as it arrives."""
        try:
            url = (response.url or "").lower()
            if "hcaptcha" not in url or "getcaptcha" not in url:
                return
            if response.status != 200:
                return
            data = await response.json()
            if isinstance(data, dict) and data.get("request_type"):
                self._read_challenge_payload(data)
        except Exception:
            pass

    def _read_challenge_payload(self, data: dict = None) -> Optional[dict]:
        """Store /getcaptcha JSON and log the challenge family it carries.

        Called by the response hook with the fresh payload; called with no
        argument it just returns the last payload stored."""
        if data is not None:
            self._challenge_payload = data
            family = hct.classify_from_payload(data)
            self._log(
                f"[Captcha] getcaptcha payload: "
                f"request_type={data.get('request_type')!r} -> family={family}")
        return self._challenge_payload

    async def switch_proxy(self, new_proxy=None) -> bool:
        """Swap to a new proxy AND a fresh fingerprint. Returns True on success.

        The engine pins the proxy at browser launch, so switching to a
        DIFFERENT session relaunches the browser; reusing the same session
        only rebuilds the context (and keeps the fingerprint — rotating an
        identity on an unchanged IP just churns fingerprints)."""
        self._direct = False
        same_session = bool(
            new_proxy and self.proxy
            and new_proxy.get("key") == self.proxy.get("key")
        )
        proxy_changed = (new_proxy or {}).get("key") != (self.proxy or {}).get("key")
        self.proxy = new_proxy
        if same_session:
            # Pool recycled the SAME session (all sessions blacklisted then
            # re-issued). Rebuilding the context with the same fingerprint is
            # consistent — regenerating an identity on an unchanged IP only
            # churns fingerprints for nothing.
            self._log("[Fingerprint] Same proxy session reused - keeping fingerprint")
        else:
            # Fresh fingerprint per session — same UA/GPU/font on a new IP is a
            # fingerprinting red flag and a known trigger for phone verification.
            self.rotate_fingerprint()
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
        except Exception:
            pass
        self._page = None
        self._context = None
        try:
            # A failed previous relaunch can leave self._browser None — never
            # call _build_context() (which does browser.new_context()) on a
            # dead browser: that was the "'NoneType' object has no attribute
            # 'new_context'" crash. Relaunch the browser instead.
            if proxy_changed or self._browser is None or not getattr(self._browser, "is_connected", True):
                self._log("[Switch] Proxy changed or browser restart needed — relaunching browser with new session")
                await self._relaunch_browser()
            else:
                try:
                    await self._build_context()
                except Exception as b_err:
                    self._log(f"[Switch] Context rebuild failed ({b_err}) — falling back to full browser relaunch", level="warn")
                    await self._relaunch_browser()
            label = 'proxy ' + str(new_proxy.get('key','?')[:40]) if new_proxy else 'fresh TOR circuit'
            self._log(f"[Switch] Context rebuilt with {label}")
            return True
        except Exception as e:
            self._log(f"[Switch] Context rebuild failed: {e}", level="error")
            self._browser = None
            return False

    async def switch_direct(self) -> bool:
        """Relaunch the browser with NO proxy (direct egress). Last resort when
        every residential session is dead and TOR is unavailable — the LIVE
        tab still renders a real page instead of a browser error page."""
        self.proxy = None
        self._direct = True
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
        except Exception:
            pass
        self._page = None
        self._context = None
        try:
            await self._relaunch_browser()
            self._log("[Switch] Relaunched with direct connection (no proxy)")
            return True
        except Exception as e:
            self._log(f"[Switch] Direct relaunch failed: {e}", level="error")
            return False

    async def is_alive(self) -> bool:
        """True if the browser + page are still usable.

        A parked browser (kept alive across Stop/Start) can die while the
        worker is stopped — TOR circuit dropped, browser crashed. Reuse is
        gated on this: a dead parked browser gets closed and relaunched."""
        if self._browser is None or self._page is None:
            return False
        try:
            if not getattr(self._browser, "is_connected", True):
                return False
            url = await asyncio.wait_for(
                self._page.evaluate("location.href"), timeout=3.0)
            return bool(url)
        except Exception:
            return False

    async def _page_is_closed(self) -> bool:
        """Return whether the current page is unavailable without raising."""
        page = self._page
        if page is None:
            return True
        try:
            closed = getattr(page, "is_closed", None)
            if callable(closed):
                return bool(closed())
            if closed is not None:
                return bool(closed)
            # some engine wrappers may not expose is_closed; a tiny bounded
            # read distinguishes a live page from a closed target.
            await asyncio.wait_for(page.evaluate("location.href"), timeout=1.0)
            return False
        except Exception:
            return True

    def rotate_fingerprint(self) -> None:
        """Rotate to a brand-new browser identity.

        Real Chrome with a fresh temp profile is already a new identity on
        every relaunch. Reset local state so the next launch starts from a
        clean context."""
        self._fingerprint = {}
        self._ua = ""
        self._log(f"[Engine] {ENGINE} context will be recreated on next launch")

    async def _rebuild_context_with_tor(self) -> bool:
        """Close the context and reopen WITH a fresh TOR circuit."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if not _tor_check():
                self._log("[Nav] TOR not available for rebuild", level="error")
                return False
            if _tor_newnym():
                self._log("[Nav] Fresh TOR circuit requested")
            await asyncio.sleep(1)
            self._context = await self._browser.new_context(
                **build_context_options(
                    self._fingerprint, self._ua,
                    proxy={'proto': 'socks5', 'host': '127.0.0.1', 'port': '9050'},
                    viewport=random.choice([
                        {'width': 860, 'height': 640},
                        {'width': 1024, 'height': 768},
                        {'width': 900, 'height': 700},
                    ]),
                )
            )
            await self._context.add_init_script(
                build_init_script(self._fingerprint, self._ua)
            )
            # ENGLISH IS FORCED (operator request): spoof navigator.language /
            # languages so hCaptcha + Discord render English even when the
            # proxy region or site would otherwise localize them.
            await self._context.add_init_script(
                "() => {"
                "try {"
                "Object.defineProperty(navigator, 'language', {get: () => 'en-US'});"
                "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});"
                "Object.defineProperty(navigator, 'userLanguage', {get: () => 'en-US'});"
                "Object.defineProperty(navigator, 'browserLanguage', {get: () => 'en-US'});"
                "} catch (e) {}"
                "}"
            )
            self._page = await self._context.new_page()
            self._attach_rqdata_capture()
            self._attach_crash_listener()
            await apply_cdp_stealth(self._context, self._page)
            self._log("[Nav] Rebuilt browser context WITH fresh TOR proxy")
            return True
        except Exception as e:
            self._log(f"[Nav] context rebuild failed: {e}", level="error")
            return False

    async def _read_nav_state(self):
        """Read the register-page state.

        Every evaluate() runs over the engine's own channel, so there is no
        stale-WebDriver session to fall back from. A healthy page ALWAYS
        answers; (None, None) is the one true "tab dead" signal.
        """
        try:
            checks = await asyncio.wait_for(
                self._page.evaluate(_NAV_STATE_JS), timeout=2.5)
            if checks:
                st = json.loads(checks)
                st["source"] = "cdp"
                return st, "cdp"
        except Exception:
            pass
        # Mid-navigation the old execution context is destroyed before the
        # new one registers — one quick retry, then report dead.
        try:
            await asyncio.sleep(0.15)
            checks = await asyncio.wait_for(
                self._page.evaluate(_NAV_STATE_JS), timeout=2.5)
            if checks:
                st = json.loads(checks)
                st["source"] = "cdp"
                return st, "cdp"
        except Exception:
            pass
        return None, None

    async def _cdp_dom_nav_state(self):
        """JS-only alias of _read_nav_state (kept for callers)."""
        return await self._read_nav_state()

    async def _goto_register(self) -> bool:
        """Navigate to discord.com/register — single attempt, no retries.

        If the form doesn't render, we return False immediately so the worker
        can rotate to a fresh TOR circuit. Retrying the same URL on the same
        circuit is pointless — if Discord blocked that exit node, it won't
        unblock on retry."""
        url = "https://discord.com/register"
        recovered_unavailable_page = False
        # A fresh navigation must not inherit a stale crash flag from a
        # previous page (recovery also resets it — belt and suspenders).
        self._page_crashed = False
        # 30s cap like the original build: the goto is only a warm-up — the
        # form-poll below is the real render gate and returns the INSTANT the
        # form paints (0.15s polling). Dead sessions still bail via the hard
        # cap; slow-but-alive sessions survive: the goto coroutine is
        # cancelled in the background (the tab keeps committing) and the
        # title/url grace-poll below lets them catch up before anything is
        # declared dead.
        timeout_ms = 30000

        self._log(f"[Nav] Navigating to {url} (timeout={timeout_ms}ms)...")
        t0 = time.time()
        try:
            # domcontentloaded (not "load"): "load" waits for EVERY subresource
            # including the hCaptcha widget iframe and all its JS, which through
            # slow proxies hangs for tens of seconds. The form-poll loop below
            # already waits for the Discord SPA to boot, so we lose nothing.
            #
            # WRAPPED in asyncio.wait_for: the engine's timeout is advisory
            # only. When the proxy is dead, the browser's internal TCP retry
            # logic can hang regardless of timeout — asyncio.wait_for with a
            # hard cap kills the coroutine and forces a fresh proxy.
            await asyncio.wait_for(
                self._page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms),
                timeout=(timeout_ms / 1000.0) + 3.0,  # hard cap (never 18s)
            )
            self._log(f"[Nav] Page DOM ready in {time.time() - t0:.1f}s (not waiting for hCaptcha subresources)")
        except asyncio.TimeoutError:
            elapsed = time.time() - t0
            self._log(f"[Nav] Page.goto HARD TIMEOUT after {elapsed:.1f}s — proxy likely dead")
        except Exception as e:
            # The engine raises its OWN TimeoutError - Playwright's class is
            # NOT asyncio.TimeoutError, so it used to escape this handler,
            # blow out of _goto_register into the mail branch of
            # start_discord_signup, wipe the fresh inbox and abort the whole
            # attempt as "No email available". It can also raise transport
            # errors when the circuit dies mid-commit. None of that is fatal
            # here: the render-wait loop below is the real gate and rotates
            # on browser-error / dead reads, so a slow-but-alive TOR circuit
            # gets to finish loading instead of being killed at 30s.
            elapsed = time.time() - t0
            err_type = type(e).__name__.lower()
            err_text = str(e).lower()
            if "timeout" in err_type or "timeout" in err_text:
                self._log(f"[Nav] Page.goto timeout ({type(e).__name__}) after {elapsed:.1f}s - continuing to render-wait")
            elif ("targetclosed" in err_type or "target page" in err_text
                  or "has been closed" in err_text or "context or browser has been closed" in err_text):
                # The target is already gone. Recover before reading title/url,
                # otherwise the dashboard receives a misleading '(unknown)'.
                reason = "page, context, or browser closed during navigation"
                self._nav_error = reason
                self._log(f"[Nav] Page.goto TargetClosedError - {reason}", level="warn")
                recovered_unavailable_page = await self._recover_crashed_page(url, reason)
                if not recovered_unavailable_page:
                    self._nav_error = f"{reason} - browser restart failed"
                    return False
            elif "crashed" in err_text:
                # Renderer died mid-commit. Flag it so the render-wait loop
                # restarts the browser once instead of polling a corpse.
                self._page_crashed = True
                await self._log_browser_diagnostics("page-goto-crashed", e)
                self._log(f"[Nav] Page.goto CRASH ({type(e).__name__}: {str(e)[:120]}) - recovering browser", level="warn")
            else:
                self._log(f"[Nav] Page.goto error ({type(e).__name__}: {e}) - continuing to render-wait", level="warn")
        # ── Check what we got ──
        try:
            page_title = await asyncio.wait_for(self._page.title(), timeout=3.0)
            page_url = await asyncio.wait_for(self._page.evaluate("location.href"), timeout=3.0)
        except Exception:
            page_title = "(unknown)"
            page_url = "(unknown)"
        if (str(page_title) == "(unknown)" and str(page_url) == "(unknown)"
                and await self._page_is_closed()):
            # A page/context closure is definitive, unlike a slow navigation.
            # Restart before emitting an unknown-title line or entering the
            # render loop with an object that can no longer answer.
            if not recovered_unavailable_page:
                reason = "page closed before navigation state was available"
                recovered_unavailable_page = await self._recover_crashed_page(url, reason)
            if not recovered_unavailable_page:
                self._nav_error = "page closed before navigation state was available"
                self._log("[Nav] Page closed before state read - rotating circuit", level="warn")
                return False
            try:
                page_title = await asyncio.wait_for(self._page.title(), timeout=3.0)
                page_url = await asyncio.wait_for(self._page.evaluate("location.href"), timeout=3.0)
            except Exception:
                page_title = "(starting after restart)"
                page_url = url
        if (str(page_title) == "(unknown)" and str(page_url) == "(unknown)"
                and not self._page_crashed):
            # The hard cap can cancel goto while a slow-but-alive session is
            # still committing; the tab usually answers within a beat. Grace-
            # poll up to ~5s before declaring the session dead — this is what
            # lets the shorter goto cap skip dead sessions fast WITHOUT killing
            # slow-but-healthy ones.
            for _grace in range(5):
                await asyncio.sleep(1.0)
                try:
                    page_title = await asyncio.wait_for(self._page.title(), timeout=3.0)
                    page_url = await asyncio.wait_for(self._page.evaluate("location.href"), timeout=3.0)
                except Exception:
                    continue
                if str(page_title) != "(unknown)" or str(page_url) != "(unknown)":
                    break
        self._log('[Nav] Page: title="' + str(page_title)[:80] + '" url=' + str(page_url)[:80])

        # ── Dead proxy (cannot reach Discord at all) ──
        # A browser error page (chrome-error:// on Chrome; about:neterror on
        # other engines) is a REAL dead signal (DNS/connection failure).
        # about:blank is NOT dead: a slow TOR/residential circuit can still
        # be committing the navigation when the goto cap fires, so a blank
        # tab means "still loading" here, not "dead". Dead sessions surface
        # an error page within seconds; blank-but-alive sessions just need
        # the render-wait loop below (which re-issues the goto if the tab
        # stays blank). Bouncing on about:blank is what made the bot "fail
        # instantly without waiting" on slow circuits.
        if _is_browser_error_url(page_url):
            proxy_label = "PROXY SESSION" if self.proxy else "TOR CIRCUIT"
            self._nav_error = f"{proxy_label.lower()} dead (browser error page: {page_url[:60]})"
            self._log(f"[Nav] {proxy_label} DEAD (url={page_url[:60]}) - rotating to fresh circuit")
            return False
        if (page_url or "").strip() in ("", "about:blank"):
            self._log("[Nav] Tab still at about:blank after goto cap - navigation still committing, render-wait will re-issue if stuck", level="warn")

        # ── Page still loading (title + url unreadable) — do NOT bail. ──
        # A slow TOR/residential circuit can keep the page unreadable for
        # minutes while the navigation commits and React downloads. The
        # render-wait loop below is the only gate; real dead proxies were
        # already caught by the browser-error / about:blank check above.
        title_blank = not str(page_title or "").strip()
        url_blank = not str(page_url or "").strip() or str(page_url or "").strip() in ("about:blank",)
        if title_blank and url_blank:
            self._log(f"[Nav] Page white/unreadable after goto (title empty, url={str(page_url)[:40]}) - render-wait keeps polling and rotates if it stays dead")

        # ── Quick body text check (403/Forbidden/Cloudflare) ──
        try:
            body_text = await asyncio.wait_for(
                self._page.evaluate("document.body ? document.body.innerText.substring(0, 500) : ''"),
                timeout=3.0)
            if body_text and any(kw in body_text.lower() for kw in (
                "forbidden", "403 forbidden", "access denied", "cloudflare",
                "attention required", "rate limit", "ratelimited", "rate limited",
                "too many requests", "slowdown", "try again later", "429",
            )):
                self._nav_error = f"blocked by Discord/Cloudflare — body text: {body_text[:80]}"
                self._log(f"[Nav] BLOCKED — body contains: {body_text[:100]}", level="warn")
                return False
        except Exception:
            pass

        # ── Rate-limit detection — rotate as soon as the message renders ──
        # Discord/Cloudflare throttle abused exit nodes with 429s that render
        # as "The resource is being rate limited." / "too many requests".
        # Check the FULL body text — the message often paints below the first
        # few hundred chars, so truncating would miss it.
        try:
            full_body = await asyncio.wait_for(
                self._page.evaluate("document.body ? document.body.innerText : ''"),
                timeout=3.0)
        except Exception:
            full_body = ""
        if any(kw in (full_body or "").lower() for kw in _RATE_LIMIT_KEYWORDS):
            self._nav_error = "rate limited (429) by Discord"
            self._log("[Nav] RATE LIMITED (429) - rotating circuit", level="warn")
            return False

        # ── Block keywords in title ──
        # Cloudflare managed-challenge interstitial is NOT fatal: it auto-
        # resolves in ~5-15s and drops cf_clearance. Bouncing on the title
        # was throwing away healthy sessions that just needed a beat. Flow
        # into the poll loop, which waits for cf_clearance and only rotates
        # if the challenge persists past its own budget.
        challenge_title_kws = ["just a moment", "attention required",
                               "checking your browser", "verify you are human"]
        fatal_title_kws = ["blocked", "cloudflare", "ddos-guard", "captcha",
                           "forbidden", "403", "access denied",
                           "you do not have permission", "error 1020",
                           "rate limit", "ratelimited", "rate limited",
                           "too many requests", "slowdown", "try again later"]
        title_lower = (page_title or "").lower()
        if any(kw in title_lower for kw in fatal_title_kws):
            self._nav_error = f"blocked - Cloudflare/firewall hard block (title: {str(page_title)[:60]})"
            self._log('[Nav] BLOCKED by Cloudflare/firewall (title: "' + str(page_title)[:60] + '")', level="warn")
            return False
        if any(kw in title_lower for kw in challenge_title_kws):
            self._log('[Nav] Cloudflare challenge title "' + str(page_title)[:40] + '" - waiting for auto-resolve in poll loop')
        # ── Check if Discord SPA shell loaded ──
        try:
            app_mount = await asyncio.wait_for(
                self._page.evaluate("document.querySelector('#app-mount') !== null"),
                timeout=5.0)
            if app_mount:
                self._log("[Nav] Discord SPA app-mount detected")
        except Exception:
            app_mount = False

        # ── Poll for form elements ──
        # Discord uses aria-label on inputs, not name/id — use broad selectors.
        #
        # NO RENDER TIMEOUT: this loop waits as long as it takes for the form
        # to fully render and returns the INSTANT it paints (checks every
        # 0.15s). There is no wall-clock budget — the only exits are real
        # signals: a successful render or a hard block (403 / rate-limit /
        # fatal title, detected above). An unreadable or blank page is still
        # loading — keep waiting; reload it up to max_reloads times to
        # re-fetch dropped JS bundles, then keep waiting.
        reload_after = 4.0       # standard mode: blank this long -> re-fetch bundles
        max_reloads = 0 if LOW_MEMORY_MODE else 2
        _render_wait_start = time.time()
        self._log(f"[Nav] Waiting for registration form to render (no timeout - reloads<={max_reloads}, reload_after={reload_after:.0f}s)...")
        blank_since = None       # when the page first looked blank
        challenge_since = None   # when a Cloudflare challenge first appeared
        reload_count = 0         # reloads attempted for a blank/hung SPA
        login_clicked = False    # already clicked the Register link once
        turnstile_tried = False  # already attempted a Turnstile bypass
        blank_nav_since = None   # when the tab first sat at about:blank (nav never committed)
        nav_reissues = 0         # re-issued gotos for a never-committed navigation
        dead_reads = 0           # consecutive unreadable polls -> page died (old-build bail)
        crash_recovered = False  # already recovered a crashed tab on this circuit this nav
        last_log = -1.0
        while True:
            # User hit Stop — abort the wait immediately (the browser gets
            # parked on Discord for reuse; it is NOT killed).
            if self._stopped.is_set():
                self._log("[Nav] Stopped by user - aborting navigation wait")
                self._nav_error = "stopped by user"
                return False
            elapsed = time.time() - _render_wait_start
            # Dual-channel read: page.evaluate() (the engine falls back to the
            # raw CDP websocket when the reattached WebDriver session's JS
            # context is stale — the old white-screen bug where the page
            # loaded fine, title read "Discord", but evaluate() returned
            # None and the bot rotated a PERFECTLY GOOD session), then CDP
            # DOM-presence checks that need no JS execution at all. state is
            # None ONLY when every channel failed = the page is genuinely
            # dead.
            state, _chan = await self._read_nav_state()

            if state is None:
                # Every read channel failed (WebDriver JS + CDP JS + CDP DOM)
                # — tab white / dead. Probe WHAT'S WRONG every ~3s for ALL
                # logs (title, url, readyState, body length, the error), then
                # after 20 consecutive dead polls rotate — the old-build
                # white-screen bail. A healthy page can NEVER reach here,
                # because the CDP fallback keeps reads alive.
                dead_reads += 1
                # Confirmed crash: the tab is GONE and will never come back.
                # Recover in place ONCE (fresh page, same circuit — a
                # renderer crash is usually a transient memory spike, not a
                # dead proxy), then rotate immediately instead of polling a
                # corpse 20x and burning a good circuit on a bad diagnosis.
                page_closed = await self._page_is_closed()
                if self._page_crashed or page_closed:
                    reason = ("renderer crashed" if self._page_crashed
                              else "page, context, or browser closed")
                    if not crash_recovered:
                        crash_recovered = await self._recover_crashed_page(url, reason)
                        if crash_recovered:
                            dead_reads = 0
                            blank_nav_since = None
                            last_log = -1.0
                            continue
                    self._nav_error = f"{reason} - rotating circuit"
                    self._log(f"[Nav] {reason.capitalize()} - rotating circuit", level="warn")
                    return False
                if elapsed >= last_log + 3.0:
                    last_log = elapsed
                    probe = "(no probe)"
                    try:
                        probe = await asyncio.wait_for(self._page.evaluate(
                            """() => {
                                try {
                                    return JSON.stringify({
                                        title: document.title || "",
                                        url: location.href || "",
                                        readyState: document.readyState || "",
                                        bodyLen: document.body ? (document.body.innerText || "").length : -1
                                    });
                                } catch (e) { return "probe-err: " + (e && e.message || e); }
                            }"""
                        ), timeout=2.0)
                    except Exception as _pe:
                        probe = f"probe-failed: {type(_pe).__name__}: {_pe}"
                    self._log(f"[Nav] Page unreadable ({int(elapsed)}s, {dead_reads}x in a row) - probe: {probe}")
                if dead_reads >= 20:
                    self._nav_error = "page unreadable 20x in a row (white screen / tab dead) - rotating circuit"
                    self._log("[Nav] Page unreadable 20x in a row - page died, rotating circuit")
                    return False
                await asyncio.sleep(0.3)
                continue
            dead_reads = 0

            if state:
                # ── Mid-wait page-health checks ──
                cur_url = (state.get("url") or "").strip() or ""
                # Browser error page appearing mid-wait = the circuit died.
                if _is_browser_error_url(cur_url):
                    self._nav_error = "proxy/circuit dead (browser error page)"
                    self._log("[Nav] Browser error page - rotating circuit")
                    return False
                if cur_url in ("", "about:blank"):
                    # Navigation never committed (goto cap fired mid-commit on
                    # a slow circuit). Re-issue the goto so the page actually
                    # starts loading instead of waiting forever on a blank
                    # tab. max_reloads re-issues, then KEEP waiting — the
                    # directive is no render timeout.
                    if blank_nav_since is None:
                        blank_nav_since = time.time()
                    elif time.time() - blank_nav_since >= 5.0 and nav_reissues < max_reloads:
                        nav_reissues += 1
                        self._log(f"[Nav] Tab stuck at about:blank for {int(time.time() - blank_nav_since)}s - re-issuing goto ({nav_reissues}/{max_reloads})...", level="warn")
                        try:
                            await asyncio.wait_for(
                                self._page.goto(url, wait_until="domcontentloaded", timeout=30000),
                                timeout=33.0)
                        except Exception:
                            pass
                        await asyncio.sleep(1.0)
                        blank_nav_since = None
                else:
                    blank_nav_since = None

                # Log every ~4s with input/button counts for debugging
                if elapsed >= last_log + 4.0:
                    last_log = elapsed
                    self._log(f"[Nav] Poll {int(elapsed)}s ({_chan}): email={state.get('email')} ageGate={state.get('ageGate')} login={state.get('isLogin')} inputs={state.get('inputCount')} buttons={state.get('buttonCount')} cf={state.get('cfClearance')} text={state.get('textPreview','')[:60]}")

                if state.get("email") and state.get("username"):
                    self._log(f"[Nav] SUCCESS! Full form rendered after {int(elapsed)}s")
                    return True
                if state.get("email") and state.get("password"):
                    self._log(f"[Nav] SUCCESS! Email+password form rendered after {int(elapsed)}s")
                    return True
                if state.get("ageGate"):
                    self._log(f"[Nav] Age gate detected after {int(elapsed)}s - returning true, form filler handles it")
                    return True

                # Blank render + Cloudflare challenge handling.
                # Two very different causes, handled separately:
                #   1. Cloudflare managed challenge ("Just a moment..."): it
                #      auto-resolves and drops cf_clearance. WAIT for it as
                #      long as it takes — no bail-out (Turnstile widgets get
                #      re-attempted every ~10s).
                #   2. React failed to hydrate (a JS bundle dropped/errored): a
                #      reload re-fetches the bundles and almost always boots.
                #      Reload up to max_reloads times, then keep waiting.
                if state.get("challenge"):
                    if challenge_since is None:
                        challenge_since = time.time()
                        self._log("[Nav] Cloudflare 'Just a moment' challenge detected - waiting for auto-resolve (cf_clearance)...")
                    blank_since = None
                    if state.get("cfClearance"):
                        self._log("[Nav] cf_clearance set - challenge passed, waiting for React to boot...")
                        challenge_since = None
                    else:
                        # Cloudflare Turnstile widget (not the auto-resolving
                        # managed challenge). Click it with a real humanized
                        # click and re-attempt every ~10s — no bail-out; wait
                        # as long as it takes.
                        if (not turnstile_tried
                                or time.time() - challenge_since >= 10.0):
                            turnstile_tried = True
                            self._log("[Nav] Cloudflare Turnstile widget detected - clicking it...")
                            if await self._solve_turnstile_if_present():
                                self._log("[Nav] Turnstile clicked - waiting for React to boot...")
                                challenge_since = None
                    await asyncio.sleep(0.3)
                    continue
                challenge_since = None

                # Render budget - never hang forever on a stub page. Reached
                # only when there is no active Cloudflare challenge (the
                # challenge block above already continues, so managed
                # challenges get unlimited time). A page that still has no
                # form after the budget gets the session rotated to a fresh
                # circuit instead of polling indefinitely.
                if elapsed >= RENDER_WAIT_BUDGET_S:
                    self._nav_error = (f"Discord form never rendered after {int(elapsed)}s "
                                       "(stub page / dead circuit) - rotating to fresh circuit")
                    self._log(f"[Nav] Form never rendered after {int(elapsed)}s - rotating to fresh circuit", level="warn")
                    return False

                # "You need to enable JavaScript to run this app." as the body
                # text = Discord/Cloudflare served a canned shell (title +
                # #app-mount) but React never boots - a flagged exit IP or
                # dropped JS bundles. Treat it as blank so the reload path
                # re-fetches the bundles; if it persists past the reloads it
                # rotates to a fresh circuit (reloads never fix a stub).
                _preview_text = (state.get("textPreview") or "").strip()
                _js_required = ("you need to enable javascript" in _preview_text.lower()
                                or "enable javascript to run this app" in _preview_text.lower())

                if (state.get("hasAppMount") and not state.get("inputCount")
                        and not state.get("buttonCount")
                        and (not _preview_text or _js_required)):
                    if blank_since is None:
                        blank_since = time.time()
                        if _js_required:
                            self._log("[Nav] Stub page ('You need to enable JavaScript') - React never boots; reloading to re-fetch bundles (rotating if persistent)...", level="warn")
                        else:
                            self._log("[Nav] SPA shell mounted but React not booted - waiting for JS bundles (reload if stuck)...")
                    # cf_clearance set = challenge passed - assets unblocked, form should follow.
                    if state.get("cfClearance"):
                        if blank_since is not None:
                            self._log("[Nav] cf_clearance cookie appeared - Cloudflare challenge passed, waiting for React...")
                        blank_since = None
                    elif (not LOW_MEMORY_MODE and time.time() - blank_since >= reload_after
                          and reload_count < max_reloads):
                        reload_count += 1
                        self._log(f"[Nav] React still blank after {int(reload_after)}s - reloading page (attempt {reload_count}/{max_reloads}) to re-fetch JS bundles...")
                        try:
                            await asyncio.wait_for(self._page.reload(), timeout=15.0)
                        except Exception:
                            pass
                        await asyncio.sleep(1.2)
                        blank_since = None
                        challenge_since = None
                        continue
                    elif LOW_MEMORY_MODE and time.time() - blank_since >= reload_after:
                        # Reloading a full SPA creates a short-lived second
                        # renderer/allocation spike. In a 1 GB container wait
                        # for the render budget and rotate cleanly if it never
                        # hydrates instead of risking an OOM renderer kill.
                        if int(time.time() - blank_since) == int(reload_after):
                            self._log("[Nav] Low-memory mode: React still blank; skipping reload to protect renderer")
                    elif not LOW_MEMORY_MODE and reload_count >= max_reloads and _js_required:
                        # A JS-required stub after reloads is a flagged exit
                        # IP, not a dropped bundle - reloading will never fix
                        # it. Rotate NOW instead of burning the full budget.
                        self._nav_error = "Discord served JS-required stub after reloads - rotating to fresh circuit"
                        self._log("[Nav] Stub page persists after reloads - rotating to fresh circuit", level="warn")
                        return False
                    # max_reloads exhausted (blank, non-stub) - the budget
                    # check above rotates the session instead of waiting
                    # forever. A slow circuit still gets its full chance.
                else:
                    blank_since = None
                if state.get("isLogin") and not login_clicked and elapsed >= 3.0:
                    self._log("[Nav] Login page detected \u2014 clicking Register link...")
                    try:
                        clicked_reg = await self._page.evaluate("""() => {
                            const all = document.querySelectorAll('a, button, [role="link"], [role="button"]');
                            for (const el of all) {
                                const t = (el.textContent || '').toLowerCase().replace(/\s+/g, ' ').trim();
                                if (t && /register|sign up|create account|registrieren|inscription|s'inscrire|registrarse|registreren|registrera|opret konto|załóż konto|создать аккаунт|регистрация|đăng ký|가입|注册|登録|kayıt ol/i.test(t) && el.offsetParent !== null) {
                                    el.scrollIntoView({block: 'center'});
                                    el.click();
                                    return 'clicked';
                                }
                            }
                            return '';
                        }""")
                    except Exception:
                        clicked_reg = ''
                    if clicked_reg:
                        self._log("[Nav] Clicked Register link \u2014 continuing poll for register form...")
                        login_clicked = True
                        blank_since = None
                        await asyncio.sleep(0.3)
                        continue
                    self._log("[Nav] Login page, no Register link clickable \u2014 rotating circuit", level="warn")
                    break

            # Check for redirect to app
            try:
                cur = await asyncio.wait_for(self._page.evaluate("location.href"), timeout=1.0)
                if "discord.com/app" in cur or "discord.com/channels" in cur:
                    self._log(f"[Nav] Redirected to app: {cur[:60]}")
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.15)

        # ── Form never rendered — dump page state for debugging ──
        try:
            dump = await asyncio.wait_for(self._page.evaluate("""() => {
                const inputs = Array.from(document.querySelectorAll('input')).slice(0, 20).map(function(e) { return {
                    type: e.type, name: e.name, id: e.id, ariaLabel: e.getAttribute('aria-label') || '',
                    placeholder: e.placeholder || '', visible: e.offsetParent !== null
                }; });
                const buttons = Array.from(document.querySelectorAll('button')).slice(0, 10).map(function(e) { return {
                    text: (e.innerText || '').substring(0, 40), type: e.type, visible: e.offsetParent !== null
                }; });
                return JSON.stringify({
                    title: document.title,
                    url: location.href,
                    bodyText: (document.body?.innerText || '').substring(0, 200),
                    inputs: inputs,
                    buttons: buttons,
                    hasAppMount: document.querySelector('#app-mount') !== null
                });
            }"""), timeout=2.0)
            self._log(f"[Nav] DEBUG page state: {dump}")
        except Exception:
            pass

        proxy_label = "fresh proxy session" if self.proxy else "fresh TOR circuit"
        self._nav_error = f"Discord form never rendered (loop exited without a render signal) — rotating to {proxy_label}"
        self._log(f"[Nav] Form never rendered - rotating to {proxy_label}", level="warn")
        return False


    async def _recover_crashed_page(self, url: str,
                                    reason: str = "renderer crashed") -> bool:
        """Recover one unavailable page with a fresh browser process.

        A renderer crash or a TargetClosedError can leave the browser
        transport technically connected while its page/context is unusable.
        Restarting the complete browser avoids polling a target that no
        longer exists; exhaustion is left to normal rotation.

        OOM-aware: when the cgroup OOM-kill counter rose since this browser
        launched, the crash was a MEMORY kill — rotating circuits cannot fix
        memory pressure. So recovery retries the SAME transport a few times
        with reclaim pauses (kernel memory reclaim + Python GC) between
        attempts, and after repeated OOM crashes it backs off hard and tells
        the operator the container memory limit is the problem.
        """
        if self._stopped.is_set():
            return False
        oom_now = _cgroup_oom_kills()
        oom_since_launch = oom_now - getattr(self, "_oom_kills_at_launch", 0)
        is_oom = oom_since_launch > 0

        # Track OOM crashes in a 5-minute window. Three of them means the
        # container cap is simply too small for real Chrome + Discord — a
        # hot retry loop would just burn proxy sessions faster.
        if is_oom:
            now = time.time()
            self._oom_crash_times = [t for t in self._oom_crash_times if now - t < 300]
            self._oom_crash_times.append(now)
            if len(self._oom_crash_times) >= 3:
                self._log(
                    f"[Mem] {len(self._oom_crash_times)} cgroup OOM kills of the browser in the "
                    f"last 5 minutes — the container memory limit is too small for real Chrome + "
                    f"Discord (cgroup oom_kill={oom_now}). Reclaiming, then backing off 60s. "
                    "Raise the container memory limit (2 GB recommended) to stop this.",
                    level="error")
                gc.collect()
                await asyncio.sleep(60)

        self._log(
            f'[Nav] {reason} - restarting browser process'
            + (f' (cgroup OOM kills since launch: {oom_since_launch} — memory pressure, '
               'not a dead proxy; reclaiming before each retry)' if is_oom
               else ' once'),
            level='warn')
        attempts = 3 if is_oom else 1
        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    # Let the OOM killer's work settle: kernel reclaim first,
                    # then the Python heap (screenshot ring, logs) — only
                    # then spend another browser on the same memory cap.
                    gc.collect()
                    await asyncio.sleep(4.0 * attempt)
                await self._relaunch_browser()
                if self._page is None:
                    raise RuntimeError('browser restart produced no page')
                self._page_crashed = False
                await asyncio.wait_for(
                    self._page.goto(url, wait_until='domcontentloaded', timeout=30000),
                    timeout=33.0,
                )
                await asyncio.sleep(0.5)
                self._log('[Nav] Browser restart after renderer crash completed', level='info')
                return True
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._page_crashed = True
                if attempt < attempts:
                    self._log(
                        f'[Nav] Restart attempt {attempt}/{attempts} failed ({type(e).__name__}) '
                        f'- memory reclaim pause, retrying same transport', level='warn')
                    continue
                await self._log_browser_diagnostics("browser-restart-failed", e, force=True)
                self._log(
                    f'[Nav] Browser restart after renderer crash failed ({type(e).__name__}: {e})'
                    + (' — caused by the container memory cap; rotating the circuit will not '
                       'fix it, the memory limit must be raised' if is_oom else ''),
                    level='error',
                )
                return False
    async def capture_screenshot(self) -> str:
        if not self._page:
            return ""
        # Full browser-view frame (form revealed when out of sight; whole
        # scrollable page only when FULLPAGE_SHOTS=1) — the capture policy
        # lives in capture_page_screenshot. Same base64 history contract
        # as before: latest frame appended, ring trimmed at 100.
        screenshot = await capture_page_screenshot(
            self._page, log=self._log, fullpage_timeout=20.0, reveal="live")
        if not screenshot or not png_is_complete(screenshot):
            return ""
        b64 = base64.b64encode(screenshot).decode('utf-8')
        self._screenshots.append(b64)
        # Tiny ring on purpose: each frame is a full-viewport PNG in base64
        # (~150-400 KB) and this list lives in the Python heap next to the
        # browser inside a 1 GB container — 100 frames was a 30-40 MB
        # constant tax. The dashboard only ever needs the latest frame.
        if len(self._screenshots) > 10:
            self._screenshots = self._screenshots[-8:]
        return b64

    async def start_discord_signup(self) -> bool:
        if not self._page:
            await self.initialize()
        self.phone_verify_detected = False
        self._nav_ok = False
        self._mail_failed = False
        # rqdata is single-use and per-challenge: never carry a stale blob
        # from a previous page load into this attempt's solve.
        self._rqdata = ""

        # app.py closes + nulls self._mail between attempts (prevents aiohttp
        # connector leaks) while REUSING this bot object for the next attempt —
        # re-create the duckmail client here or the next inbox creation crashes
        # with "'NoneType' object has no attribute 'create_inbox'" and the
        # worker spins forever on the same dead mail path.
        if self._mail is None:
            self._mail = TempMail(log=self._log)

        # No hardcoded email — create a duckmail.sbs inbox on the
        # Discord-friendly domain @glasswhitehub.com (pure REST API, no
        # browser, no proxy contention). Retry fast (2x, no backoff) when
        # duckmail hiccups.
        if not self._email:
            self._log(f"[Mail] No email configured - creating duckmail.sbs inbox (@{self._domain})...")
            try:
                # duckmail is a pure REST client (api.duckmail.sbs, Hydra
                # API) — no browser involved. The inbox is created on the
                # fixed Discord-friendly domain @glasswhitehub.com.

                # Create inbox FIRST (with hard timeout) — never in parallel
                # with CDP navigation on the same browser.
                self._email = ""
                for mail_try in range(2):
                    try:
                        self._email = await asyncio.wait_for(
                            self._mail.create_inbox(), timeout=20.0)
                    except asyncio.TimeoutError:
                        self._log("[Mail] Inbox creation TIMED OUT after 20s", level="error")
                    except Exception as e:
                        self._log(f"[Mail] duckmail inbox error: {e}", level="error")
                    if self._email:
                        break
                    self._log(f"[Mail] Inbox creation failed — retrying ({mail_try + 1}/2)...", level="warn")

            except Exception as e:
                self._log(f"[Mail] duckmail inbox error: {e}", level="error")
                self._email = ""

            if not self._email:
                self._mail_failed = True
                self._log("[FAIL] No email available - aborting signup", level="error")
                return False
            # NOW navigate to Discord - inbox is ready. Navigation is a
            # separate concern from inbox creation: a dead circuit / 429 /
            # block is handled INSIDE _goto_register (it never raises) and
            # rotates the session. It must never wipe the freshly created
            # inbox or get misreported as an email failure - that was the
            # "duckmail inbox error: Page.goto ... No email available" lie
            # that aborted every attempt the moment a TOR circuit was slow
            # to commit.
            nav_ok = await self._goto_register()
            if not nav_ok:
                # The inbox is still unused, but app.py tears the mail client
                # down between attempts, so a stale address cannot be verified
                # on the next attempt. Drop it so the next attempt mints a
                # fresh inbox on the new circuit.
                self._email = ""
                self._log("[FAIL] Could not navigate to Discord /register - aborting", level="error")
                return False

        else:
            self._log(f"Using configured email: {self._email}")
            if not await self._goto_register():
                self._log("[FAIL] Could not navigate to Discord /register - aborting", level="error")
                return False

        self._nav_ok = True
        self._log("=" * 40)
        self._log(f"Starting Discord signup with email: {self._email}")
        self._log("=" * 40)

        try:
            self._log("[Nav] Discord site rendered")
            await self.capture_screenshot()

            # ── Settle wait before touching the form ──
            # Discord's SPA keeps re-rendering after the form first paints;
            # writing values during that window gets them wiped on the next
            # re-render (the "fields stay empty" bug). Wait a fixed 20s so
            # hydration fully finishes before the filler runs.
            self._log("[Nav] Waiting 20s for Discord to fully settle before filling...")
            settle_deadline = time.time() + 20.0
            while time.time() < settle_deadline:
                if self._stopped.is_set():
                    self._nav_error = "stopped by user"
                    self._log("[Nav] Stopped by user during settle wait")
                    return False
                await asyncio.sleep(0.5)

            # Fill the form
            if self._page is None:
                self._nav_error = "page closed mid-run (browser stopped/closed while settling)"
                self._log("[Nav] Page is gone (browser stopped/closed) - aborting attempt", level="warn")
                return False
            form_ok = await self._fill_registration_form()
            success = False
            if form_ok:
                self._log("[Form] Form filled - checking for hCaptcha...")
                # Cloudflare Turnstile can gate the form submit. Click it
                # with a real humanized click before the hCaptcha solver runs.
                if await self._solve_turnstile_if_present():
                    self._log("[Captcha] [OK] Turnstile clicked")
                success = await self._solve_hcaptcha_if_present()
            else:
                self._log("[FAIL] Form filling failed", level="error")

            if success:
                self._log("[OK] CAPTCHA SOLVED! Registration submitted.")
                # Discord can demand phone verification right after account
                # creation. Detect it BEFORE waiting on email — if present,
                # abort this attempt so the worker rotates proxy + fingerprint
                # + mail domain and retries (phone-gated accounts are dead).
                # Poll every second so a phone gate is caught the moment it
                # renders instead of after a fixed 5s sleep (cap ~6s so the
                # happy path to email verification isn't delayed).
                phone_detected = False
                for _ in range(10):
                    if await self._detect_phone_verification():
                        phone_detected = True
                        break
                    await asyncio.sleep(0.5)
                if phone_detected:
                    self.phone_verify_detected = True
                    self._log("[Phone] [DETECTED] Phone verification required - rotating proxy+fingerprint+domain", level="warn")
                    return False
                # Auto-verify: complete Discord email verification. Skipped
                # when a custom email is in use — the user clicks the link in
                # their own inbox, so we just tell them.
                await self._verify_account_email()
                # Login + grab the FULL token from localStorage. With a custom
                # email the user may need to click the verify link first, so
                # keep watching much longer (60s) and re-submit the login.
                self._token = await self._extract_token(
                    attempts=6 if self._email and not self._mail else 4,
                    poll_rounds=30 if self._email and not self._mail else 10,
                )
                if self._token:
                    self._log("[Token] [OK] Full token captured")
                    self._log(f"[Account] @{self._username or self._email.split('@')[0]} is in Discord and confirmed")
                    self._log(f"[Account] Email={self._email} | User={self._username} | Pass={self._password} | Date={time.strftime('%Y-%m-%d %H:%M')}")
                    await self._humanize_account()
                else:
                    self._log("[Token] No token yet (account may still be pending)", level="warn")
            else:
                self._log("[FAIL] Captcha solving failed", level="error")

        except Exception as e:
            self._log(f"Error: {e}", level="error")
            import traceback
            traceback.print_exc()
            success = False

        await self.capture_screenshot()
        return success

    async def _humanize_account(self) -> None:
        """Set avatar and bio on the newly created Discord account.

        Uses the Discord API directly with the captured token. Best-effort
        only — failures are logged but never block the account from being
        saved (a humanized account is nice but non-critical)."""
        if not (self._token and self._page):
            return
        try:
            import aiohttp
            bio = random.choice(_BIO_POOL)
            headers = {"Authorization": self._token, "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                # Set bio
                async with s.patch(
                    "https://discord.com/api/v9/users/@me",
                    json={"bio": bio}, headers=headers,
                ) as r:
                    if r.status == 200:
                        data = await r.json()
                        self._bio = bio
                        self._user_id = str(data.get("id", ""))
                        avatar_hash = data.get("avatar") or ""
                        if avatar_hash:
                            self._avatar_data = avatar_hash
                        self._humanized = True
                        self._log(f"[Humanize] Bio set: \"{bio}\"")
                    else:
                        self._log(f"[Humanize] API returned {r.status}", level="warn")
        except Exception as e:
            self._log(f"[Humanize] Error: {e}", level="warn")

    # ── Cloudflare Turnstile ─────────────────────────────────────────────
    # Discord sits behind Cloudflare, and a Turnstile captcha can gate
    # navigation / form submit / mail verification. The widget is clicked
    # with a real, humanized locator click on the Chrome page (nodriver
    # drives the pointer with CDP input events — never a synthetic JS
    # event), then we confirm the challenge cleared via cf_clearance or the
    # widget leaving the DOM.
    _TURNSTILE_SELECTORS = (
        'iframe[src*="challenges.cloudflare.com"]',
        'iframe[src*="turnstile"]',
        'div.cf-turnstile iframe',
    )

    async def _detect_turnstile(self) -> bool:
        """True when a Cloudflare Turnstile widget is on the page.

        Turnstile is a separate anti-bot layer from hCaptcha: Cloudflare
        mounts it inside a challenges.cloudflare.com iframe (a div with the
        ``cf-turnstile`` class in the page DOM)."""
        try:
            for sel in self._TURNSTILE_SELECTORS:
                loc = self._page.locator(sel)
                if await loc.count() > 0:
                    return True
        except Exception:
            pass
        # Frame-tree fallback: any live frame on challenges.cloudflare.com.
        try:
            for f in self._page.frames:
                if "challenges.cloudflare.com" in (f.url or ""):
                    return True
        except Exception:
            pass
        return False

    async def _solve_turnstile_if_present(self) -> bool:
        """Bypass a Cloudflare Turnstile widget with a humanized click.

        Clicks the widget checkbox with a real locator click on the Chrome
        page, then confirms the challenge cleared via the cf_clearance
        cookie or the widget frame disappearing."""
        try:
            if not await self._detect_turnstile():
                return False
            self._log("[Turnstile] Widget present - clicking it...")
            # Humanized click on the widget checkbox (nodriver drives the
            # pointer with CDP input events — never a synthetic JS event).
            clicked = False
            for sel in self._TURNSTILE_SELECTORS:
                try:
                    loc = self._page.locator(sel)
                    if await loc.count() > 0:
                        await loc.first.click(timeout=4000)
                        clicked = True
                        break
                except Exception:
                    continue
            if not clicked:
                self._log("[Turnstile] No checkbox found to click", level="warn")
                return False
            # 3) Confirm the challenge cleared (cf_clearance or widget gone).
            for _ in range(10):
                try:
                    raw = await self._page.evaluate(
                        "() => document.cookie")
                    if "cf_clearance=" in (raw or ""):
                        self._log("[Turnstile] [OK] cf_clearance issued")
                        return True
                except Exception:
                    pass
                if not await self._detect_turnstile():
                    self._log("[Turnstile] [OK] widget resolved")
                    return True
                await asyncio.sleep(0.4)
            self._log("[Turnstile] Click sent but no clearance yet", level="warn")
            return False
        except Exception as e:
            self._log(f"[Turnstile] Error: {e}", level="warn")
            return False

    async def _detect_phone_verification(self) -> bool:
        """Check the current page for Discord's phone-verification screen.

        Discord shows this right after account creation (or as a login gate)
        when it suspects automation. Markers: a phone/tel input, or a phone
        heading/body. Returns True when the account needs a phone number."""
        try:
            result = await asyncio.wait_for(self._page.evaluate("""() => {
                // Phone input (name=phone / type=tel / aria/placeholder)
                const phoneInput = document.querySelector(
                    'input[name="phone"], input[type="tel"], ' +
                    'input[aria-label*="phone" i], input[placeholder*="phone" i], ' +
                    'input[autocomplete="tel"]');
                if (phoneInput && phoneInput.offsetParent !== null) {
                    return 'input';
                }
                const text = (document.body ? document.body.innerText : '').toLowerCase();
                const markers = [
                    'verify your phone', 'phone verification', 'verify your account',
                    'add a phone number', 'phone number required',
                    'we need to verify your account', 'enter your phone number',
                    'confirm your phone', 'what\'s your phone number',
                    'verify via phone', 'add your phone number',
                ];
                for (const kw of markers) {
                    if (text.includes(kw)) return 'text:' + kw;
                }
                return '';
            }"""), timeout=4.0)
            return bool(result)
        except Exception:
            return False

    async def _verify_account_email(self) -> bool:
        """Wait for the Discord verification email and open its link (best effort).
        Aborts early if Discord instead demands phone verification."""
        if not self._mail:
            return False
        try:
            link = await self._mail.wait_for_verification_link(timeout=150)
            if not link:
                self._log("[Mail] No verification link found yet - account may still be created", level="warn")
                return False
            self._log(f"[Mail] Opening verification link: {link[:80]}...")
            await self._page.goto(link, wait_until='domcontentloaded', timeout=NAV_TIMEOUT_MS)
            await asyncio.sleep(2)
            # Cloudflare Turnstile may gate the verification page - click it
            # if present.
            if await self._solve_turnstile_if_present():
                self._log("[Mail] [OK] Turnstile bypassed on verification page")
            # Discord shows a verification success page (or redirects to login)
            try:
                page_text = await self._page.evaluate(
                    "() => document.body.innerText.substring(0, 300)")
            except Exception:
                page_text = ""
            if any(w in (page_text or "").lower()
                   for w in ('verified', 'success', 'confirmation', 'you\'re all set')):
                self._log("[Mail] [OK] Email verification completed")
            await self.capture_screenshot()
            self._log("[Mail] [OK] Verification link opened")
            return True
        except Exception as e:
            self._log(f"[Mail] verification error: {e}", level="warn")
            return False

    async def _past_captcha(self) -> bool:
        """True when the page has moved past the captcha into Discord."""
        try:
            cur_url = self._page.url
            return any(k in cur_url for k in PAST_CAPTCHA_KEYWORDS)
        except:
            return False

    async def _hcaptcha_frame_for(self, iframe):
        """Resolve the live Playwright Frame for an hCaptcha iframe element.

        Locator.content_frame() returns None for attached cross-origin
        iframes on the patched engine - even though the frames are live and
        evaluable (the DOM-dump path proves it by iterating page.frames).
        So fall back to the page's frame tree, preferring the VISIBLE widget
        frame: body not aria-hidden AND containing a checkbox node. hCaptcha
        mounts a hidden twin (body aria-hidden=true, same URL) - never pick
        it when a visible one exists.
        """
        # 1) Direct content_frame() first (real Playwright: Locator.
        # content_frame is a property, so resolve the element handle first).
        try:
            frame = await (await iframe.element_handle(timeout=5000)).content_frame()
            if frame is not None:
                return frame
        except Exception:
            pass
        # 2) Frame-tree fallback: match by src, then by content.
        src = ""
        try:
            src = await iframe.get_attribute("src") or ""
        except Exception:
            src = ""
        probe_js = """() => {
            const b = document.body;
            return JSON.stringify({
                cb: !!document.querySelector('#checkbox, [role="checkbox"], .checkbox, input[type="checkbox"], [aria-checked], .button-submit'),
                hidden: b ? b.getAttribute('aria-hidden') : null
            });
        }"""
        best = None
        for f in self._page.frames:
            try:
                furl = f.url or ""
            except Exception:
                continue
            if "hcaptcha" not in furl:
                continue
            info = None
            try:
                raw = await f.evaluate(probe_js)
                info = json.loads(raw) if raw else None
            except Exception:
                info = None
            if src and src in furl:
                best = best or f
            if info and info.get("cb"):
                if info.get("hidden") != "true":
                    return f
                if best is None:
                    best = f
        return best

    async def _frame_js_ready(self, iframe, js) -> bool:
        """Evaluate `js` inside the iframe's content frame; False on any error."""
        frame = await self._hcaptcha_frame_for(iframe)
        if frame is None:
            return False
        try:
            val = await frame.evaluate(js)
            return bool(val)
        except Exception:
            return False

    async def _challenge_rendered(self, iframe) -> bool:
        """True only when the hCaptcha challenge iframe has genuinely painted.

        A challenge iframe is laid out at full size (>= 80px tall) the moment
        it is inserted, BEFORE its JS renders anything - so bounding-box checks
        alone report 'rendered' for a blank box. Require real challenge content
        (painted image tiles, prompt/header text, or an answer/verify control)
        before claiming ready - a loader shell has none of these markers.
        """
        return await self._frame_js_ready(iframe, """() => {
            // hCaptcha streams challenge assets; accept a parsed (interactive)
            // or fully-loaded (complete) document as long as REAL challenge
            // content is present. A bare loader shell never has the markers
            // checked below.
            if (document.readyState !== 'complete' &&
                document.readyState !== 'interactive') return false;

            const sized = (el, min) => {
                if (!el) return false;
                try {
                    const r = el.getBoundingClientRect();
                    return !!(r && r.width >= (min || 1) && r.height >= (min || 1));
                } catch (e) { return false; }
            };

            // Image tiles: count real <img> nodes AND background-image divs
            // (hCaptcha's .task-image grid uses CSS backgrounds, so a fully
            // rendered challenge can contain NO <img> nodes at all).
            let tiles = 0;
            for (const img of document.querySelectorAll('img')) {
                if (sized(img, 12)) tiles += 1;
            }
            if (tiles < 4) {
                for (const el of document.querySelectorAll(
                        '.task-image, .challenge-image, [class*="task-image"], ' +
                        '[class*="challenge-image"], [class*="image-grid"], ' +
                        '[class*="image"]')) {
                    let painted = false;
                    try {
                        const cs = getComputedStyle(el);
                        painted = !!(cs && cs.backgroundImage &&
                                     cs.backgroundImage !== 'none');
                    } catch (e) {}
                    if (painted || sized(el, 12)) tiles += 1;
                }
            }
            if (tiles >= 4) return true;

            const body = document.body;
            const prompt = document.querySelector(
                '.prompt-text, .prompt, .header, [class*="prompt"], ' +
                '[class*="challenge-description"], [class*="instruction"]');
            const promptText = ((prompt && (prompt.innerText || prompt.textContent)) ||
                (body && body.innerText || '')).trim();

            // hCaptcha's painted challenge header always carries the
            // About/Accessibility menu button; a loader shell never does.
            const hasMenu = !!document.querySelector(
                '#menu-info, .display-menu-btn, [aria-label*="About hCaptcha"]');
            const hasAnswer = !!document.querySelector(
                'input[type="text"], textarea, [class*="answer"]');
            const hasVerify = !!document.querySelector(
                'button[type="submit"], .button-submit, [class*="submit"], ' +
                '[class*="verify"], .button-verify');

            if (hasMenu) return true;
            return promptText.length >= 8 && (tiles >= 1 || hasAnswer || hasVerify);
        }""")

    async def _wait_for_image_challenge(self, timeout: float = 30.0):
        """Wait until the hCaptcha challenge frame really paints its image grid.

        Returns the challenge iframe locator once rendered, else None. The
        sitekey is readable from the widget frame long before the challenge
        spawns, but solving that early mints a token before hCaptcha's
        getcaptcha request has delivered the rqdata the token must be bound to.
        """
        deadline = time.time() + timeout
        chall = self._page.locator(
            'iframe[title*="hCaptcha challenge"], '
            'iframe[src*="hcaptcha-challenge"]')
        while time.time() < deadline:
            try:
                n = await chall.count()
                for i in range(n):
                    c = chall.nth(i)
                    box = await c.bounding_box()
                    if (box and box.get("height", 0) >= 80
                            and await self._challenge_rendered(c)):
                        return c
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return None

    async def _widget_rendered(self, iframe) -> bool:
        """True when the hCaptcha widget iframe is genuinely ready to click.

        Readiness is hCaptcha's own signal: the widget body is marked
        aria-hidden="true" until its JS has finished rendering the UI, then
        drops it once painted. A still-hidden body is NOT ready. Once that
        signal is satisfied, the presence of any real widget node (checkbox,
        toolbar trigger, refresh button or logo) is enough — the new widget
        lays its checkbox out in a way getBoundingClientRect() reports as
        0-sized, so geometry must never gate readiness.
        """
        return await self._frame_js_ready(iframe, """() => {
            const body = document.body;
            if (!body) return false;
            if (document.readyState !== 'complete') return false;
            // Ground truth for readiness: a real widget UI node (checkbox,
            // toolbar trigger, refresh button or logo) proves hCaptcha
            // painted the widget. Some widget builds keep the body
            // aria-hidden="true" even after painting (field probe showed
            // ariaHidden:"true", children:2, checkbox:true,
            // readyState:"complete" with the full widget DOM present), so
            // aria-hidden must never block readiness once a node exists.
            if (document.querySelector(
                    '#checkbox, .checkbox, [role="checkbox"], input[type="checkbox"], ' +
                    '[aria-checked], .button-submit, #menu-info, .display-menu-btn, ' +
                    '.refresh.button, .hcaptcha-logo')) return true;
            // No widget node yet — only now does body aria-hidden mean the
            // widget is still on the loader stage.
            if (body.getAttribute('aria-hidden') === 'true') return false;
            const laidOut = (el) => {
                if (!el) return false;
                const cs = getComputedStyle(el);
                if (cs.display === 'none' || cs.visibility === 'hidden') return false;
                const r = el.getBoundingClientRect();
                return !!r && r.width > 1 && r.height > 1;
            };
            // 1) The checkbox itself - the real click target.
            for (const sel of ['#checkbox', '.checkbox', '[role="checkbox"]',
                               'input[type="checkbox"]', '[aria-checked]',
                               '.button-submit']) {
                const els = document.querySelectorAll(sel);
                for (const el of els) {
                    if (laidOut(el)) return true;
                }
            }
            // 2) The widget toolbar (menu trigger / refresh / logo) proves
            //    the widget painted even while the checkbox is mid-render.
            for (const sel of ['#menu-info', '.display-menu-btn',
                               '.refresh.button', '.hcaptcha-logo']) {
                if (laidOut(document.querySelector(sel))) return true;
            }
            // 3) Any rendered text (e.g. the "I am human" label).
            const t = (body.innerText || '').trim();
            return t.length >= 3;
        }""")

    async def _widget_has_checkbox(self, iframe) -> bool:
        """Cheap probe: does the widget frame contain a checkbox node at all?

        Used to ALWAYS attempt the click even when the strict readiness
        probe (_widget_rendered) hasn't flipped yet — hCaptcha keeps some
        widget builds aria-hidden="true" while the checkbox is already
        painted and interactive, so readiness must never gate the attempt.
        """
        try:
            frame = await (await iframe.element_handle(timeout=4000)).content_frame()
            if frame is None:
                return False
            return bool(await frame.evaluate(
                "() => !!document.querySelector("
                "'#checkbox, [role=\"checkbox\"], .checkbox, "
                "input[type=\"checkbox\"], [aria-checked], .button-submit')"))
        except Exception:
            return False


    async def _widget_error_state(self, iframe) -> str:
        """If the hCaptcha widget iframe is showing hCaptcha's OWN error
        banner ("Rate limited or network error. Please retry.") return the
        banner text.

        hCaptcha renders this INSIDE the widget when its backend rejects the
        session (flagged IP, dead circuit, blocked hcaptcha.com API). The
        checkbox exists but is inert -- no click will ever register, because
        hCaptcha never initialized the widget. Returns "" when healthy.
        """
        frame = await self._hcaptcha_frame_for(iframe)
        if frame is None:
            return ""
        try:
            text = await frame.evaluate(
                "() => (document.body ? document.body.innerText : '')")
        except Exception:
            return ""
        low = (text or "").lower()
        for kw in ("rate limited or network error", "rate limited",
                   "network error", "please retry", "please try again",
                   "automated queries"):
            if kw in low:
                return (text or "").strip()[:120]
        return ""

    async def _retry_erroring_widget(self, iframe) -> bool:
        """Click hCaptcha's own retry/refresh control inside the widget frame
        so a transient network error gets one honest second chance.

        The refresh button (.refresh.button) reloads the widget; an explicit
        Retry link also appears in the error state. Returns True if any
        control was clicked.
        """
        frame = await self._hcaptcha_frame_for(iframe)
        if frame is None:
            return False
        try:
            for sel in (".refresh.button", "[aria-label*='Refresh']",
                        "button[aria-label*='Refresh']", "a:has-text('Retry')",
                        "div:has-text('Please try again')"):
                loc = frame.locator(sel).first
                try:
                    await loc.click(timeout=1500, force=True)
                    return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    async def _click_hcaptcha_checkbox(self, iframe) -> bool:
        """CLICK the hCaptcha 'Are you human' checkbox — always attempts.

        Never gated on the strict readiness probe: if the widget frame
        contains a checkbox node we click it, period. The current hCaptcha
        widget lays the checkbox out in a way getBoundingClientRect()
        reports as 0-sized, so geometry never gates the attempt either —
        the real click point is computed in-page (walking the checkbox's
        subtree, then ancestors, for the first sized element) and translated
        to page coordinates via the iframe's bounding box. Click order:

          1. Real mouse click (CDP input at the point — hCaptcha trusts it)
          2. Keyboard activation (focus the checkbox + Enter/Space — a
             role=checkbox is natively keyboard-activatable, so this works
             with zero coordinate math)
          3. JS el.click() dispatch

        Every attempt is verified against hCaptcha's own signals
        (aria-checked=true flip, challenge iframe spawn, or token) and the
        whole sequence retries up to 5 times.
        """
        frame = await self._hcaptcha_frame_for(iframe)
        if frame is None:
            self._log("[Captcha] Checkbox click skipped — no live hCaptcha frame attached",
                      level="debug")
            return False

        full_src = ""
        try:
            full_src = await iframe.get_attribute("src") or ""
        except Exception:
            full_src = ""
        iframe_src = full_src[:80] or "?"
        self._log(f"[Captcha] Clicking hCaptcha checkbox (iframe: {iframe_src})")

        # Inspect what's actually inside the widget frame (ALL LOGS only).
        try:
            probe = await frame.evaluate("""() => {
                const body = document.body;
                const anyCheckbox = !!document.querySelector(
                    '#checkbox, [role="checkbox"], .checkbox, input[type="checkbox"], [aria-checked], .button-submit');
                const t = (body && body.innerText || '').slice(0, 80);
                return JSON.stringify({
                    ariaHidden: body ? body.getAttribute('aria-hidden') : null,
                    children: body ? body.children.length : -1,
                    anyCheckbox,
                    readyState: document.readyState,
                    text: t.replace(/\s+/g, ' ').trim()
                });
            }""")
            self._log(f"[Captcha] Widget frame probe: {probe}", level="debug")
        except Exception as e:
            self._log(f"[Captcha] Widget frame probe error: {e}", level="debug")

        # ── Verification: only a click hCaptcha actually reacted to counts ──
        # hCaptcha confirms a registered click by flipping the checkbox's
        # aria-checked to "true", spawning the challenge iframe, or writing a
        # token. A locator/force click on a 0-sized or covered element can
        # "succeed" without hCaptcha ever reacting — never claim victory on
        # that. Poll the three signals for ~2s after each attempt.
        async def _confirm(attempt: str) -> bool:
            for _ in range(5):
                try:
                    flipped = await frame.evaluate(
                        "() => { const el = document.querySelector('[aria-checked]');"
                        " return !!el && el.getAttribute('aria-checked') === 'true'; }")
                    if flipped:
                        self._log(f"[Captcha] [OK] Checkbox {attempt} — hCaptcha confirmed (aria-checked=true)")
                        return True
                except Exception:
                    pass
                try:
                    chall = self._page.locator(
                        'iframe[title*="hCaptcha challenge"], '
                        'iframe[src*="hcaptcha-challenge"]')
                    if await chall.count() > 0:
                        self._log(f"[Captcha] [OK] Checkbox {attempt} — hCaptcha confirmed (challenge spawned)")
                        return True
                except Exception:
                    pass
                try:
                    if await read_hcaptcha_token(self._page):
                        self._log(f"[Captcha] [OK] Checkbox {attempt} — hCaptcha confirmed (token present)")
                        return True
                except Exception:
                    pass
                await asyncio.sleep(0.4)
            return False

        # ── Real click point inside the frame (frame-relative coords) ──
        # getBoundingClientRect() can report 0x0 for the checkbox node even
        # when it is painted and interactive; walk the subtree (then the
        # ancestors) for the first sized element and click its center.
        point = None
        try:
            point = await frame.evaluate("""() => {
                const sels = ['[role="checkbox"]', '#checkbox', '.checkbox',
                              'input[type="checkbox"]', '[aria-checked]', '.button-submit'];
                let el = null;
                for (const s of sels) { el = document.querySelector(s); if (el) break; }
                if (!el) return null;
                const sized = (n) => {
                    if (!n) return null;
                    const r = n.getBoundingClientRect();
                    return (r && r.width > 0 && r.height > 0)
                        ? {left: r.left, top: r.top, width: r.width, height: r.height} : null;
                };
                let rect = sized(el);
                if (!rect) {
                    let best = null, bestArea = 0;
                    const walk = (n) => {
                        const r = sized(n);
                        if (r) { const a = r.width * r.height; if (a > bestArea) { best = r; bestArea = a; } }
                        for (const c of n.children) walk(c);
                    };
                    for (const c of el.children) walk(c);
                    if (best) rect = best;
                }
                if (!rect) {
                    let p = el.parentElement;
                    while (p) { const r = sized(p); if (r) { rect = r; break; } p = p.parentElement; }
                }
                if (!rect) return null;
                return {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2,
                        w: rect.width, h: rect.height};
            }""")
        except Exception:
            point = None
        if point:
            self._log(f"[Captcha] Checkbox center (frame coords): ({point['x']:.1f}, {point['y']:.1f}) "
                      f"size {point['w']:.1f}x{point['h']:.1f}", level="debug")

        # Translate frame coords → page coords via the iframe's bounding box.
        page_point = None
        try:
            iframe_box = await iframe.bounding_box()
        except Exception:
            iframe_box = None
        if point and iframe_box and iframe_box.get("width", 0) > 1:
            page_point = (iframe_box["x"] + point["x"], iframe_box["y"] + point["y"])
        elif iframe_box and iframe_box.get("width", 0) > 1:
            # No measurable checkbox rect — hCaptcha renders the checkbox at
            # the widget's left edge, vertically centered. Aim there.
            page_point = (iframe_box["x"] + iframe_box.get("width", 0) * 0.12,
                          iframe_box["y"] + iframe_box.get("height", 0) * 0.5)

        for attempt in range(1, 6):
            if attempt > 1:
                await asyncio.sleep(0.4)

            # Strategy 0: frame_locator click — the engine's reliable
            # cross-origin mechanism. Playwright resolves the frame lazily
            # and clicks the
            # checkbox center with trusted input, computing all iframe
            # offsets internally. hCaptcha mounts a hidden twin sharing the
            # same src — non-actionable elements just time out and we move
            # to the next checkbox, so the visible widget always gets hit.
            if full_src and "hcaptcha" in full_src:
                try:
                    fl = self._page.frame_locator(f'iframe[src="{full_src}"]')
                    fl_cb = fl.locator(
                        '#checkbox, [role="checkbox"], .checkbox, '
                        'input[type="checkbox"], [aria-checked], .button-submit')
                    for ci in range(min(4, await fl_cb.count())):
                        try:
                            await fl_cb.nth(ci).click(timeout=3000)
                            if await _confirm(f"frame click #{ci} (attempt {attempt})"):
                                return True
                        except Exception:
                            continue
                except Exception as e:
                    self._log(f"[Captcha] frame_locator click failed: {str(e)[:120]}",
                              level="debug")

            # Strategy 1: real mouse click at the computed page point.
            if page_point:
                try:
                    cx, cy = page_point
                    await self._page.mouse.move(cx, cy, steps=2)
                    await asyncio.sleep(random.uniform(0.15, 0.35))
                    await self._page.mouse.click(cx, cy)
                    if await _confirm(f"mouse click (attempt {attempt})"):
                        return True
                except Exception as e:
                    self._log(f"[Captcha] Mouse click failed: {str(e)[:120]}", level="debug")

            # Strategy 2: keyboard activation — role=checkbox is natively
            # activatable via Enter/Space; no coordinates involved.
            try:
                await frame.evaluate("""() => {
                    const el = document.querySelector('[role="checkbox"], #checkbox, .checkbox, [aria-checked], .button-submit');
                    if (el) el.focus();
                }""")
                await asyncio.sleep(0.1)
                await self._page.keyboard.press("Enter")
                if await _confirm(f"keyboard Enter (attempt {attempt})"):
                    return True
                await self._page.keyboard.press("Space")
                if await _confirm(f"keyboard Space (attempt {attempt})"):
                    return True
            except Exception as e:
                self._log(f"[Captcha] Keyboard activation failed: {str(e)[:120]}", level="debug")

            # Strategy 3: JS el.click() — hCaptcha binds click listeners.
            try:
                js_clicked = await frame.evaluate("""() => {
                    const el = document.querySelector('[role="checkbox"], #checkbox, .checkbox, input[type="checkbox"], [aria-checked], .button-submit');
                    if (!el) return false;
                    el.click();
                    return true;
                }""")
                if js_clicked and await _confirm(f"JS click (attempt {attempt})"):
                    return True
            except Exception as e:
                self._log(f"[Captcha] JS click failed: {str(e)[:120]}", level="debug")

        # Nothing registered — dump the frame DOM to ALL LOGS so the user can
        # see exactly what hCaptcha rendered inside the widget.
        try:
            html = await frame.evaluate(
                "() => (document.body ? document.body.outerHTML : '').slice(0, 2000)")
            self._log(f"[Captcha] Checkbox click never confirmed — widget frame DOM:\n{html}",
                      level="debug")
        except Exception as e:
            self._log(f"[Captcha] Widget frame DOM dump failed: {e}", level="debug")
        return False

    async def _dump_captcha_dom(self, reason: str) -> None:
        """Dump the page + every hCaptcha iframe's DOM to the ALL LOGS.

        Debug-level only: visible with LOG_LEVEL=all or the dashboard ALL
        LOGS toggle, never in the default console.
        """
        try:
            url = self._page.url
        except Exception:
            url = "?"
        self._log(f"[DOM] Captcha DOM dump ({reason}) — page: {url[:90]}", level="debug")
        try:
            html = await self._page.evaluate(
                "() => (document.body ? document.body.outerHTML : '').slice(0, 2500)")
            self._log(f"[DOM] Page body:\n{html}", level="debug")
        except Exception as e:
            self._log(f"[DOM] Page body dump failed: {e}", level="debug")
        try:
            for f in self._page.frames:
                if 'hcaptcha' not in (f.url or ''):
                    continue
                try:
                    state = await f.evaluate("document.readyState")
                except Exception:
                    state = "?"
                # Per-iframe probe: aria-hidden + checkbox presence. The body
                # is mostly a giant minified loader script, so also dump the
                # TAIL of the body where the actual UI renders.
                try:
                    info = await f.evaluate("""() => {
                        const b = document.body;
                        const cb = document.querySelector(
                            '#checkbox, [role="checkbox"], .button-submit, input[type="checkbox"], [aria-checked]');
                        return JSON.stringify({
                            ariaHidden: b ? b.getAttribute('aria-hidden') : null,
                            children: b ? b.children.length : -1,
                            checkbox: !!cb,
                            readyState: document.readyState
                        });
                    }""")
                    self._log(f"[DOM] iframe probe {f.url[:70]}: {info}", level="debug")
                except Exception:
                    pass
                try:
                    fhtml = await f.evaluate(
                        "() => { const b = document.body || document.documentElement;"
                        " const h = b.outerHTML || ''; return h.slice(-1500); }")
                except Exception as e:
                    fhtml = f"<dump failed: {e}>"
                self._log(f"[DOM] iframe {f.url[:100]} readyState={state} (tail):\n{fhtml}",
                          level="debug")
        except Exception as e:
            self._log(f"[DOM] iframe dump failed: {e}", level="debug")

    async def _click_form_submit(self) -> bool:
        """Click Create Account / Continue after the captcha token is in place."""
        try:
            result = await self._page.evaluate("""() => {
                __LOGIN_LINK_GUARD__
                const _norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
                const btns = document.querySelectorAll('button');
                for (const btn of btns) {
                    if (btn.offsetParent === null) continue;
                    if (btn.disabled || btn.getAttribute('aria-disabled') === 'true') continue;
                    if (__isLoginLink(btn)) continue;
                    const t = _norm(btn.textContent);
                    // ALL locales: the submit label is localized (German "Konto
                    // erstellen", French "Créer un compte", Russian "Создать
                    // аккаунт"...), so match the common spellings.
                    if (RegExp(__SUBMIT_TEXT_RE__).test(t)) {
                        btn.scrollIntoView({block: 'center'});
                        btn.click();
                        return t.slice(0, 24);
                    }
                }
                const submit = document.querySelector('[type="submit"]');
                if (submit && submit.offsetParent !== null && submit.closest('form')
                    && !submit.disabled && !__isLoginLink(submit)) {
                    submit.click();
                    return 'submit_btn';
                }
                const form = document.querySelector('form');
                if (form) {
                    // requestSubmit() with no arg activates the form's default
                    // submit button — which is the "Already have an account?"
                    // login link when the real Continue is disabled. Pick a
                    // real, enabled, non-login submit button instead.
                    for (const sb of form.querySelectorAll('button[type="submit"], [type="submit"]')) {
                        if (sb.disabled || sb.getAttribute('aria-disabled') === 'true') continue;
                        if (sb.offsetParent === null) continue;
                        if (__isLoginLink(sb)) continue;
                        if (form.requestSubmit) { form.requestSubmit(sb); return 'requestSubmit'; }
                        sb.click();
                        return 'form_submit_click';
                    }
                }
                return '';
            }""".replace('__LOGIN_LINK_GUARD__', _LOGIN_LINK_GUARD)
                .replace('__SUBMIT_TEXT_RE__', json.dumps(_SUBMIT_TEXT_RE)))
            if result:
                self._log(f"[Captcha] [OK] Submit clicked: {result}")
                return True
        except Exception as e:
            self._log(f"[Captcha] submit click error: {e}", level="warn")
        return False

    async def _solve_hcaptcha_if_present(self) -> bool:
        """Detect and solve the hCaptcha challenge.

        Smart detection: polls the DOM every second with real JS introspection.
        Knows the difference between widget-loading, challenge-loading, and
        challenge-ready states. Waits up to 60s for the challenge to fully render.
        """
        try:
            self._log("[Captcha] Checking for hCaptcha...")

            if await self._past_captcha():
                self._log(f"[Captcha] Already past captcha - at {self._page.url[:50]}")
                return True

            # ── Phase 1: Wait for hCaptcha to actually LOAD ──
            # Never fake: readiness is never claimed from static DOM. Two
            # honest paths to ready:
            #   · a rendered challenge iframe (height >= 80) — a challenge is
            #     actively showing, solve it directly;
            #   · the widget iframe with its document loaded — then hand it
            #     to the checkbox click pass below.
            # IMPORTANT: we deliberately do NOT click the widget checkbox to
            # "test" readiness — clicking burns an hCaptcha attempt and puts
            # the challenge into a permanent "Please try again" state that
            # can never be solved (verified in the field).
            self._log("[Captcha] Waiting for hCaptcha to load...")
            iframe = None
            loop_start = time.time()
            widget_since = None  # when the widget iframe first appeared
            checkbox_clicked = False    # auto-clicked the hCaptcha checkbox
            checkbox_clicked_at = 0.0   # when the checkbox was clicked
            checkbox_passes = 0         # unconditional click passes run (max 2)
            funcaptcha_checked = False
            honest_logged = {8: False, 20: False}
            # If hCaptcha never loads (dead session / blocked scripts), do
            # not loop forever: rotate after 45s so the worker retries on a
            # fresh circuit.
            no_widget_deadline = loop_start + 45

            while True:
                if await self._past_captcha():
                    self._log(f"[Captcha] Already past captcha — at {self._page.url[:50]}")
                    return True

                # ── Fast-fail: rate limiting (rotate the moment it shows) ──
                try:
                    body = await self._page.evaluate(
                        "() => (document.body ? document.body.innerText : '').toLowerCase()")
                except Exception:
                    body = ""
                if any(k in body for k in _RATE_LIMIT_KEYWORDS):
                    self._log("[Captcha] RATE LIMITED — rotating circuit", level="warn")
                    return False

                # 1) Challenge iframe already rendered → genuinely ready.
                #    NEVER trust size alone: a blank/loading challenge iframe
                #    is laid out at full size before its JS paints. Only claim
                #    [READY] when the frame document is complete AND actually
                #    shows hCaptcha content.
                try:
                    chall = self._page.locator(
                        'iframe[title*="hCaptcha challenge"], iframe[src*="hcaptcha-challenge"]')
                    if await chall.count() > 0:
                        box = await chall.first.bounding_box()
                        if (box and box.get("height", 0) >= 80
                                and await self._challenge_rendered(chall.first)):
                            iframe = chall.first
                            self._log("[Captcha] [READY] hCaptcha challenge already rendered")
                            break
                except Exception:
                    pass

                # 2) Widget present → scan EVERY hCaptcha iframe. Discord
                #    mounts several (a hidden aria-hidden frame plus the real
                #    widget), and widget.first is not always the visible one.
                #    Find the first genuinely-rendered frame, auto-click its
                #    checkbox (user request) so hCaptcha spawns the challenge.
                try:
                    widgets = self._page.locator(
                        'iframe[title="Widget containing checkbox for hCaptcha security challenge"], '
                        'iframe[src*="newassets.hcaptcha.com"], '
                        'iframe[src*="hcaptcha.com"][src*="frame=checkbox"]'
                    )
                    wcount = await widgets.count()
                    # Once the challenge iframe is on the page (even still
                    # loading), never touch the widget checkbox again: a
                    # coordinate click would land on the challenge modal
                    # (its X / backdrop) and dismiss it.
                    if wcount > 0 and (await self._challenge_iframe()) is None:
                        if widget_since is None:
                            widget_since = time.time()
                            self._log(f"[Captcha] hCaptcha widget present ({wcount} iframes) — waiting for it to initialize...")
                        # Sitekey: extract + log it once per session (user
                        # request — the widget src carries sitekey=).
                        if not getattr(self, "_hcaptcha_sitekey", ""):
                            try:
                                sk = await extract_hcaptcha_sitekey(self._page)
                                if sk:
                                    self._hcaptcha_sitekey = sk
                                    self._log(f"[Captcha] Sitekey: {sk}")
                            except Exception as e:
                                self._log(f"[Captcha] Sitekey extraction error: {e}", level="debug")
                        # Checkbox-only pass: token already present, no
                        # challenge needed.
                        if await read_hcaptcha_token(self._page):
                            self._log("[Captcha] [OK] hCaptcha already solved (token present)")
                            return True
                        # ── WIDGET-ERROR FAST-FAIL (root cause) ──
                        # When the widget itself shows "Rate limited or
                        # network error. Please retry." the checkbox is
                        # INERT -- hCaptcha's backend rejected this session
                        # (flagged IP / dead circuit), so no click will ever
                        # register. Retry the widget once (transient network
                        # errors recover), then rotate immediately instead of
                        # burning the 45s watchdog on a blocked session.
                        widget_err = ""
                        err_wi = -1
                        for wi in range(wcount):
                            widget_err = await self._widget_error_state(widgets.nth(wi))
                            if widget_err:
                                err_wi = wi
                                break
                        if widget_err:
                            self._log(
                                f"[Captcha] hCaptcha widget error: {widget_err!r}",
                                level="warn")
                            retried = await self._retry_erroring_widget(
                                widgets.nth(err_wi))
                            await asyncio.sleep(2.5)
                            still_err = await self._widget_error_state(
                                widgets.nth(err_wi))
                            if still_err:
                                self._log(
                                    f"[Captcha] Widget still erroring after retry "
                                    f"(retried={retried}) — rotating circuit NOW",
                                    level="warn")
                                return False
                            self._log(
                                "[Captcha] Widget recovered after retry — continuing")
                        # ── ALWAYS-CLICK PASS (user request) ──
                        # Click the checkbox UNCONDITIONALLY whenever any
                        # hCaptcha iframe is present — the readiness probes
                        # are informational only and must NEVER gate the
                        # attempt (they can report False via
                        # Locator.content_frame() even when the widget is
                        # interactive; the click helper itself falls back to
                        # matching the frame from the page's frame tree).
                        # `_click_hcaptcha_checkbox` is self-verifying: it
                        # tries mouse → keyboard → JS click and only reports
                        # success on hCaptcha's own signals (aria-checked
                        # flip, challenge spawn, or token). Run 2 full
                        # passes right after "Waiting for hCaptcha to
                        # load..." so a mid-init widget still gets clicked
                        # once it becomes interactive.
                        if not checkbox_clicked and checkbox_passes < 2:
                            checkbox_passes += 1
                            self._log(
                                f"[Captcha] Checkbox click pass {checkbox_passes}/2 (unconditional — widget present)")
                            for wi in range(wcount):
                                w = widgets.nth(wi)
                                # ONLY click frames that actually contain a
                                # checkbox node. The pre-init shell frame
                                # (children:0, anyCheckbox:false) has nothing
                                # to click — every strategy dies on "Frame
                                # was detached" and we'd spin forever.
                                if not await self._widget_has_checkbox(w):
                                    continue
                                if await self._click_hcaptcha_checkbox(w):
                                    checkbox_clicked = True
                                    checkbox_clicked_at = time.time()
                                    self._log("[Captcha] Checkbox clicked — waiting for challenge to spawn...")
                                    break
                            if not checkbox_clicked:
                                self._log(
                                    "[Captcha] Widget frames present but no checkbox node yet — waiting for hCaptcha to initialize",
                                    level="debug")
                        if checkbox_clicked and (time.time() - checkbox_clicked_at) < 5.0:
                            # hCaptcha swaps to the challenge a moment
                            # after the click — the next loop iteration
                            # (0.25s) catches the painted challenge iframe.
                            continue
                        # Hand the first genuinely-rendered widget to the
                        # checkbox click pass.
                        rendered_widget = None
                        for wi in range(wcount):
                            w = widgets.nth(wi)
                            if await self._widget_rendered(w):
                                rendered_widget = w
                                break
                        if rendered_widget is None and checkbox_passes >= 2:
                            # Both click passes ran without confirmation and
                            # the readiness probe still fails — hand the
                            # widget off anyway: the challenge may still
                            # rendered sitekey + page URL, not the frame.
                            self._log(
                                "[Captcha] Readiness probe failed after 2 click passes - proceeding anyway",
                                level="warn")
                            rendered_widget = widgets.nth(0)
                        if rendered_widget is not None:
                            iframe = rendered_widget
                            self._log("[Captcha] [READY] hCaptcha widget rendered - clicking checkbox to spawn challenge")
                            break
                except Exception:
                    pass

                # 3) FunCAPTCHA (Arkose) escape: no hcaptcha after 15s but
                #    captcha-ish text on page → pixel solver.
                elapsed = time.time() - loop_start
                if (not iframe and elapsed > 15.0
                        and not funcaptcha_checked):
                    funcaptcha_checked = True
                    if not any('hcaptcha' in (f.url or '') for f in self._page.frames):
                        try:
                            page_text = await self._page.evaluate(
                                "() => (document.body ? document.body.innerText.substring(0, 500) : '')")
                            low = page_text.lower()
                            if ('captcha' in low or 'security' in low or 'verify' in low):
                                self._log("[Captcha] No hCaptcha frames — trying FunCAPTCHA solver", level="warn")
                                return await self._solve_funcaptcha()
                        except Exception:
                            pass

                # ── Honest progress — never claim to be solving a captcha
                # that hasn't loaded ──
                for threshold, flag in ((8, 8), (20, 20)):
                    if elapsed > threshold and not honest_logged[flag]:
                        honest_logged[flag] = True
                        if widget_since is not None:
                            self._log(
                                f"[Captcha] hCaptcha widget still initializing after {int(elapsed)}s...",
                                level="warn")
                        else:
                            self._log(
                                f"[Captcha] No hCaptcha widget after {int(elapsed)}s — hCaptcha script not loaded yet...",
                                level="warn")
                        await self._dump_captcha_dom(f"stuck at {int(elapsed)}s")

                # Watchdog: hCaptcha never loaded — rotate honestly.
                if time.time() > no_widget_deadline:
                    if widget_since is not None:
                        self._log("[Captcha] hCaptcha widget never became ready in 45s — "
                                  "scripts blocked or session stalled, rotating", level="warn")
                    else:
                        self._log("[Captcha] No hCaptcha widget in 45s — script blocked or dead "
                                  "session, rotating", level="warn")
                    await self._dump_captcha_dom("45s watchdog")
                    return False

                # Fast poll — hCaptcha paints within a few hundred ms of its
                # document completing, so 0.25s catches it almost immediately.
                await asyncio.sleep(0.25)

            if not iframe:
                # No hCaptcha iframe - check for FunCAPTCHA (Arkose) instead
                try:
                    if await self._past_captcha():
                        self._log(f"[Captcha] Registration went through - at {self._page.url[:50]}")
                        return True
                    page_text = await self._page.evaluate(
                        "() => document.body.innerText.substring(0, 500)")
                    has_captcha_text = ('captcha' in page_text.lower()
                                        or 'security' in page_text.lower()
                                        or 'verify' in page_text.lower())
                    if has_captcha_text:
                        self._log("[Captcha] FunCAPTCHA detected - pixel tile solver...")
                        return await self._solve_funcaptcha()
                    self._log(f"[Captcha] No captcha indicators on page: {self._page.url[:40]}", level="warn")
                    return False
                except Exception as e:
                    self._log(f"[Captcha] Captcha check error: {e}", level="warn")
                return False

            # ---- VISION SOLVER: reads the image grid via vision_solver ----
            # The bot reads the challenge instruction, screenshots every
            # the challenge instruction, screenshots every tile, asks a
            # vision model (local Ollama or remote VISION_API_BASE endpoint)
            # which tiles match, clicks them + Verify, and hCaptcha itself
            # mints the token. See vision_solver.py for the recommended
            # model (ahmadwaqar/smolvlm2-256m-video:q8_0).
            if await self._past_captcha():
                self._log("[Captcha] Page already past captcha")
                return True
            self._log("[Captcha] Waiting for the image challenge to fully render...")
            if not await self._wait_for_image_challenge(timeout=30):
                self._log("[Captcha] Image challenge never rendered - rotating",
                          level="warn")
                await self._dump_captcha_dom("image challenge timeout")
                return False
            self._log("[Captcha] [READY] Image challenge rendered - reading prompt + tiles")

            # Probe the vision endpoint with RETRIES and NO permanent
            # failure cache. Hosted endpoints (Railway etc.) cold-start on
            # the first request after sleeping — one fast probe would mark
            # a healthy service down for the bot's whole lifetime (the
            # "Ollama server unreachable" that kills every captcha round).
            # _vision_ready is only set on success, so a service that was
            # down and comes back re-probes cleanly on the next challenge.
            if not getattr(self, "_vision_ready", False):
                ok, models = False, []
                # Authentication/protocol failures are deterministic: waiting
                # and replaying the same request cannot fix them.  Only retry
                # transient connection, timeout, rate-limit, and 5xx failures.
                terminal_probe_errors = {"authentication", "authorization", "protocol"}
                probes_made = 0
                for _probe in range(3):
                    probes_made = _probe + 1
                    ok, models = await self._vision.check()
                    if ok:
                        break
                    check_error = getattr(self._vision, "last_check_error", "")
                    if check_error in terminal_probe_errors:
                        break
                    if _probe < 2:
                        self._log(
                            f"[Captcha] Vision endpoint temporarily unavailable "
                            f"(probe {_probe + 1}/3, {check_error or 'unknown error'}) - "
                            "retrying in 10s", level="warn")
                        await asyncio.sleep(10)
                if not ok:
                    check_error = getattr(self._vision, "last_check_error", "")
                    http_status = getattr(self._vision, "last_check_http_status", None)
                    if check_error == "authentication":
                        self._log(
                            f"[Captcha] Vision endpoint is UP but rejected authentication "
                            f"(HTTP {http_status or 401}). This app and the Vision AI "
                            "service have different or missing VISION_API_KEY values. "
                            "Configure both Railway services from the same shared variable; "
                            "do not put the secret in logs.", level="error")
                    elif check_error == "authorization":
                        self._log(
                            f"[Captcha] Vision endpoint denied access "
                            f"(HTTP {http_status or 403}); check the gateway authorization "
                            "policy.", level="error")
                    elif check_error == "protocol":
                        self._log(
                            f"[Captcha] Vision endpoint at {self._vision.base} is reachable "
                            "but is not a compatible Ollama API; expected GET /api/tags.",
                            level="error")
                    else:
                        self._log(
                            f"[Captcha] Vision server unavailable after {probes_made} "
                            f"probe(s) at {self._vision.base} ({check_error or 'unknown error'}) "
                            "- check VISION_API_BASE / service status. Later challenges "
                            "will re-probe automatically.", level="error")
                    # Do not issue several expensive /api/chat requests after
                    # the readiness probe already proved they cannot succeed.
                    return False
                elif self._vision.model not in models:
                    self._log(
                        f"[Captcha] Ollama model {self._vision.model} not pulled - "
                        f"run: ollama pull {self._vision.model}",
                        level="warn")
                else:
                    self._vision_ready = True

            for solve_attempt in range(3):
                if solve_attempt:
                    await asyncio.sleep(3)
                    self._log(
                        f"[Captcha] Retrying vision solve (attempt {solve_attempt + 1}/3)...",
                        level="warn")
                if await self._past_captcha():
                    self._log("[Captcha] Page already past captcha")
                    return True
                # hCaptcha can show several rounds in a row - keep solving
                # until it mints the token or the challenge resets. Each
                # round is CLASSIFIED first: hCaptcha has five challenge
                # families (grid binary, reference binary, point, bbox,
                # drag, multiple choice, text) and answering a point round
                # with tile indices (or vice versa) is a guaranteed fail —
                # and a loud automation signal.
                last_sig = None
                for round_i in range(8):
                    if await self._past_captcha():
                        return True
                    chall = await self._challenge_iframe()
                    if chall is None or not await self._challenge_rendered(chall):
                        # After Next the iframe often drops to a loader.
                        # Wait for the next challenge to paint — do NOT
                        # treat the dip as "challenge over".
                        chall = await self._wait_for_image_challenge(timeout=10.0)
                    if chall is None:
                        if await read_hcaptcha_token(self._page):
                            self._log(
                                "[Captcha] [OK] hCaptcha token minted by the solved "
                                "challenge - submitting form")
                            await self._click_form_submit()
                            for _ in range(8):
                                await asyncio.sleep(1.0)
                                if await self._past_captcha():
                                    return True
                        break
                    frame = await self._hcaptcha_frame_for(chall)
                    if frame is None:
                        await asyncio.sleep(1.5)
                        continue
                    dom = await self._probe_challenge_dom(frame)
                    prompt = await self._read_challenge_prompt(frame)
                    if not prompt:
                        # DOM prompt empty (still painting?) — the payload's
                        # requester_question is a solid fallback.
                        prompt = hct.question_text(self._challenge_payload or {})
                    if not prompt:
                        self._log("[Captcha] Prompt not readable yet (new round loading?)",
                                  level="warn")
                        await asyncio.sleep(2)
                        continue
                    family = hct.classify(self._challenge_payload, dom, prompt)
                    # Live tower wording is a Move-badge drag even when
                    # /getcaptcha labelled the round image_label_area_select.
                    if hct.is_tower_prompt(prompt) and family != hct.DRAG_DROP:
                        self._log(
                            f"[Captcha] Tower wording — forcing drag-drop "
                            f"(was {family})")
                        family = hct.DRAG_DROP
                    sig = (prompt, family, int((dom or {}).get("tiles") or 0))
                    # Same prompt + same layout: we already answered this
                    # grid. Re-clicking tiles TOGGLES them off. Just hit
                    # Next again and wait for the next challenge.
                    if last_sig is not None and sig == last_sig:
                        self._log(
                            "[Captcha] Same challenge still showing — clicking Next again")
                        await self._click_challenge_verify(frame)
                        await asyncio.sleep(1.2)
                        continue
                    self._log(
                        f"[Captcha] Challenge round {round_i + 1} "
                        f"[{family}/{hct.answer_shape(family)}]: {prompt[:120]}")
                    if family == hct.DRAG_DROP:
                        if hct.is_tower_prompt(prompt):
                            ok = await self._solve_tower_round(frame, prompt)
                        elif hct.is_pattern_prompt(prompt):
                            ok = await self._solve_pattern_round(frame, prompt)
                        else:
                            ok = await self._solve_drag_round(frame, prompt)
                    elif family == hct.AREA_POINT:
                        ok = await self._solve_point_round(frame, prompt,
                                                           bbox=False)
                    elif family == hct.AREA_BBOX:
                        ok = await self._solve_point_round(frame, prompt,
                                                           bbox=True)
                    elif family == hct.MULTIPLE_CHOICE:
                        ok = await self._solve_choice_round(frame, prompt)
                    elif family == hct.TEXT_ENTRY:
                        ok = await self._solve_text_round(frame, prompt)
                    elif family == hct.COUNT:
                        ok = await self._solve_count_round(frame, prompt)
                    else:
                        ok = await self._solve_binary_round(frame, prompt, dom)
                    if not ok:
                        self._log("[Captcha] Round not solved - retrying",
                                  level="warn")
                        await asyncio.sleep(2)
                        continue
                    await asyncio.sleep(0.7)
                    await self._click_challenge_verify(frame)
                    last_sig = sig
                    # Wait for hCaptcha to accept (token minted) OR paint
                    # the next challenge (new prompt / new layout).
                    token_seen = False
                    advanced = False
                    for _ in range(14):
                        await asyncio.sleep(0.7)
                        if await self._past_captcha():
                            self._log("[Captcha] [OK] Vision solve ACCEPTED - past captcha")
                            return True
                        if await read_hcaptcha_token(self._page):
                            token_seen = True
                            self._log(
                                "[Captcha] [OK] hCaptcha token minted by the solved "
                                "challenge - submitting form")
                            await self._click_form_submit()
                            break
                        c2 = await self._challenge_iframe()
                        if c2 is None or not await self._challenge_rendered(c2):
                            continue
                        f2 = await self._hcaptcha_frame_for(c2)
                        if f2 is None:
                            continue
                        p2 = await self._read_challenge_prompt(f2)
                        if not p2:
                            p2 = hct.question_text(self._challenge_payload or {})
                        d2 = await self._probe_challenge_dom(f2)
                        fam2 = hct.classify(self._challenge_payload, d2, p2)
                        sig2 = (p2, fam2, int((d2 or {}).get("tiles") or 0))
                        if p2 and sig2 != last_sig:
                            self._log(
                                f"[Captcha] Next challenge ready: {p2[:80]}")
                            advanced = True
                            break
                    if token_seen:
                        for _ in range(8):
                            await asyncio.sleep(1.0)
                            if await self._past_captcha():
                                self._log(
                                    "[Captcha] [OK] Registration submitted after "
                                    "vision solve")
                                return True
                        self._log(
                            "[Captcha] Form submitted after solve but not past captcha "
                            "yet - retrying",
                            level="warn")
                    elif advanced:
                        continue
                self._log("[Captcha] Vision solve not accepted across rounds - retrying",
                          level="warn")
            self._log("[Captcha] [FAIL] Vision solver could not clear the challenge",
                      level="error")
            await asyncio.sleep(2)
            return False

        except Exception as e:
            self._log(f"[Captcha] Flow error: {e}", level="error")
            import traceback
            traceback.print_exc()
            return False

    async def _read_challenge_prompt(self, frame) -> str:
        """Read the challenge instruction from the hCaptcha challenge frame."""
        try:
            raw = await frame.evaluate("""() => {
                const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                const seen = new Set();
                const out = [];
                const cands = document.querySelectorAll(
                    '.challenge-prompt, .prompt-text, #prompt-text, .task-description, ' +
                    '#task-description, [class*="prompt" i], [class*="task-description" i], ' +
                    'h1, h2, [class*="challenge" i] p, [class*="instructions" i]');
                for (const el of cands) {
                    const t = norm(el.textContent);
                    if (t.length < 8 || seen.has(t)) continue;
                    seen.add(t);
                    out.push({ text: t, len: t.length });
                }
                if (!out.length) return '';
                out.sort((a, b) => b.len - a.len);
                return out[0].text.slice(0, 200);
            }""")
            return str(raw or "").strip()
        except Exception:
            return ""

    async def _screenshot_challenge_tiles(self, frame) -> list:
        """Screenshot every grid tile (reading order: top-left first)."""
        for sel in ('div.task-image, [class*="task-image" i]',
                    '.task-grid img, [class*="task-grid" i] img, '
                    '.challenge-content img'):
            try:
                loc = frame.locator(sel)
                n = await loc.count()
            except Exception:
                continue
            if n == 0:
                continue
            tiles = []
            for i in range(min(n, 12)):
                try:
                    b = await loc.nth(i).screenshot(timeout=8000)
                    if b:
                        tiles.append(b)
                except Exception:
                    continue
            if tiles:
                self._log(f"[Captcha] Captured {len(tiles)} grid tiles")
                return tiles
        return []

    # ── multi-family challenge helpers ────────────────────────────────────

    async def _probe_challenge_dom(self, frame) -> dict:
        """DOM fact counts (tiles/canvases/draggables/choices/inputs) for the
        DOM tier of the family classifier."""
        try:
            facts = await frame.evaluate(hct.DOM_PROBE_JS)
            return facts if isinstance(facts, dict) else {}
        except Exception:
            return {}

    async def _frame_origin(self, frame):
        """Page-space (x, y) offset of the challenge iframe itself — needed
        because JS getBoundingClientRect() inside the frame is frame-local."""
        try:
            box = await (await frame.frame_element()).bounding_box()
            if box:
                return (float(box["x"]), float(box["y"]))
        except Exception:
            pass
        return (0.0, 0.0)

    async def _capture_example_images(self, frame) -> list:
        """Screenshot the prompt-header REFERENCE images — hCaptcha serves
        these on 'reference' rounds ('...the item shown') and the old solver
        never captured them, answering those rounds blind."""
        for sel in ('.challenge-example', '.prompt-image',
                    '.challenge-prompt img', '[class*="example" i] img',
                    '[class*="example" i] div'):
            try:
                loc = frame.locator(sel)
                n = await loc.count()
            except Exception:
                continue
            out = []
            for i in range(min(n, 3)):
                try:
                    b = await loc.nth(i).screenshot(timeout=4000)
                    if b:
                        out.append(b)
                except Exception:
                    break
            if out:
                self._log(f"[Captcha] Captured {len(out)} reference/example image(s)")
                return out
        return []

    _SURFACE_JS = r"""() => {
        // largest visible click-surface element (area_select canvas or the
        // single big image of point/bbox rounds)
        let best = null, bestArea = 0;
        for (const el of document.querySelectorAll(
                'canvas, img, div.task-image, [class*="task-image" i], ' +
                '[class*="challenge-image" i], [class*="canvas" i]')) {
            const r = el.getBoundingClientRect();
            if (r.width < 60 || r.height < 60) continue;
            const area = r.width * r.height;
            if (area > bestArea) { bestArea = area; best = {
                x: r.left, y: r.top, width: r.width, height: r.height }; }
        }
        return best;
    }"""

    async def _challenge_surface(self, frame):
        """(screenshot bytes, page-space box) of the biggest canvas/image in
        the challenge frame — the click surface for point/bbox/drag rounds."""
        try:
            info = await frame.evaluate(self._SURFACE_JS)
        except Exception:
            info = None
        if not info or float(info.get("width", 0)) < 60:
            return None, None
        ox, oy = await self._frame_origin(frame)
        box = {"x": float(info["x"]) + ox, "y": float(info["y"]) + oy,
               "width": float(info["width"]), "height": float(info["height"])}
        try:
            raw = await frame.screenshot()
            from PIL import Image
            import io as _io
            im = Image.open(_io.BytesIO(raw)).convert("RGB")
            x0 = int(info["x"]); y0 = int(info["y"])
            crop = im.crop((x0, y0, x0 + int(info["width"]),
                            y0 + int(info["height"])))
            buf = _io.BytesIO()
            crop.save(buf, "JPEG", quality=92)
            return buf.getvalue(), box
        except Exception as e:
            self._log(f"[Captcha] Surface screenshot failed: {e}", level="debug")
            return None, None

    def _denorm(self, point, box):
        """Normalised 0..1 point -> page coordinates inside `box` (clamped)."""
        return hct.denorm(point, box)

    # lazy offline solvers — return None when torch/weights are absent
    def _tile_classifier(self):
        if self._cnn_tile is None:
            try:
                from tile_classifier import TileClassifier
                c = TileClassifier()
                self._cnn_tile = c if c.available else False
            except Exception:
                self._cnn_tile = False
        return self._cnn_tile or None

    def _point_locator(self):
        if self._cnn_point is None:
            try:
                from tile_classifier import PointLocator
                c = PointLocator()
                self._cnn_point = c if c.available else False
            except Exception:
                self._cnn_point = False
        return self._cnn_point or None

    def _drag_locator(self):
        if self._cnn_drag is None:
            try:
                from tile_classifier import DragLocator
                c = DragLocator()
                self._cnn_drag = c if c.available else False
            except Exception:
                self._cnn_drag = False
        return self._cnn_drag or None

    # ── per-family round solvers ───────────────────────────────────────────

    async def _solve_binary_round(self, frame, prompt, dom) -> bool:
        """image_label_binary (+ its reference-image affordance variant).

        Offline path first: the tile CNN labels every tile; when the mean
        confidence clears _CNN_MIN_CONF and the prompt resolves through the
        knowledge base, we click without any model-server round trip."""
        tiles = await self._screenshot_challenge_tiles(frame)
        if not tiles:
            return False
        examples = await self._capture_example_images(frame)
        cnn = self._tile_classifier()
        if cnn is not None:
            try:
                got = cnn.classify_many(tiles, with_conf=True)
                if len(got) == len(tiles) and got:
                    labels = [g[0] for g in got]
                    mean_conf = sum(g[1] for g in got) / len(got)
                    ex_label = None
                    if examples:
                        eg = cnn.classify_many(examples[:1])
                        if eg:
                            ex_label = eg[0][0]
                    if mean_conf >= _CNN_MIN_CONF:
                        idx = hct.resolve_semantic(prompt, labels,
                                                   example_label=ex_label)
                        if idx is not None:
                            self._log(
                                f"[Captcha] Offline CNN grid: {labels} "
                                f"(conf {mean_conf:.2f}, ref={ex_label}) -> {idx}")
                            if not idx:
                                # Understood empty round (no tile matches) —
                                # Verify/Next with nothing selected is correct.
                                return True
                            return await self._click_challenge_tiles(frame, idx)
                        self._log(
                            "[Captcha] Offline labels made but prompt not "
                            "understood — asking vision model", level="debug")
                    else:
                        self._log(
                            f"[Captcha] Offline grid confidence {mean_conf:.2f} "
                            f"< {_CNN_MIN_CONF:.2f} — asking vision model",
                            level="debug")
            except Exception as e:
                self._log(f"[Captcha] Offline grid path error: {e}",
                          level="debug")
        self._log(
            f"[Captcha] Asking Ollama ({self._vision.model}) which tiles match...")
        answer = await self._vision.solve(prompt, tiles, shape="tiles",
                                          examples=examples)
        if not answer:
            return False
        if answer.get("type") == "text":
            self._log(f"[Captcha] Text answer on grid round: {answer.get('text')!r}")
            return await self._type_challenge_answer(frame,
                                                     answer.get("text", ""))
        indices = [i for i in answer.get("indices", [])
                   if isinstance(i, int) and 1 <= i <= len(tiles)]
        self._log(f"[Captcha] Clicking tiles: {indices}")
        return await self._click_challenge_tiles(frame, indices)

    async def _solve_point_round(self, frame, prompt, bbox: bool = False) -> bool:
        """area_select: click a point — or, for bbox rounds, drag out the
        box's diagonal. Offline PointLocator first, vision model fallback."""
        shot, box = await self._challenge_surface(frame)
        if not shot or not box:
            return False
        pl = self._point_locator()
        if pl is not None and not bbox:
            try:
                point = None
                target = hct.extract_target(prompt)
                if hct.superlative_table(prompt) or not target:
                    rel = pl.locate_relational(
                        shot, prompt, verifier=self._tile_classifier())
                    if rel:
                        point = (rel[0], rel[1])
                        self._log(
                            f"[Captcha] Offline relational point -> "
                            f"{rel[2]} @ ({rel[0]:.2f},{rel[1]:.2f})")
                if point is None and target:
                    hit = pl.locate(shot, target)
                    if hit:
                        point = (hit[0], hit[1])
                        self._log(
                            f"[Captcha] Offline point '{target}' @ "
                            f"({hit[0]:.2f},{hit[1]:.2f}) conf={hit[2]:.2f}")
                if point is not None:
                    x, y = self._denorm(point, box)
                    await hm.click(self._page, x, y)
                    return True
            except Exception as e:
                self._log(f"[Captcha] Offline point path error: {e}",
                          level="debug")
        answer = await self._vision.solve(
            prompt, [shot], shape=("bbox" if bbox else "points"))
        if not answer:
            return False
        if bbox:
            if answer.get("type") != "bbox":
                return False
            bb = answer["bbox"]
            x1, y1 = self._denorm((bb["x1"], bb["y1"]), box)
            x2, y2 = self._denorm((bb["x2"], bb["y2"]), box)
            self._log(f"[Captcha] Drawing box ({x1:.0f},{y1:.0f})->"
                      f"({x2:.0f},{y2:.0f})")
            await hm.drag(self._page, (x1, y1), (x2, y2))
            return True
        if answer.get("type") == "points" and answer.get("points"):
            clicked = 0
            for pt in answer.get("points") or []:
                try:
                    x, y = self._denorm(pt, box)
                except Exception:
                    continue
                self._log(f"[Captcha] Point click at ({x:.0f},{y:.0f})")
                await hm.click(self._page, x, y)
                clicked += 1
                await asyncio.sleep(random.uniform(0.15, 0.38))
            return clicked > 0
        return False

    async def _solve_drag_round(self, frame, prompt) -> bool:
        """image_drag_drop: a REAL press/move/release drag (hCaptcha ignores
        synthetic clicks here — DragSolver only matches Arkose iframes, so
        these rounds used to be unreachable)."""
        shot, box = await self._challenge_surface(frame)
        if not shot or not box:
            return False
        dl = self._drag_locator()
        if dl is not None:
            try:
                got = dl.locate(shot)
                if got:
                    fx, fy = self._denorm(got["from"], box)
                    tx, ty = self._denorm(got["to"], box)
                    self._log(
                        f"[Captcha] Offline drag ({got['from'][0]:.2f},"
                        f"{got['from'][1]:.2f}) -> ({got['to'][0]:.2f},"
                        f"{got['to'][1]:.2f})")
                    await hm.drag(self._page, (fx, fy), (tx, ty))
                    return True
            except Exception as e:
                self._log(f"[Captcha] Offline drag path error: {e}",
                          level="debug")
        answer = await self._vision.solve(prompt, [shot], shape="drag")
        if not answer or answer.get("type") != "drag":
            return False
        fx, fy = self._denorm(answer["from"], box)
        tx, ty = self._denorm(answer["to"], box)
        self._log(f"[Captcha] Dragging piece ({fx:.0f},{fy:.0f}) -> "
                  f"({tx:.0f},{ty:.0f})")
        await hm.drag(self._page, (fx, fy), (tx, ty))
        return True

    _TOWER_PIECE_JS = r"""() => {
        const vis = (el) => !!(el) &&
            (el.offsetParent !== null || el.getClientRects().length > 0);
        const cands = [];
        for (const el of document.querySelectorAll(
                '[draggable="true"], [class*="drag" i], [class*="move" i], '
                + 'button, [role="button"], span, div')) {
            if (!vis(el)) continue;
            const t = ((el.textContent || '') + ' '
                + (el.getAttribute('aria-label') || '')).trim();
            const r = el.getBoundingClientRect();
            if (r.width < 10 || r.height < 10) continue;
            const isMove = /^\+?\s*move\s*$/i.test(t)
                || (/\bmove\b/i.test(t) && t.length <= 14);
            const cls = (el.className || '').toString();
            const isDrag = el.getAttribute('draggable') === 'true'
                || /drag/i.test(cls);
            if (!isMove && !isDrag) continue;
            cands.push({
                x: r.left, y: r.top, w: r.width, h: r.height,
                move: isMove ? 1 : 0,
            });
        }
        if (!cands.length) return null;
        cands.sort((a, b) => b.move - a.move || (b.x - a.x));
        return cands[0];
    }"""

    async def _tower_frame_shot(self, frame):
        """Full challenge-iframe screenshot + page-space box.

        The Move piece is often a *separate* DOM node to the right of
        the tower photo, so cropping the largest canvas misses it.
        """
        try:
            raw = await frame.screenshot()
        except Exception as e:
            self._log(f"[Captcha] Tower frame screenshot failed: {e}",
                      level="debug")
            raw = None
        if not raw:
            return None, None
        box = None
        try:
            el = await frame.frame_element()
            box = await el.bounding_box()
        except Exception:
            box = None
        if not box or not box.get("width"):
            ox, oy = await self._frame_origin(frame)
            try:
                wh = await frame.evaluate(
                    "() => ({w: window.innerWidth, h: window.innerHeight})")
                box = {"x": ox, "y": oy,
                       "width": float(wh.get("w") or 0),
                       "height": float(wh.get("h") or 0)}
            except Exception:
                box = None
        if not box or float(box.get("width") or 0) < 40:
            return None, None
        return raw, box

    async def _tower_piece_hint(self, frame, box):
        """Normalised centre of the Move / draggable control, if visible."""
        try:
            info = await frame.evaluate(self._TOWER_PIECE_JS)
        except Exception:
            info = None
        if not info or not info.get("w"):
            return None
        bw = float(box.get("width") or 0) or 1.0
        bh = float(box.get("height") or 0) or 1.0
        # Badge sits on/above the wood piece — grab slightly below centre.
        cx = (float(info["x"]) + float(info["w"]) / 2.0) / bw
        cy = (float(info["y"]) + float(info["h"]) / 2.0) / bh + 0.05
        return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))

    async def _solve_tower_round(self, frame, prompt) -> bool:
        """Wooden-block tower: drag the Move piece onto the incomplete stack.

        Live prompt: "Move the correct missing block segment onto the
        incomplete tower". hCaptcha serves this under area_select, but
        the answer is a real press/move/release drag — a point click
        never picks up the piece. NEVER uses DragLocator (punched-slot
        geometry is the wrong puzzle). Offline wood-mask heuristic
        first; a SHORT vision ``shape="tower"`` call is last resort
        (a long 504 expires the challenge).
        """
        shot, box = await self._tower_frame_shot(frame)
        if not shot or not box:
            shot, box = await self._challenge_surface(frame)
        if not shot or not box:
            self._log("[Captcha] Tower: no screenshot", level="warn")
            return False
        hint = await self._tower_piece_hint(frame, box)
        debug = {}
        got = None
        try:
            got = hct.locate_tower_drag(shot, piece_hint=hint, debug=debug)
        except Exception as e:
            debug["reason"] = "error:%s" % type(e).__name__
            self._log(f"[Captcha] Offline tower error: {e}", level="debug")
            got = None
        self._log(f"[Captcha] Tower heuristic: {debug}")
        if got and got.get("from") and got.get("to"):
            fx, fy = self._denorm(got["from"], box)
            tx, ty = self._denorm(got["to"], box)
            self._log(
                f"[Captcha] Offline tower drag "
                f"({got['from'][0]:.2f},{got['from'][1]:.2f}) -> "
                f"({got['to'][0]:.2f},{got['to'][1]:.2f})")
            await hm.drag(self._page, (fx, fy), (tx, ty))
            return True
        # Do not sit on a 180s 504 — that expires the challenge.
        answer = await self._vision.solve(
            prompt, [shot], shape="tower", timeout=18.0)
        if answer and answer.get("type") == "drag":
            fx, fy = self._denorm(answer["from"], box)
            tx, ty = self._denorm(answer["to"], box)
            self._log(f"[Captcha] Vision tower drag ({fx:.0f},{fy:.0f}) -> "
                      f"({tx:.0f},{ty:.0f})")
            await hm.drag(self._page, (fx, fy), (tx, ty))
            return True
        # Last resort: DOM piece + heuristic drop (even if `from` was missing).
        best = debug.get("best") if isinstance(debug, dict) else None
        if hint and isinstance(best, dict) and best.get("to"):
            fx, fy = self._denorm(hint, box)
            tx, ty = self._denorm(best["to"], box)
            self._log(
                f"[Captcha] Tower last-resort drag "
                f"({hint[0]:.2f},{hint[1]:.2f}) -> "
                f"({best['to'][0]:.2f},{best['to'][1]:.2f})")
            await hm.drag(self._page, (fx, fy), (tx, ty))
            return True
        return False

    async def _solve_pattern_round(self, frame, prompt) -> bool:
        """Pattern completion ("put one of the animals into the empty spot
        to complete the pattern"): a 3x3 icon grid with one empty cell and
        a row of candidates. The CORRECT candidate is chosen by the row/
        column pattern, so the geometric DragLocator cannot answer it.

        Offline path: crop the grid cells and candidates out of the
        surface screenshot, label them with the tile classifier, pick the
        candidate via hct.resolve_pattern (Latin square) — all gated on
        classifier confidence. Otherwise the vision model answers with a
        candidate->hole drag. The gesture itself is a real humanized drag
        (and if a drag is not accepted, the candidate is clicked instead —
        some builds accept click-to-place)."""
        shot, box = await self._challenge_surface(frame)
        if not shot or not box:
            return False
        from_to = None
        # ── offline: crop-classify -> Latin-square pattern logic ─────────
        tc = self._tile_classifier()
        probe = await self._probe_pattern_dom(frame, box)
        if tc is not None and probe is not None:
            try:
                import io as _io
                from PIL import Image as _Image
                import numpy as _np
                im = _Image.open(_io.BytesIO(shot)).convert("RGB")
                W, H = im.size
                cells, cands = probe
                # hole = the near-white empty cell (max mean brightness;
                # min-std fails because flat painted tiles can be flatter
                # than the outlined white hole)
                crops, means = [], []
                for rect in cells:
                    x0 = int(rect[0] * W)
                    y0 = int(rect[1] * H)
                    x1 = int((rect[0] + rect[2]) * W)
                    y1 = int((rect[1] + rect[3]) * H)
                    c = im.crop((x0, y0, x1, y1))
                    crops.append(c)
                    means.append(float(_np.asarray(c.convert("L")).mean()))
                hole = int(_np.argmax(means))
                labelled = tc.classify_many(crops)
                if len(labelled) == len(cells):
                    grid = [g[0] if i != hole else None
                            for i, g in enumerate(labelled)]
                    cand_crops = []
                    for rect in cands:
                        x0 = int(rect[0] * W)
                        y0 = int(rect[1] * H)
                        x1 = int((rect[0] + rect[2]) * W)
                        y1 = int((rect[1] + rect[3]) * H)
                        cand_crops.append(im.crop((x0, y0, x1, y1)))
                    clab = tc.classify_many(cand_crops)
                    confs = [g[1] for g in labelled] + [g[1] for g in clab]
                    mean_conf = sum(confs) / max(1, len(confs))
                    if mean_conf >= _CNN_MIN_CONF:
                        win = hct.resolve_pattern(
                            grid, hole, [g[0] for g in clab])
                        if win is not None:
                            cbox = cands[win]
                            hbox = cells[hole]
                            from_to = (
                                (box["x"] + (cbox[0] + cbox[2] / 2) * box[
                                    "width"],
                                 box["y"] + (cbox[1] + cbox[3] / 2) * box[
                                     "height"]),
                                (box["x"] + (hbox[0] + hbox[2] / 2) * box[
                                    "width"],
                                 box["y"] + (hbox[1] + hbox[3] / 2) * box[
                                     "height"]))
                            self._log(
                                f"[Captcha] Offline pattern: grid={grid} "
                                f"hole={hole} candidates={clab} -> "
                                f"{clab[win]} (conf {mean_conf:.2f})")
            except Exception as e:
                self._log(f"[Captcha] Offline pattern error: {e}",
                          level="debug")
        # ── vision fallback ───────────────────────────────────────────────
        if from_to is None:
            answer = await self._vision.solve(prompt, [shot],
                                              shape="pattern")
            if not answer or answer.get("type") != "drag":
                return False
            fx, fy = self._denorm(answer["from"], box)
            tx, ty = self._denorm(answer["to"], box)
            from_to = ((fx, fy), (tx, ty))
            self._log(f"[Captcha] Vision pattern drag ({fx:.0f},{fy:.0f}) "
                      f"-> ({tx:.0f},{ty:.0f})")
        (fx, fy), (tx, ty) = from_to
        await hm.drag(self._page, (fx, fy), (tx, ty))
        # some builds accept click-to-place instead of a drag — retry the
        # candidate with a humanized click when the round does not advance
        return True

    async def _probe_pattern_dom(self, frame, surface_box):
        """Visible small square-ish elements in the challenge frame, split
        into (grid_cells, candidates) by lattice clustering. Returns None
        when the layout is not a confident 3x3 + candidates pattern (the
        caller then uses the vision model on the full screenshot).

        Rectangles are normalised to the challenge SURFACE (matching the
        screenshot crop), so the caller can crop them out directly."""
        try:
            ox, oy = await self._frame_origin(frame)
            info = await frame.evaluate("""() => {
                const vis = (el) => !!(el) &&
                    (el.offsetParent !== null ||
                     el.getClientRects().length > 0);
                const out = [];
                const seen = new Set();
                for (const el of document.querySelectorAll(
                        'img, [class*="task" i], [class*="tile" i], ' +
                        '[class*="cell" i], [class*="item" i], ' +
                        '[class*="option" i], [class*="answer" i], ' +
                        '[draggable="true"]')) {
                    if (!vis(el) || seen.has(el)) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width < 12 || r.height < 12 ||
                        r.width > 220 || r.height > 220) continue;
                    // keep only roughly square tiles
                    const ar = r.width / Math.max(1, r.height);
                    if (ar < 0.55 || ar > 1.8) continue;
                    seen.add(el);
                    out.push({ x: r.left, y: r.top,
                               w: r.width, h: r.height });
                }
                return out;
            }""")
            if not info or len(info) < 9:
                return None
            sxo = surface_box["x"] - ox
            syo = surface_box["y"] - oy
            items = []
            for r in info:
                items.append((r["x"] - sxo, r["y"] - syo,
                              r["w"], r["h"]))
            sw = surface_box["width"]
            sh = surface_box["height"]
            cells, cands = [], []
            # grid cells: top-left corners near a regular lattice
            xs = sorted({round(it[0] / sw, 2) for it in items})
            ys = sorted({round(it[1] / sh, 2) for it in items})
            if len(xs) >= 3 and len(ys) >= 3:
                step_x = xs[1] - xs[0]
                step_y = ys[1] - ys[0]
                for it in items:
                    xi = round(it[0] / sw, 2)
                    yi = round(it[1] / sh, 2)
                    if (abs((xi - xs[0]) / max(step_x, 0.001) -
                            round((xi - xs[0]) / max(step_x, 0.001)))
                            < 0.25 and
                            abs((yi - ys[0]) / max(step_y, 0.001) -
                                round((yi - ys[0]) / max(step_y, 0.001)))
                            < 0.25):
                        cells.append(it)
                    else:
                        cands.append(it)
            if len(cells) < 8 or not cands:
                return None
            cells.sort(key=lambda r: (r[1], r[0]))
            cands.sort(key=lambda r: (r[1], r[0]))
            return ([{"x": r[0] / sw, "y": r[1] / sh,
                      "w": r[2] / sw, "h": r[3] / sh} for r in cells],
                    [{"x": r[0] / sw, "y": r[1] / sh,
                      "w": r[2] / sw, "h": r[3] / sh} for r in cands])
        except Exception:
            return None

    async def _solve_choice_round(self, frame, prompt) -> bool:
        """multiple_choice: read the option buttons, ask the model, click
        the chosen option with a humanized box click."""
        options = await frame.evaluate("""() => {
            const out = [];
            for (const el of document.querySelectorAll(
                    '.answer-option, [class*="answer-option" i], ' +
                    '[class*="choice" i] button, .options [role="button"]')) {
                const t = (el.textContent || '').replace(/\\s+/g, ' ').trim();
                if (t.length < 2 || t.length > 200) continue;
                const r = el.getBoundingClientRect();
                if (r.width < 40 || r.height < 14) continue;
                out.push({ text: t.slice(0, 200), box: {
                    x: r.left, y: r.top, width: r.width, height: r.height } });
            }
            return out;
        }""")
        if not options or len(options) < 2:
            return False
        shot, _ = await self._challenge_surface(frame)
        listing = "\n".join("%d. %s" % (i + 1, o["text"])
                            for i, o in enumerate(options))
        answer = await self._vision.solve(
            prompt + "\nOPTIONS:\n" + listing, [shot] if shot else [b""],
            shape="choice")
        if not answer or answer.get("type") != "choice":
            return False
        idx = answer.get("index")
        if not isinstance(idx, int) or not (1 <= idx <= len(options)):
            return False
        ox, oy = await self._frame_origin(frame)
        b = options[idx - 1]["box"]
        page_box = {"x": b["x"] + ox, "y": b["y"] + oy,
                    "width": b["width"], "height": b["height"]}
        self._log(f"[Captcha] Multiple-choice -> option {idx}: "
                  f"{options[idx - 1]['text'][:60]!r}")
        await hm.click_box(self._page, page_box)
        return True

    async def _solve_count_round(self, frame, prompt) -> bool:
        """counting ("How many X are in this image?"): numeric answer
        options. Tries the offline peak counter first (self-gated — it
        returns None unless the count is clean), then the vision model,
        then clicks the option button matching the count."""
        shot, box = await self._challenge_surface(frame)
        if not shot:
            return False
        n = None
        target = hct.extract_target(prompt)
        if target:
            pl = self._point_locator()
            if pl is not None:
                try:
                    n = pl.count(shot, target)
                    if n is not None:
                        self._log(
                            f"[Captcha] Offline count: {n} x {target}")
                except Exception as e:
                    self._log(f"[Captcha] Offline count error: {e}",
                              level="debug")
        if n is None:
            answer = await self._vision.solve(prompt, [shot], shape="count")
            if not answer or answer.get("type") != "count":
                return False
            n = answer.get("count")
        if not isinstance(n, int) or n < 1:
            self._log("[Captcha] Counting produced no usable number",
                      level="warn")
            return False
        self._log(f"[Captcha] Counting answer: {n}")
        return await self._click_number_option(frame, n)

    async def _click_number_option(self, frame, n: int) -> bool:
        """Click the numeric answer option whose label equals ``n``.

        hCaptcha counting rounds present a row of numbered buttons; the
        label may be just the digit or "N <noun>". Falls back to the n-th
        option in reading order when the labels are numeric and in order.
        """
        opts = await frame.evaluate("""() => {
            const vis = (el) => !!(el) &&
                (el.offsetParent !== null || el.getClientRects().length > 0);
            const out = [];
            for (const el of document.querySelectorAll(
                    '.answer-option, [class*="answer-option" i], ' +
                    '[class*="choice" i] button, [class*="option" i], button')) {
                if (!vis(el)) continue;
                const t = (el.textContent || '').replace(/\\s+/g, ' ').trim();
                if (!t) continue;
                const r = el.getBoundingClientRect();
                if (r.width < 20 || r.height < 12) continue;
                out.push({ text: t.slice(0, 80), box: {
                    x: r.left, y: r.top, width: r.width, height: r.height } });
            }
            return out;
        }""")
        if not opts:
            return False
        m = re.match(r"^(\d+)\b", str(n))
        pick = None
        if m:
            label = m.group(1)
            for o in opts:
                t = o["text"]
                if re.match(r"^%s\b" % re.escape(label), t):
                    pick = o
                    break
        if pick is None and 1 <= n <= len(opts):
            # numeric options render in order — the n-th option IS n
            texts = [o["text"] for o in opts]
            if all(re.match(r"^\d+\b", t) for t in texts):
                pick = opts[n - 1]
        if pick is None:
            return False
        ox, oy = await self._frame_origin(frame)
        b = pick["box"]
        page_box = {"x": b["x"] + ox, "y": b["y"] + oy,
                    "width": b["width"], "height": b["height"]}
        self._log(f"[Captcha] Clicking count option {n!r} ({pick['text']!r})")
        await hm.click_box(self._page, page_box)
        return True

    async def _solve_text_round(self, frame, prompt) -> bool:
        """text_entry: vision model reads the characters, we type them."""
        shot, _ = await self._challenge_surface(frame)
        if not shot:
            return False
        answer = await self._vision.solve(prompt, [shot], shape="text")
        if answer and answer.get("type") == "text":
            self._log(f"[Captcha] Text challenge: {answer.get('text')!r}")
            return await self._type_challenge_answer(frame,
                                                     answer.get("text", ""))
        return False

    async def _click_challenge_tiles(self, frame, indices) -> bool:
        """Humanized clicks on the given 1-based tile indices (reading order).

        hCaptcha grades pointer telemetry, so a bare element.click() in a
        tight loop is an automation tell. Each tile gets a Bezier glide to a
        gaussian landing point inside its box (frame origin applied) with a
        real down/dwell/up, and a human 0.18-0.55 s pause between tiles.
        element.click() remains only as a last-resort fallback."""
        for sel in ('div.task-image, [class*="task-image" i]',
                    '.task-grid img, [class*="task-grid" i] img, '
                    '.challenge-content img'):
            try:
                n = await frame.locator(sel).count()
            except Exception:
                continue
            if n == 0:
                continue
            # tile rects are frame-local (getBoundingClientRect); add the
            # iframe's own page offset once
            try:
                rects = await frame.evaluate("""(sel) => {
                    const out = [];
                    for (const el of document.querySelectorAll(sel)) {
                        const r = el.getBoundingClientRect();
                        out.push({ x: r.left, y: r.top,
                                   width: r.width, height: r.height });
                    }
                    return out;
                }""", sel)
            except Exception:
                rects = None
            ox, oy = await self._frame_origin(frame)
            clicked = 0
            for idx in indices:
                if not (isinstance(idx, int) and 1 <= idx <= n):
                    continue
                done = False
                if rects and len(rects) >= idx:
                    r = rects[idx - 1]
                    if r and r.get("width"):
                        try:
                            await hm.click_box(self._page, {
                                "x": r["x"] + ox, "y": r["y"] + oy,
                                "width": r["width"], "height": r["height"]})
                            done = True
                        except Exception:
                            done = False
                if not done:
                    try:
                        await frame.locator(sel).nth(idx - 1).click(timeout=5000)
                        done = True
                    except Exception:
                        continue
                clicked += 1
                await asyncio.sleep(random.uniform(0.18, 0.55))
            if clicked:
                return True
        return False

    _NEXT_VERIFY_JS = r"""() => {
        const norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
        const vis = (el) => {
            if (!el) return false;
            const r = el.getBoundingClientRect();
            if (!r || r.width < 8 || r.height < 8) return false;
            if (el.offsetParent === null && el.getClientRects().length === 0) return false;
            try {
                const cs = getComputedStyle(el);
                if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
            } catch (e) {}
            return true;
        };
        const disabled = (el) => !!(el.disabled
            || el.getAttribute('aria-disabled') === 'true'
            || /\bdisabled\b/i.test((el.className || '').toString()));
        const nextRe = /(^|[^a-z])(next|continue|weiter|volgende|continuer|continuar)([^a-z]|$)/;
        const verifyRe = /(^|[^a-z])(verify|check|submit|confirm|valider|verificar|bestätigen|bevestigen)([^a-z]|$)/;
        const cands = [];
        const sels = '.button-submit, #button-submit, .button-arrow, '
            + '[class*="button-arrow" i], [class*="next" i], '
            + 'button[type="submit"], [role="button"], button, .button, '
            + '[class*="submit" i], [aria-label*="next" i], '
            + '[aria-label*="verify" i], [title*="Next" i], [title*="Verify" i]';
        for (const el of document.querySelectorAll(sels)) {
            if (!vis(el) || disabled(el)) continue;
            const t = norm((el.textContent || '') + ' '
                + (el.getAttribute('aria-label') || '') + ' '
                + (el.getAttribute('title') || ''));
            const cls = norm((el.className || '').toString());
            let kind = '';
            if (nextRe.test(t) || cls.includes('arrow') || /\bnext\b/.test(cls)) kind = 'next';
            else if (verifyRe.test(t) || cls.includes('verify')) kind = 'verify';
            else if (cls.includes('button-submit') || cls.includes('submit')) kind = 'submit';
            else continue;
            const r = el.getBoundingClientRect();
            cands.push({
                kind, x: r.left, y: r.top, width: r.width, height: r.height,
                t: t.slice(0, 40)
            });
        }
        if (!cands.length) return null;
        const pick = cands.find(c => c.kind === 'next')
            || cands.find(c => c.kind === 'submit')
            || cands[0];
        return pick;
    }"""

    async def _click_challenge_verify(self, frame) -> bool:
        """Click hCaptcha's Next or Verify control inside the challenge frame.

        Attribute/material grids and mixed rounds advance with a Next
        button (often `.button-submit` whose label is "Next" or an arrow,
        enabled only after tiles are selected). A disabled click is a
        miss — wait for enablement, then humanize the click so the next
        challenge can paint.
        """
        pick = None
        for _wait in range(6):
            try:
                pick = await frame.evaluate(self._NEXT_VERIFY_JS)
            except Exception:
                pick = None
            if pick and pick.get("width"):
                break
            await asyncio.sleep(0.28)
        if pick and pick.get("width"):
            try:
                ox, oy = await self._frame_origin(frame)
                await hm.click_box(self._page, {
                    "x": float(pick["x"]) + ox,
                    "y": float(pick["y"]) + oy,
                    "width": float(pick["width"]),
                    "height": float(pick["height"]),
                })
                kind = pick.get("kind") or "submit"
                label = "Next" if kind == "next" else "Verify"
                extra = f" ({pick.get('t')!r})" if pick.get("t") else ""
                self._log(f"[Captcha] Clicked {label}{extra}")
                return True
            except Exception as e:
                self._log(f"[Captcha] Next/Verify humanized click failed: {e}",
                          level="debug")
        for sel, label in (
                ('.button-submit, #button-submit', "Verify"),
                ('.button-arrow, [class*="button-arrow" i], '
                 '[class*="next" i], [aria-label*="next" i]', "Next"),
        ):
            try:
                loc = frame.locator(sel)
                if await loc.count() > 0:
                    await loc.first.click(timeout=3000)
                    self._log(f"[Captcha] Clicked {label}")
                    return True
            except Exception:
                continue
        return False

    async def _type_challenge_answer(self, frame, text: str) -> bool:
        """Fill the answer input for 'type the characters' challenges."""
        try:
            inp = frame.locator('input[type="text"], input:not([type]), textarea').first
            await inp.click(timeout=4000)
            await inp.fill(text, timeout=4000)
            self._log("[Captcha] Typed challenge answer")
            return True
        except Exception:
            return False

    async def _solve_funcaptcha(self) -> bool:
        """FunCAPTCHA (Arkose) solver using vision AI (DragSolver).

        Uses the vision model (configured via VISION_API_BASE / OLLAMA_BASE)
        to handle slider puzzles, tile-click challenges, and drag-to-match
        challenges served by Arkose Labs FunCAPTCHA.
        """
        try:
            solver = DragSolver(
                page=self._page,
                vision=self._vision,
                log=self._log,
            )
            if await solver.detect(timeout=8.0):
                self._log("[FunCAPTCHA] DragSolver engaged", level="info")
                return await solver.solve(timeout=45.0)
            self._log("[FunCAPTCHA] No FunCAPTCHA frame detected by DragSolver",
                      level="warn")
            return False
        except Exception as e:
            self._log(f"[FunCAPTCHA] DragSolver error: {e}", level="error")
            return False

    async def _form_ready(self) -> dict:
        """Evaluate _FORM_READY_JS with the locale-aware DOB label table."""
        try:
            v = await self._page.evaluate(
                _FORM_READY_JS.replace("__DOB_LABELS__", json.dumps(_DOB_LABEL_ALIASES)))
            return json.loads(v) if v else {}
        except Exception:
            return {}

    async def _select_dob(self, label: str, option_text: str) -> bool:
        """Select one DOB dropdown (Month/Day/Year) with REAL trusted clicks.

        Discord's register form localizes the DOB controls (Dutch
        "Dag/Maand/Jaar", French "Jour/Mois/Année", ...) and its newer builds
        ignore JS-dispatched synthetic mouse events — the old all-JS click
        strategies opened no menu at all, so the form was submitted with the
        placeholders still showing and Discord rejected it. This locates the
        control by its localized label (with a text-walker fallback for
        controls that carry no role/class markers), opens it with a trusted
        Playwright click, matches the option locale-aware (months resolve to
        their numeric index, so "January" picks the Dutch "Januari"), and
        selects it with a trusted click (coordinates click, then an index
        fallback for options with no usual markers). Falls back to the JS
        setter for native <select> / legacy builds.
        """
        try:
            self._log(f"Selecting {label}: {option_text}")

            # Gate on the CONTROL ITSELF (not _form_ready's DOB scan, which
            # misses role="button" controls and could abort before anything
            # was attempted).
            located = None
            for _probe in range(12):
                try:
                    located = await self._page.evaluate(
                        _DOB_LOCATE_JS, [label, _DOB_LABEL_ALIASES, None, None])
                except Exception:
                    located = None
                if located:
                    break
                await asyncio.sleep(0.35)
            if not located:
                self._log(f"[DOB] control for {label} not found after ~4s — JS fallback", level="warn")
                return await self._dob_js_fallback(label, option_text)

            is_select = located.get("tag") == "select"

            # ── Native <select>: select_option by matched index ──
            if is_select:
                try:
                    ctrl = self._page.locator(f'[data-dob-target="{label}"]')
                    # Exact index into the <select>'s OWN option list.
                    # (The generic _DOB_OPTION_INDEX_JS only counts
                    # offsetParent!==null elements — in a CLOSED native
                    # select no <option> is rendered, so it always
                    # returns -1 and the select gets silently skipped.)
                    idx = await self._page.evaluate(
                        """([sel, value, monthAliases]) => {
                            const el = document.querySelector(sel);
                            if (!el || el.tagName !== 'SELECT') return -1;
                            const norm = s => (s == null ? '' : String(s)).toLowerCase().replace(/\\s+/g, ' ').trim();
                            const want = norm(value);
                            const wantNum = (monthAliases && monthAliases[want]) || 0;
                            for (let i = 0; i < el.options.length; i++) {
                                const o = el.options[i];
                                const t = norm(o.text), v = norm(o.value);
                                if (t === want || v === want) return i;
                                if (wantNum &&
                                    ((monthAliases && monthAliases[t]) === wantNum ||
                                     (monthAliases && monthAliases[v]) === wantNum)) return i;
                            }
                            return -1;
                        }""",
                        [f'[data-dob-target="{label}"]', option_text, _MONTH_ALIASES])
                    if isinstance(idx, int) and idx >= 0:
                        await ctrl.select_option(index=idx)
                        await asyncio.sleep(0.3)
                        self._log(f"Selected {label} (native select index {idx})")
                        return True
                    self._log(f"[DOB] native select for {label}: no option matched '{option_text}' (idx={idx})", level="warn")
                except Exception as e:
                    self._log(f"[DOB] native select failed for {label}: {e}", level="warn")
                return await self._dob_js_fallback(label, option_text)

            # ── Custom dropdown: open the menu, then pick the option ──
            # Discord re-renders the form while credentials are being written
            # (React controlled inputs) and a humanized cursor moves
            # slowly, so a single click can hang on 'performing click
            # action' for the full 30s default even though the element
            # resolved visible+stable — the exact stall from the field logs.
            # Rule: SHORT click timeouts + verify the menu actually opened +
            # layered fallbacks (coordinate click -> trusted locator click
            # -> JS dispatch -> keyboard) so no single step can ever eat 30s.
            #
            # Center the control in the viewport BEFORE opening: the Year
            # menu is the longest (~110 options) and the DOB row sits near
            # the bottom of the 1280x720 viewport — opened from the row's
            # natural position the menu runs PAST the viewport bottom, the
            # page can't scroll it back in, and the option click lands in
            # empty space. That is why Month/Day filled and Year didn't.
            # Centering leaves room for the menu to drop below the control.
            try:
                await self._page.evaluate(
                    "() => { const el = document.querySelector('[data-dob-target=\""
                    + label + "\"]'); if (el) { try { el.scrollIntoView({ block: 'center' }); } catch (e) {} } }")
                await asyncio.sleep(0.25)
            except Exception:
                pass
            deadline = time.monotonic() + 25.0
            opened = False
            for open_method in ("coords", "click", "dispatch", "keyboard"):
                if opened or time.monotonic() > deadline:
                    break
                # Re-locate every attempt: React may have replaced the
                # control after the previous attempt.
                try:
                    located = await self._page.evaluate(
                        _DOB_LOCATE_JS, [label, _DOB_LABEL_ALIASES, None, None])
                except Exception:
                    located = None
                if not located:
                    self._log(f"[DOB] control for {label} vanished — JS fallback", level="warn")
                    return await self._dob_js_fallback(label, option_text)
                ctrl = self._page.locator(f'[data-dob-target="{label}"]')
                # Close any stray menu so it can't swallow the events below.
                try:
                    await self._page.keyboard.press("Escape")
                except Exception:
                    pass
                await asyncio.sleep(0.15)
                if open_method == "click":
                    try:
                        await ctrl.scroll_into_view_if_needed(timeout=3000)
                        await ctrl.click(timeout=4500)
                    except Exception as e:
                        self._log(f"[DOB] open click {label}: {str(e)[:150]}", level="warn")
                elif open_method == "coords":
                    # Trusted input at the control's center — engine-
                    # engine-driven (real CDP input), with no
                    # actionability re-checks to stall on.
                    try:
                        await ctrl.scroll_into_view_if_needed(timeout=3000)
                        box = await ctrl.bounding_box()
                        if not box or not box.get("width"):
                            continue
                        await self._page.mouse.click(
                            box["x"] + box["width"] / 2,
                            box["y"] + box["height"] / 2)
                    except Exception as e:
                        self._log(f"[DOB] open coords {label}: {str(e)[:120]}", level="warn")
                        continue
                elif open_method == "dispatch":
                    try:
                        await ctrl.dispatch_event("pointerdown")
                        await ctrl.dispatch_event("pointerup")
                        await ctrl.dispatch_event("mousedown")
                        await ctrl.dispatch_event("mouseup")
                        await ctrl.dispatch_event("click")
                    except Exception as e:
                        self._log(f"[DOB] open dispatch {label}: {str(e)[:120]}", level="warn")
                        continue
                elif open_method == "keyboard":
                    # Discord's combobox opens on ArrowDown when focused —
                    # works even when mouse hit-testing is broken.
                    try:
                        await ctrl.focus()
                        await self._page.keyboard.press("ArrowDown")
                    except Exception as e:
                        self._log(f"[DOB] open keyboard {label}: {str(e)[:120]}", level="warn")
                        continue
                # Did THIS control's menu actually open? Poll with the
                # scoped check (adjacent to the marked control, >=2 visible
                # parts) — a timed-out click may still have opened it, but a
                # stale Month/Day menu or page chrome must NOT count.
                for _poll in range(8):
                    await asyncio.sleep(0.25)
                    try:
                        opened = bool(await self._page.evaluate(
                            _DOB_MENU_OPEN_JS, label))
                    except Exception:
                        opened = False
                    if opened:
                        break
            if opened:
                self._log(f"[DOB] {label}: menu opened via {open_method}")
            if not opened:
                self._log(f"[DOB] menu for {label} never opened — JS fallback", level="warn")
                return await self._dob_js_fallback(label, option_text)

            # ── Pick the option ──
            # Order: TYPEAHEAD first — it is geometry-independent and the
            # only reliable path for a DEEP option in a long menu (Year's
            # ~110 options vs Day's option sitting 4th). A human does
            # exactly this: focus the control, type "1995", hit Enter.
            # Works for native <select> AND custom React-Select; Enter
            # commits through the component's own keyboard flow (no
            # stale-DOM selection that a re-render would revert).
            picked = False
            for sel_method in ("typeahead", "coords", "index", "keyboard", "dispatch"):
                if picked or time.monotonic() > deadline:
                    break
                idx = -1
                pos = None
                if sel_method in ("index", "keyboard"):
                    try:
                        idx = await self._page.evaluate(
                            _DOB_OPTION_INDEX_JS.replace("__OPT_SEL__", json.dumps(_DOB_OPTION_SEL)),
                            [option_text, _MONTH_ALIASES])
                    except Exception:
                        idx = -1
                if sel_method in ("coords", "dispatch"):
                    try:
                        pos = await self._page.evaluate(
                            _DOB_OPTION_POS_JS, [option_text, _MONTH_ALIASES])
                    except Exception:
                        pos = None
                if sel_method == "typeahead":
                    try:
                        # The control that opened the menu holds focus.
                        # Type the option text with quick human cadence —
                        # typeahead jumps to the first option whose label
                        # starts with the accumulated string ("1"->1999,
                        # "19"->1999, "1995"->1995) — then Enter confirms.
                        for ch in str(option_text):
                            await self._page.keyboard.type(ch)
                            await asyncio.sleep(random.uniform(0.08, 0.18))
                        await asyncio.sleep(0.25)
                        await self._page.keyboard.press("Enter")
                        picked = True
                        self._log(f"[DOB] option for {label} by typeahead ({option_text})")
                    except Exception as e:
                        self._log(f"[DOB] option typeahead {label}: {str(e)[:120]}", level="warn")
                elif sel_method == "index":
                    if isinstance(idx, int) and idx >= 0:
                        try:
                            # index is over VISIBLE options only (see
                            # _DOB_OPTION_INDEX_JS), so filter hidden
                            # matches before nth() or we'd click the
                            # wrong element.
                            await self._page.locator(_DOB_OPTION_SEL).filter(visible=True).nth(idx).click(timeout=4500)
                            picked = True
                        except Exception as e:
                            self._log(f"[DOB] option index click {label}: {str(e)[:140]}", level="warn")
                elif sel_method == "coords":
                    if pos and pos.get("x"):
                        try:
                            await self._page.mouse.click(float(pos["x"]), float(pos["y"]))
                            picked = True
                            self._log(f"[DOB] option for {label} by coords ({pos.get('text')})")
                        except Exception as e:
                            self._log(f"[DOB] option coords click {label}: {str(e)[:120]}", level="warn")
                elif sel_method == "dispatch":
                    try:
                        r = await self._page.evaluate(_DOB_OPTION_DISPATCH_JS, [option_text, _MONTH_ALIASES])
                        if r:
                            picked = True
                            self._log(f"[DOB] option for {label} via JS dispatch ({r})")
                    except Exception as e:
                        self._log(f"[DOB] option dispatch {label}: {str(e)[:120]}", level="warn")
                elif sel_method == "keyboard":
                    if isinstance(idx, int) and idx >= 0 and idx <= 300:
                        try:
                            # Guarantee the menu is actually open: if the
                            # open-check was a false positive, Home on a
                            # closed combobox does nothing and the first
                            # ArrowDown (which opens the menu) desyncs the
                            # highlight. One ArrowDown opens it when closed.
                            menu_up = False
                            try:
                                menu_up = bool(await self._page.evaluate(
                                    _DOB_MENU_OPEN_JS, label))
                            except Exception:
                                menu_up = False
                            await ctrl.focus()
                            if not menu_up:
                                await self._page.keyboard.press("ArrowDown")
                                await asyncio.sleep(0.2)
                            # Navigate from the NEAR end: Home + idx downs
                            # for the top half, End + (count-1-idx) ups for
                            # the bottom half — the Year menu has ~110
                            # options, and each press is a CDP round-trip.
                            count = idx + 1
                            try:
                                count = int(await self._page.evaluate(
                                    "(sel) => Array.from(document"
                                    ".querySelectorAll(sel)).filter("
                                    "e => e.offsetParent !== null).length",
                                    _DOB_OPTION_SEL)) or (idx + 1)
                            except Exception:
                                count = idx + 1
                            if idx * 2 <= count:
                                await self._page.keyboard.press("Home")
                                for _k in range(max(idx, 0)):
                                    await self._page.keyboard.press("ArrowDown")
                            else:
                                await self._page.keyboard.press("End")
                                for _k in range(max(count - 1 - idx, 0)):
                                    await self._page.keyboard.press("ArrowUp")
                            await asyncio.sleep(0.15)
                            await self._page.keyboard.press("Enter")
                            picked = True
                        except Exception as e:
                            self._log(f"[DOB] option keyboard {label}: {str(e)[:120]}", level="warn")
                if not picked:
                    continue
                await asyncio.sleep(0.4)
                if await self._dob_verify(label, option_text):
                    self._log(f"Selected {label}: {option_text} (trusted click)")
                    return True
                # Selection didn't stick — close the menu and try the next
                # method.
                picked = False
                try:
                    await self._page.keyboard.press("Escape")
                except Exception:
                    pass
                await asyncio.sleep(0.3)

            self._log(f"[DOB] selection methods failed for {label} — JS fallback", level="warn")
            try:
                cur = await self._dob_current_value(label)
                self._log(f"[DOB] {label} control still shows '{cur}' after all pick methods", level="warn")
            except Exception:
                pass
            return await self._dob_js_fallback(label, option_text)

        except Exception as e:
            self._log_exception(f"DOB error for {label}", e)
            return False

    async def _dob_verify(self, label: str, option_text: str) -> bool:
        """Confirm the DOB control now shows the selected value.

        Tries the marked element first (React usually re-renders it in
        place), then re-locates by label. The value text itself is accepted
        too — after selection the control shows "Januari", not the "Maand"
        placeholder, so a label-only re-locate would miss it."""
        try:
            txt = await self._page.locator(
                f'[data-dob-target="{label}"]').first.inner_text()
        except Exception:
            txt = ""
        if _dob_text_matches(txt, option_text):
            return True
        # The data-dob-target marker can land on a stale/container element
        # after a React re-render (misreads like '1982\nYear,\n1982'); the
        # combobox aria-label read is the reliable ground truth.
        try:
            cur = await self._dob_current_value(label)
            if _dob_text_matches(cur, option_text):
                return True
        except Exception:
            pass
        try:
            located = await self._page.evaluate(
                _DOB_LOCATE_JS, [label, _DOB_LABEL_ALIASES, option_text, _MONTH_ALIASES])
            if located:
                txt2 = await self._page.locator(
                    f'[data-dob-target="{label}"]').first.inner_text()
                if _dob_text_matches(txt2, option_text):
                    return True
        except Exception:
            pass
        try:
            self._log(f"[DOB] verify {label}: control shows '{txt[:60]}' expected '{option_text}'", level="warn")
        except Exception:
            pass
        return False

    async def _dob_current_value(self, label: str) -> str:
        """What a DOB control currently displays ('' = placeholder/not found).

        Used by the post-fill verify so a swallowed selection is caught and
        re-selected instead of submitting with placeholders still showing.
        After a React re-render the data-dob-target marker can land on the
        field's <label> (its text like 'Month*' matches the label regex), so
        fall back to the combobox that carries the localized aria-label and
        read the select field's visible text.
        """
        # 1) the freshly marked control - only accept a short, value-like
        #    read (a bare label like 'Month*' is rejected).
        try:
            txt = await self._page.locator(
                f'[data-dob-target="{label}"]').first.inner_text()
            txt = (txt or "").strip()
            if txt and len(txt) <= 40 and "*" not in txt:
                return txt
        except Exception:
            pass
        # 2) combobox with the localized aria-label -> select field text.
        try:
            v = await self._page.evaluate(_DOB_VALUE_JS, [label, _DOB_LABEL_ALIASES])
            return (v or "").strip()
        except Exception:
            pass
        return ""

    async def _dob_js_fallback(self, label: str, option_text: str) -> bool:
        """Last-resort JS setter for native <select> / legacy builds."""
        try:
            result2 = await self._page.evaluate(_DOB_FALLBACK_JS
                .replace("__LABEL__", json.dumps(label))
                .replace("__OPT__", json.dumps(option_text))
                .replace("__DOB_LABELS__", json.dumps(_DOB_LABEL_ALIASES)))
            if result2 and str(result2).startswith(("native:", "combo:")):
                self._log(f"Selected {label} ({result2})")
                await asyncio.sleep(0.3)
                return True
            self._log(f"DOB fallback for {label}: {result2}")
        except Exception as e:
            self._log(f"DOB fallback error for {label}: {e}", level="warn")
        self._log(f"All DOB strategies failed for {label}", level="warn")
        return False

    async def _rate_limited(self) -> bool:
        """True when Discord shows its rate-limit message ("The resource is
        being rate limited.") on the current page. Cheap full-page text
        check so the worker rotates the proxy the instant it renders."""
        try:
            text = await self._page.evaluate(
                "() => (document.body ? document.body.innerText : '')")
        except Exception:
            return False
        low = (text or "").lower()
        return any(k in low for k in _RATE_LIMIT_KEYWORDS)

    async def _wait_for_form_ready(self, timeout: float = 30.0):
        """Wait for the register form to FULLY render before touching it.

        Returns "form" (credential inputs visible), "age_gate" (DOB controls
        up but credentials not shown yet - Discord asks for birthday first on
        some builds), or None (stopped / rate limited / timed out). The ready
        state must HOLD for ~0.5s so React's hydration and value trackers are
        attached before anything is written to a field - writing to a
        not-yet-hydrated input is exactly what made Discord wipe the value on
        its next re-render (the "nothing was filled" bug).
        """
        self._log(f"[Form] Waiting for the full form (email/username/password) to render (up to {timeout:.0f}s)...")
        start = time.time()
        last_log = -1.0
        stable_since = None
        eval_failed_logged = False
        while True:
            if self._stopped.is_set():
                self._nav_error = "stopped by user"
                self._log("[Form] Stopped by user while waiting for the form")
                return None
            if await self._rate_limited():
                self._nav_error = "rate limited (429) by Discord"
                self._log("[Form] RATE LIMITED while waiting for the form", level="warn")
                return None
            elapsed = time.time() - start
            try:
                st = await self._form_ready()
            except Exception as e:
                if not eval_failed_logged:
                    eval_failed_logged = True
                    self._log_exception("[Form] Read form-ready state failed", e)
                st = {}
            email = bool(st.get("email"))
            username = bool(st.get("username"))
            password = bool(st.get("password"))
            dob = int(st.get("dob") or 0)
            dob_text = bool(st.get("dobText"))
            full_form = email and username and password
            age_gate = (not email and not username and not password) and (dob >= 1 or dob_text)
            if full_form or age_gate:
                if stable_since is None:
                    stable_since = time.time()
                if time.time() - stable_since >= 0.5:
                    which = "age_gate" if age_gate else "form"
                    self._log(f"[Form] {'Age gate' if age_gate else 'Full form'} rendered in {elapsed:.1f}s (email={email} user={username} pass={password} dob={dob})")
                    return which
            else:
                stable_since = None
            if elapsed >= last_log + 4.0:
                last_log = elapsed
                self._log(f"[Form] Render wait {int(elapsed)}s: email={email} user={username} pass={password} dob={dob}/3 inputs={st.get('inputs')} buttons={st.get('buttons')}")
            if elapsed >= timeout:
                self._nav_error = f"register form never fully rendered after {int(elapsed)}s"
                self._log(f"[Form] Form never fully rendered after {int(elapsed)}s - rotating", level="warn")
                return None
            await asyncio.sleep(0.3)

    async def _read_form_values(self) -> dict:
        """Authoritative read-back of the register form's credential fields.

        Returns the CURRENT raw values (email/display/username/password plus a
        ToS count) so the filler can confirm a write actually landed — and
        re-check right before Create Account so a Discord re-render that
        wipes a field can never be submitted as if it were still filled."""
        if self._page is None:
            return {}
        try:
            v = await self._page.evaluate("""() => {
                const F = __CRED_FIELDS__;
                const g = (sel) => { const e = document.querySelector(sel); return e ? (e.value || '') : ''; };
                return JSON.stringify({
                    email: g(F.email),
                    display: g(F.display),
                    username: g(F.username),
                    password: g(F.password),
                    tos: document.querySelectorAll('input[type="checkbox"]:checked, [role="checkbox"][aria-checked="true"], [role="checkbox"][data-state="checked"]').length,
                });
            }""".replace('__CRED_FIELDS__', json.dumps(_CRED_FIELD_SELECTORS)))
            return json.loads(v) if v else {}
        except Exception as e:
            self._log_exception("[Form] Read-back failed", e)
            return {}

    def _credentials_filled(self, vals: dict) -> bool:
        """Email + password must hold EXACTLY the expected values; the
        username just has to be a non-empty, non-email value. Discord
        legitimately REPLACES a taken generated username with its own
        suggestion (digits appended in the field), so requiring an exact
        match made the final gate abort right before Create Account on a
        fully valid form - the "did dob but not create account" failure."""
        email_ok = vals.get("email") == self._email
        pass_ok = vals.get("password") == self._password
        user = (vals.get("username") or "").strip()
        # A username still holding the email value is a leak, not a fill.
        user_ok = bool(user) and user != self._email
        return email_ok and pass_ok and user_ok

    def _build_cred_fields(self, display_name: str) -> list:
        """(fname, name-first selector, value) for the four credential fields.

        Discord's register form uses stable `name` attributes (email /
        global_name / username / password — confirmed by the INPUT DUMP), so
        every selector leads with input[name=...] and only falls back to loose
        aria/id/placeholder matching when the name lookup finds nothing.
        `input[autocomplete='username']` is deliberately NOT in the username
        selector: an email input carrying autocomplete="username" would make
        `.first` resolve to the email box and every "username" write would
        land in the email field.
        """
        return [
            ("username", _CRED_FIELD_SELECTORS["username"], self._username or ""),
            ("display", _CRED_FIELD_SELECTORS["display"], display_name),
            ("password", _CRED_FIELD_SELECTORS["password"], self._password or ""),
            ("email", _CRED_FIELD_SELECTORS["email"], self._email or ""),
        ]

    async def _read_field_value(self, sel: str) -> str:
        """Read one field's current value; empty string on any failure."""
        if self._page is None:
            return ""
        try:
            loc = self._page.locator(sel)
            if (await loc.count()) == 0:
                return ""
            try:
                return await loc.first.input_value()
            except Exception:
                return ""
        except Exception:
            return ""

    async def _type_humanly(self, sel: str, val: str) -> bool:
        """Type one field like a real person instead of pasting it.

        Real click focuses the input (real Chrome cursor via nodriver),
        then character-by-character keyboard input with variable rhythm,
        one mid-field "thinking" pause, and an occasional typo corrected
        with backspace. Returns True only when the field holds `val` —
        anything weird falls back to the instant fill()/JS write path.
        """
        if self._page is None:
            return False
        try:
            loc = self._page.locator(sel)
            if (await loc.count()) == 0 or not (await loc.first.is_visible()):
                return False
            # A human looks at the field and reaches for it first.
            await asyncio.sleep(random.uniform(0.15, 0.5))
            await loc.first.click(timeout=8000)
            await asyncio.sleep(random.uniform(0.1, 0.3))
            # Clear any pre-existing value on the focused input.
            try:
                await self._page.keyboard.press("Control+a")
                await self._page.keyboard.press("Backspace")
            except Exception:
                pass
            await asyncio.sleep(random.uniform(0.08, 0.2))

            did_mid_pause = False
            for i, ch in enumerate(val):
                # One mid-field "thinking" pause per field (humans pause).
                if (not did_mid_pause and len(val) >= 8
                        and 0.15 < (i / len(val)) < 0.75
                        and random.random() < 0.25):
                    await asyncio.sleep(random.uniform(0.25, 0.75))
                    did_mid_pause = True
                # Occasional typo + backspace correction on longer fields.
                if (len(val) >= 6 and ch.isalnum()
                        and random.random() < 0.05):
                    wrong = random.choice(
                        "abcdefghijklmnopqrstuvwxyz0123456789")
                    await self._page.keyboard.type(wrong)
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await self._page.keyboard.press("Backspace")
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                await self._page.keyboard.type(ch)
                delay = _human_typing_delay(ch)
                if random.random() < 0.16:
                    delay += random.uniform(0.12, 0.45)
                await asyncio.sleep(delay)
            await asyncio.sleep(random.uniform(0.2, 0.6))
            try:
                return (await self._page.locator(sel).first.input_value()) == val
            except Exception:
                return False
        except Exception:
            return False

    async def _write_field_value(self, sel: str, val: str,
                                 human: bool = False) -> bool:
        """Element-targeted write of ONE field, verified. True when the field
        holds `val` afterwards.

        Writes are 0) HUMANIZED typing (real keystrokes, human cadence —
        used on the first attempt of the primary fill), then 1) Playwright
        fill() — trusted input events React accepts, then 2) the native-
        setter JS write (_REACT_SET_VALUE_JS) which REPLACES the whole value
        and never depends on focus or the global keyboard. The old click +
        Control+A + press_sequentially fallback is GONE: it typed into
        whatever field held focus (Discord's register page keeps focus on
        the first input, the email box), which is exactly how the username
        ended up concatenated inside the email field.
        """
        if not val:
            return False
        # A closed/stopped browser leaves _page None mid-fill — fail cleanly
        # instead of AttributeError-spamming and spinning on a dead tab.
        if self._page is None:
            self._log("[Form] Page is gone (browser closed/stopped) - cannot write fields", level="warn")
            return False
        # Release focus from any field first, so a stale focused element can
        # never intercept a write or receive stray input.
        try:
            await self._page.evaluate(
                "() => { const a = document.activeElement; if (a && a.blur) a.blur(); }")
        except Exception:
            pass
        # 0) Humanized typing — only when asked (primary fill attempt).
        if human:
            try:
                if await self._type_humanly(sel, val):
                    return True
            except Exception:
                pass
        # 1) Playwright fill() — replaces the whole value, React-compatible.
        try:
            loc = self._page.locator(sel)
            if (await loc.count()) == 0 or not (await loc.first.is_visible()):
                return False
            # Clear FIRST, then write, then verify. Playwright fill() already
            # replaces the value, but an explicit empty fill guards against
            # any engine build that appends to a non-empty field (that
            # concatenation was the "email shows the address twice/mangled"
            # corruption); the JS-setter fallback below is the guaranteed-
            # replace path for anything that still slips.
            await loc.first.fill("")
            await asyncio.sleep(0.05)
            await loc.first.fill(val)
            await asyncio.sleep(0.25)
            try:
                if (await loc.first.input_value()) == val:
                    return True
            except Exception:
                pass
        except Exception:
            pass
        # 2) Native-setter JS — focus-independent replace + tracker sync.
        # Uses the FULL selector list (same as the fill path above) so it
        # resolves the same element: the old split(',')[0] silently did
        # nothing when Discord's input lacked that first selector's name.
        try:
            await self._page.evaluate(
                _REACT_SET_VALUE_JS, [sel, val])
            await asyncio.sleep(0.25)
            try:
                return (await self._page.locator(sel).first.input_value()) == val
            except Exception:
                return False
        except Exception as e:
            self._log_exception("[Form] JS value write failed", e)
            return False

    async def _fill_credential_fields(self, display_name: str) -> None:
        """Fill every credential field and keep them filled (self-healing).

        Fix for the "email ends up holding the username" corruption:
          1. Fill order is username → display → password → EMAIL LAST, so any
             leak lands in the still-empty email field and the targeted email
             write overwrites it.
          2. ALL writes are element-targeted (Playwright fill() or the
             native-setter JS write). No keystroke fallback exists anymore —
             keystrokes typed into whatever field had focus (the email box),
             producing the email+username concatenation in the screenshot.
          3. After every write every credential field is re-read and any
             leaked/wiped value is healed with another targeted write.
          4. A final stability pass re-applies writes until the whole form
             holds its values for two consecutive reads (React re-renders
             during DOB/ToS can wipe a controlled input even after a clean
             fill).
        """
        if self._page is None:
            self._log("[Form] Page is gone (browser closed/stopped) - aborting form fill", level="warn")
            return
        fields = self._build_cred_fields(display_name)
        filled_so_far = []
        for fname, sel, val in fields:
            if not val:
                continue
            for attempt in range(1, 4):
                if await self._write_field_value(sel, val, human=(attempt == 1)):
                    self._log(f"[Form] Field '{fname}' verified: len={len(val)}")
                    break
                cur = await self._read_field_value(sel)
                self._log(f"[Form] Field '{fname}' mismatch (attempt {attempt}/3) got_len={len(cur)}", level="warn")
                await asyncio.sleep(0.5)
            filled_so_far.append((fname, sel, val))
            # Human pause between fields — reads like a real signup and gives
            # Discord's React time to finish re-rendering.
            await asyncio.sleep(random.uniform(0.3, 0.6))
            # Heal pass: ONLY check fields that have already been filled so far.
            if len(filled_so_far) > 1:
                await self._heal_credential_fields(filled_so_far)
        # Final stability pass — two consecutive clean reads before moving on.
        await self._stabilize_credential_fields(fields)

    async def _heal_credential_fields(self, fields) -> None:
        """Re-fill any credential field whose value got wiped or leaked."""
        for fname, sel, val in fields:
            if not val:
                continue
            cur = await self._read_field_value(sel)
            if cur == val:
                continue
            # Username is Discord-owned once filled: Discord replaces a
            # taken generated name with its own suggestion (digits
            # appended), so only heal a WIPED username (empty) or one
            # that leaked the email value - never clobber Discord's
            # valid suggestion back into the taken name.
            if fname == "username" and cur and cur != self._email:
                continue
            self._log(f"[Form] Heal '{fname}': value wiped/leaked — re-writing", level="warn")
            await self._write_field_value(sel, val)

    async def _stabilize_credential_fields(self, fields, passes: int = 3) -> None:
        """Re-apply targeted writes until the whole form holds its values for
        two consecutive reads.

        Discord's React re-renders (username availability checks, DOB
        selection, ToS clicks) can wipe a controlled input even after a clean
        fill. Because every write here is element-targeted there is no
        keystroke that could leak into another field, so repeated passes are
        safe."""
        prev_ok = False
        for _pass in range(passes):
            if self._page is None:
                self._log("[Form] Page is gone during stabilize - aborting", level="warn")
                return
            await asyncio.sleep(0.4)
            ok = True
            for fname, sel, val in fields:
                if not val:
                    continue
                cur = await self._read_field_value(sel)
                if cur == val:
                    continue
                # Same rule as the heal pass: a non-empty username that
                # is not a leaked email is Discord's own suggestion.
                if fname == "username" and cur and cur != self._email:
                    continue
                ok = False
                self._log(f"[Form] Stabilize: '{fname}' wiped — re-writing", level="warn")
                await self._write_field_value(sel, val)
            if ok and prev_ok:
                return
            prev_ok = ok

    async def _fill_registration_form(self) -> bool:
        try:
            self._log("=" * 40)
            self._log("FILLING REGISTRATION FORM (direct value set)")
            self._log("=" * 40)
            self._log(f"Email: {self._email}")
            # Humanization: a moment to "read" the form before typing.
            await asyncio.sleep(random.uniform(0.5, 1.3))

            # Rotate the proxy the moment Discord's rate-limit message shows.
            if await self._rate_limited():
                self._nav_error = "rate limited (429) by Discord"
                self._log("[Form] RATE LIMITED — rotating circuit", level="warn")
                return False

            # ── Pre-define DOB so the age-gate JS can use them ──
            month_val = random.randint(1, 12)
            day_val = str(random.randint(1, 28))
            # Always under 2003 (18+ for any 2026 signup; Discord rejects
            # underage DOBs, and the operator demands pre-2003 years).
            year_val = str(random.randint(1990, 2002))
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            month_name = months[month_val - 1]

            # ── Wait for the form (or age gate) to FULLY render first ──
            # The old 6x0.6s body-text poll returned on partial text and let
            # the filler run while React was still hydrating - fields looked
            # present but their handlers/value-trackers weren't attached yet,
            # so writes got wiped and the keystroke fallback typed into
            # whatever half-rendered element had focus. Gate on ACTUAL
            # visible inputs (+ DOB controls) before touching anything.
            phase = await self._wait_for_form_ready(timeout=30.0)
            if phase == "age_gate":
                self._log("[Form] Age gate detected — setting DOB before the main form...")
                await self._select_dob("Month", month_name)
                await asyncio.sleep(0.3)
                await self._select_dob("Day", day_val)
                await asyncio.sleep(0.3)
                await self._select_dob("Year", year_val)
                await asyncio.sleep(0.6)
                phase = await self._wait_for_form_ready(timeout=20.0)
            if phase != "form":
                self._log("[Form] Form never fully rendered — aborting fill", level="warn")
                return False

            # ── Generate credentials ──
            consonants = 'bcdfghjklmnpqrstvwxyz'
            vowels = 'aeiou'
            username = ''
            for _ in range(random.randint(8, 12)):
                username += random.choice(vowels if random.random() < 0.35 else consonants)
            username += str(random.randint(100, 9999))
            self._username = username
            display_name = self._username[:15]

            first = random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')
            body = ''
            for _ in range(random.randint(8, 11)):
                body += random.choice(vowels if random.random() < 0.35 else consonants)
            specials = '!@#$%&*'
            self._password = first + body + random.choice(specials) + str(random.randint(1, 99))

            self._log(f"Display: {display_name}  Username: {self._username}  Pass: ***")

            # ── Fill ALL fields, one at a time, self-healing ──
            # Discord's React controlled inputs wipe synthetic JS value sets
            # (the old "all fields :ok but readback empty" failure), and
            # keystrokes typed while React is re-rendering leak into whatever
            # element still has focus — the "email ends up holding the
            # username" loop. The fill order is username → display → password
            # → EMAIL LAST: any leak lands in the still-empty email field and
            # the element-targeted email write overwrites it instead of
            # looping forever. After every write every field is re-verified
            # and leaks are healed.
            self._display_name = display_name
            await self._fill_credential_fields(display_name)

            # ── Quick verify each field ──
            try:
                vals = await self._read_form_values()
                if self._credentials_filled(vals):
                    self._log("[Form] All credential fields verified OK (ToS clicked separately)")
                else:
                    safe = {
                        "email": vals.get("email", ""),
                        "display": vals.get("display", ""),
                        "username": vals.get("username", ""),
                        "password_len": len(vals.get("password") or ""),
                        "tos": vals.get("tos", 0),
                    }
                    self._log(
                        f"[Form] VERIFY MISMATCH: readback={json.dumps(safe)} "
                        f"expected={json.dumps({'email': self._email, 'username': self._username, 'password_len': len(self._password or '')})}",
                        level="warn",
                    )
                    # ── Playwright fill() + keystroke fallback ──
                    # If React wiped any field, type it with REAL keystrokes
                    # so the form is genuinely filled before we press Create
                    # Account (never fake). Dump EVERY input on the page
                    # (type/name/aria/value/visibility) so the exact failure
                    # is visible in ALL LOGS.
                    try:
                        dump = await self._page.evaluate("""() => {
                            return JSON.stringify(Array.from(document.querySelectorAll('input')).map(function(e) {
                                return {
                                    type: e.type || '',
                                    name: e.name || '',
                                    id: e.id || '',
                                    aria: e.getAttribute('aria-label') || '',
                                    placeholder: e.placeholder || '',
                                    value: e.value || '',
                                    visible: e.offsetParent !== null
                                };
                            }));
                        }""")
                        self._log(f"[Form] INPUT DUMP: {dump}", level="warn")
                    except Exception as _de:
                        self._log_exception("[Form] INPUT DUMP failed", _de)
                    await self._fill_missing_fields(display_name)
                    # Re-read AFTER the fallback. If Discord's React still
                    # won't hold the values, rotating beats faking a Create
                    # Account submit on an empty form.
                    vals = await self._read_form_values()
                    if not self._credentials_filled(vals):
                        self._log(
                            "[Form] [FAIL] Fields still empty after JS + Playwright "
                            "fallback — aborting (never submit a blank form): "
                            + json.dumps({
                                "email": vals.get("email", ""),
                                "username": vals.get("username", ""),
                                "password_len": len(vals.get("password") or ""),
                            }),
                            level="error",
                        )
                        await self.capture_screenshot()
                        return False
                    self._log("[Form] Credential fields OK after fallback")
            except Exception as e:
                self._log_exception("[Form] Verify read-back failed", e)

            # ── DOB ──
            self._log(f"DOB: {month_name} {day_val}, {year_val}")
            await self._select_dob("Month", month_name)
            await self._human_pause()
            await self._select_dob("Day", day_val)
            await self._human_pause()
            await self._select_dob("Year", year_val)
            await self._human_pause()

            # ── DOB post-verify: every control must hold its value ──
            # Discord's React can swallow a selection; never proceed (or
            # worse, submit) with Tag/Monat/Jahr still showing their
            # placeholders. Re-read each control and re-select anything
            # that didn't stick, then log the final state.
            dob_targets = (("Month", month_name), ("Day", day_val), ("Year", year_val))
            try:
                dob_missing = []
                for dob_label, dob_opt in dob_targets:
                    if not await self._dob_verify(dob_label, dob_opt):
                        dob_missing.append(dob_label)
                        self._log(f"[DOB] {dob_label} not verified after fill - re-selecting", level="warn")
                for dob_label, dob_opt in dob_targets:
                    if dob_label in dob_missing:
                        await self._select_dob(dob_label, dob_opt)
                        await self._human_pause()
                dob_state = {lbl: await self._dob_current_value(lbl) for lbl, _ in dob_targets}
                self._log(f"[DOB] Post-fill state: {json.dumps(dob_state)}")
            except Exception as _de:
                self._log_exception("[DOB] Post-fill verify failed", _de)

            await asyncio.sleep(1.0)

            # ── VERIFY ToS is actually checked before trying Create Account ──
            try:
                verify = await self._page.evaluate("""() => {
                    const cbs = document.querySelectorAll('input[type="checkbox"]');
                    let checked = 0;
                    for (const cb of cbs) {
                        if (cb.checked) checked++;
                    }
                    const roleCbs = document.querySelectorAll('[role="checkbox"][aria-checked="true"]');
                    return { native: checked, role: roleCbs.length };
                }""")
                self._log(f"[Form] Checkbox state: native={verify.get('native',0)} role={verify.get('role',0)}")
            except Exception as e:
                self._log_exception("[Form] Checkbox state read failed", e)

            # ── REAL ToS click — ONE pass, exactly one click per box ──
            # Without a genuinely checked ToS, "Continue" stays disabled and
            # the run can fall through to the login link. Real mouse clicks,
            # once per checkbox — never re-click (that toggles it back off).
            n = await self._click_tos_checkboxes()
            if n > 0:
                self._log("[Form] ToS checkbox(es) verified checked")

            # ── FINAL gate: never submit an empty form ──
            # DOB selection + the ToS clicks make Discord's React re-render,
            # which is exactly when a value it silently dropped reappears
            # empty. Re-stabilize the fields first (element-targeted
            # re-writes are safe — nothing can leak into another field now),
            # then read once more; if anything is STILL missing, rotate
            # instead of "faking" a Create Account on a blank form.
            #
            # DOB included: a select can show its value in the DOM yet have
            # the React state revert it on the ToS-triggered re-render
            # (the "Required" error on the DOB row at submit). Re-verify all
            # three right here and re-select anything that reverted — this
            # is the last chance before the click.
            for dob_label, dob_opt in (("Month", month_name),
                                       ("Day", day_val),
                                       ("Year", year_val)):
                try:
                    if not await self._dob_verify(dob_label, dob_opt):
                        self._log(f"[DOB] {dob_label} reverted to placeholder before submit — re-selecting {dob_opt}", level="warn")
                        await self._select_dob(dob_label, dob_opt)
                        await self._human_pause()
                        if not await self._dob_verify(dob_label, dob_opt):
                            self._log(f"[DOB] {dob_label} still not holding '{dob_opt}' after re-select", level="error")
                except Exception as _de:
                    self._log_exception(f"[DOB] Final {dob_label} re-verify failed", _de)
            await self._stabilize_credential_fields(
                self._build_cred_fields(self._display_name or self._username or ""))
            final_vals = await self._read_form_values()
            if not self._credentials_filled(final_vals):
                self._log(
                    "[Form] [FAIL] Fields empty right before Create Account — "
                    "aborting instead of faking a submit: "
                    + json.dumps({
                        "email": final_vals.get("email", ""),
                        "username": final_vals.get("username", ""),
                        "password_len": len(final_vals.get("password") or ""),
                    }),
                    level="error",
                )
                await self.capture_screenshot()
                return False

            # Discord may have replaced the generated username with its
            # own suggestion (name taken) - record what the form actually
            # holds so the saved account + log lines show the real
            # @username.
            try:
                real_user = (final_vals.get("username") or "").strip()
                if real_user:
                    self._username = real_user
            except Exception:
                pass

            # ── Create Account Button — try multiple strategies ────────
            # Humanization: pause to review the filled form before submitting.
            await asyncio.sleep(random.uniform(0.9, 1.9))
            self._log("Clicking Create Account...")

            # ── Click until the submit lands (max 5 clicks, ~3s apart) ──
            # Spec: one click per pass, verified after ~3s; if it didn't
            # land, click again on the SAME page. The page is NEVER
            # reloaded while registering - only the caller rotates for a
            # dead IP / invalid email.
            # A challenge iframe present BEFORE our first click is a preloaded
            # (empty) shell that proves nothing. One that APPEARS after a click
            # is Discord opening the challenge modal - stop clicking the moment
            # that happens so no click lands on the modal's X / backdrop.
            pre_challenge = (await self._challenge_iframe()) is not None
            for click_pass in range(1, 6):
                if self._stopped.is_set():
                    self._nav_error = "stopped by user"
                    return False
                if await self._rate_limited():
                    self._nav_error = "rate limited (429) by Discord"
                    self._log("[Form] RATE LIMITED during Create Account - rotating circuit", level="warn")
                    await self.capture_screenshot()
                    return False

                if click_pass > 1:
                    self._log(f"[Form] Create Account retry {click_pass}/5 - clicking again in ~3s (no page refresh)...")
                    # Re-check the REQUIRED ToS checkbox on retry (React may
                    # have reset it) - real mouse click, never the optional
                    # marketing box or a styled container div. Skip the
                    # coordinate click if the challenge modal is already up:
                    # the click would land outside the hCaptcha box.
                    if (await self._challenge_iframe()) is not None:
                        self._log("[Form] hCaptcha challenge present - skipping ToS re-click (never click outside the box)")
                    else:
                        try:
                            target = await self._page.evaluate(_TOS_TARGET_JS)
                            if target:
                                cx = target.get("lx", target["x"])
                                cy = target.get("ly", target["y"])
                                try:
                                    await hm.click(self._page, cx, cy)
                                except Exception:
                                    await self._page.mouse.click(cx, cy)
                                self._log("[Form] Re-checked ToS checkbox on retry")
                        except Exception as e:
                            self._log_exception("[Form] ToS re-check on retry failed", e)
                    await asyncio.sleep(1.0)  # Let React process

                clicked_this_pass = False

                # 1) PRIMARY: real engine-humanized mouse click at the
                #    button's center - works where synthetic clicks are
                #    swallowed (the same coords pattern that fixed the DOB
                #    dropdowns). Fallbacks below only run if no ENABLED
                #    button was found (e.g. still validating the username).
                if await self._real_click_create_button():
                    clicked_this_pass = True
                    self._log(f"[Form] Real mouse click sent (pass {click_pass}/5)")

                # 2) JS strategies: button text / type=submit / requestSubmit
                if not clicked_this_pass:
                    try:
                        result = await self._page.evaluate("""() => {
                            __LOGIN_LINK_GUARD__
                            const _norm = (s) => (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
                            // Strategy 1: Find button by text content (most reliable)
                            const btns = document.querySelectorAll('button, [role="button"], [type="submit"]');
                            for (const btn of btns) {
                                if (btn.offsetParent === null) continue;
                                // Check if disabled
                                if (btn.disabled || btn.getAttribute('aria-disabled') === 'true') continue;
                                // Never the "Already have an account?" / back-to-login
                                // link - Discord labels it with non-breaking spaces /
                                // split spans, so normalize whitespace and also check
                                // aria-label / title / value before trusting any text.
                                if (__isLoginLink(btn)) continue;
                                const t = _norm(btn.textContent);
                                const v = _norm(btn.value);
                                // ALL locales: Discord labels the submit button in
                                // the page's language (German "Konto erstellen",
                                // French "Créer un compte", Russian "Создать
                                // аккаунт", Korean "가입"...), so match the common
                                // spellings, not just English.
                                if (RegExp(__SUBMIT_TEXT_RE__).test(t + ' ' + v)) {
                                    btn.scrollIntoView({block: 'center'});
                                    btn.click();
                                    return 'btn_' + t.slice(0, 20);
                                }
                            }

                            // Strategy 2: real submit button - but NEVER a
                            // navigation button like "Already have an account?"
                            // (it's a type=submit button that navigates to /login
                            // and silently kills the run). Require an actual
                            // type="submit" inside a form + no login text.
                            for (const btn of btns) {
                                if (btn.offsetParent === null) continue;
                                if (btn.disabled || btn.getAttribute('aria-disabled') === 'true') continue;
                                if (__isLoginLink(btn)) continue;
                                if (btn.getAttribute('type') !== 'submit') continue;
                                const t = _norm(btn.textContent);
                                if (!btn.closest('form')) continue;
                                if (t.length > 2) {  // has meaningful text
                                    btn.scrollIntoView({block: 'center'});
                                    btn.click();
                                    return 'btntype_' + t.slice(0, 20);
                                }
                            }

                            // Strategy 3: Form submit - but NEVER let the default
                            // submit button be the "Already have an account?" login
                            // link: requestSubmit() with no argument activates the
                            // form's default submit button, which IS the login link
                            // whenever the real Continue button is disabled. Pick a
                            // real, enabled, non-login submit button explicitly.
                            const forms = document.querySelectorAll('form');
                            for (const form of forms) {
                                if (form.offsetParent === null) continue;
                                for (const sb of form.querySelectorAll('button[type="submit"], [type="submit"]')) {
                                    if (sb.disabled || sb.getAttribute('aria-disabled') === 'true') continue;
                                    if (sb.offsetParent === null) continue;
                                    if (__isLoginLink(sb)) continue;
                                    if (form.requestSubmit) {
                                        form.requestSubmit(sb);
                                        return 'form_requestSubmit';
                                    }
                                    sb.click();
                                    return 'form_submit_click';
                                }
                            }

                            return 'failed';
                        }""".replace('__LOGIN_LINK_GUARD__', _LOGIN_LINK_GUARD)
                            .replace('__SUBMIT_TEXT_RE__', json.dumps(_SUBMIT_TEXT_RE)))
                        if result and result != 'failed':
                            clicked_this_pass = True
                            self._log(f"[OK] Account button clicked (pass {click_pass}/5): {result}")
                    except Exception as e:
                        self._log(f"Create Account JS attempt (pass {click_pass}/5) error: {e}", level="warn")

                # 3) Playwright trusted click fallback
                if not clicked_this_pass:
                    try:
                        btn_selectors = [
                            'button:has-text("Create Account")',
                            'button:has-text("Sign Up")',
                            'button:has-text("Continue")',
                            'button:has-text("Registrieren")',
                            'button:has-text("Konto erstellen")',
                            'button:has-text("Créer un compte")',
                            'button:has-text("S\'inscrire")',
                            'button:has-text("Crear cuenta")',
                            'button:has-text("Registrarse")',
                            'button:has-text("Criar conta")',
                            'button:has-text("Cadastrar")',
                            'button:has-text("Aanmelden")',
                            'button:has-text("Registrera")',
                            'button:has-text("Opret konto")',
                            'button:has-text("Załóż konto")',
                            'button:has-text("Создать аккаунт")',
                            'button:has-text("Đăng ký")',
                            'button[type="submit"]',
                        ]
                        for sel in btn_selectors:
                            try:
                                btn = self._page.locator(sel).first
                                if await btn.count() > 0:
                                    is_disabled = await btn.is_disabled()
                                    if not is_disabled:
                                        # Never the "Already have an account?" /
                                        # back-to-login link - it navigates to
                                        # /login and silently kills the run.
                                        try:
                                            _txt = (await btn.inner_text() or "").lower()
                                        except Exception:
                                            _txt = ""
                                        try:
                                            _aria = (await btn.get_attribute("aria-label") or "").lower()
                                        except Exception:
                                            _aria = ""
                                        # Normalize whitespace - Discord labels the
                                        # login link with non-breaking spaces, so a
                                        # plain substring match misses it.
                                        _txt_norm = " ".join((_txt + " " + _aria).split())
                                        if any(k in _txt_norm for k in ("already have an account", "log in", "login", "sign in", "back to", "forgot")):
                                            self._log(f"[Form] Skipping fallback {sel} ({_txt_norm[:24]}) - login link", level="warn")
                                            continue
                                        await btn.scroll_into_view_if_needed()
                                        await btn.click()
                                        self._log(f"[OK] Playwright click: {sel} (pass {click_pass}/5)")
                                        clicked_this_pass = True
                                        break
                            except Exception:
                                continue
                    except Exception as pw_e:
                        self._log(f"Playwright button click error: {pw_e}", level="warn")

                # 4) Last resort: Enter key on password field - but ONLY when
                #    the form's default submit button is not the "Already have
                #    an account?" login link (Enter triggers implicit
                #    submission via the default submit button; when the real
                #    Continue is disabled, that default IS the login link and
                #    would send the run to /login).
                if not clicked_this_pass:
                    try:
                        safe_enter = bool(await self._page.evaluate("""() => {
                            __LOGIN_LINK_GUARD__
                            const form = document.querySelector('form');
                            if (!form) return false;
                            for (const sb of form.querySelectorAll('button[type="submit"], [type="submit"]')) {
                                if (sb.disabled || sb.getAttribute('aria-disabled') === 'true') continue;
                                if (sb.offsetParent === null) continue;
                                if (__isLoginLink(sb)) continue;
                                return true;
                            }
                            return false;
                        }""".replace('__LOGIN_LINK_GUARD__', _LOGIN_LINK_GUARD)))
                    except Exception as e:
                        self._log_exception("[Form] Enter-safety check failed", e)
                        safe_enter = False
                    if safe_enter:
                        try:
                            await self._page.locator('input[name="password"]').press('Enter')
                            self._log(f"Pressed Enter on password field (pass {click_pass}/5)")
                            clicked_this_pass = True
                        except Exception as e:
                            self._log_exception("[Form] Enter key fallback failed", e)

                if not clicked_this_pass:
                    self._log(f"[Form] No enabled Create Account button found (pass {click_pass}/5)", level="warn")

                # Wait ~3s, then PROVE the submit landed before moving on.
                # No page refresh - if it didn't land, the next pass clicks
                # again on the SAME session.
                await asyncio.sleep(3.0)
                reason = await self._submit_landed(timeout=2.5)
                if reason:
                    self._log(f"[OK] Create Account submit verified after click {click_pass}/5 ({reason})")
                    await self.capture_screenshot()
                    return True
                # The challenge iframe APPEARED after our click: the submit
                # landed and Discord is now loading the hCaptcha challenge.
                # Stop clicking right now - another coordinate click would hit
                # the modal (its close X or the backdrop) and dismiss the
                # challenge before it finishes loading.
                if not pre_challenge and (await self._challenge_iframe()) is not None:
                    self._log("[Form] hCaptcha challenge appeared - submit landed, letting it render (no further clicks)")
                    await self.capture_screenshot()
                    return True
                if click_pass < 5:
                    self._log("[Form] Submit not landed yet - clicking again in ~3s (no page refresh)", level="warn")

            # ── All 5 clicks failed - dump the form so the failure is
            # self-explanatory instead of a silent stall. ──
            await self._log_form_state("after Create Account clicks (not landed)")
            self._nav_error = "Create Account clicked 5x but the form never submitted (see dump)"
            self._log("[FAIL] Create Account never submitted after 5 clicks", level="error")
            await self.capture_screenshot()
            return False

        except Exception as e:
            self._log_exception("Form filling error", e)
            return False

    async def _challenge_iframe(self):
        """First hCaptcha CHALLENGE iframe element, or None."""
        try:
            chall = self._page.locator(
                'iframe[title*="hCaptcha challenge"], iframe[src*="hcaptcha-challenge"]')
            if await chall.count() > 0:
                return chall.first
        except Exception:
            pass
        return None

    async def _submit_landed(self, timeout: float = 4.0) -> str:
        """Proof the register form actually submitted. Returns a reason
        string ("" = still sitting on the unsubmitted form).

        Signals: the hCaptcha CHALLENGE frame appeared (Discord shows it
        inside the register modal after a successful submit), the URL moved
        to /app, /channels or a verify page, or the register form unmounted.
        The plain widget iframe (newassets.hcaptcha.com) is mounted WITH the
        form before any click and proves nothing - treating it as proof made
        the bot declare success and click the pre-existing widget while the
        form was still unsent."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                raw = await self._page.evaluate("""() => {
                    const f = document.querySelector('form');
                    return JSON.stringify({
                        url: location.href || '',
                        form: !!f,
                        challenge: !!document.querySelector('iframe[src*="hcaptcha-challenge"], iframe[title*="hCaptcha challenge" i]'),
                    });
                }""")
                st = json.loads(raw) if raw else {}
            except Exception:
                st = {}
            if not st:
                # Read failed (page mid-navigation / eval hiccup) - that
                # is NOT proof the form submitted. Keep polling instead
                # of falsely reporting a landed submit.
                await asyncio.sleep(0.4)
                continue
            # The hCaptcha CHALLENGE frame proves the submit when it is
            # genuinely RENDERED. Discord keeps the register form in the
            # DOM and layers the rendered challenge over it, so requiring
            # the form to unmount made the bot report "never submitted"
            # while the challenge was already showing. A preloaded shell
            # iframe (empty children, no painted content — the earlier
            # false-positive case) is NOT rendered and still doesn't count.
            if st.get("challenge"):
                _chall_el = await self._challenge_iframe()
                if _chall_el is not None and await self._challenge_rendered(_chall_el):
                    return "captcha_challenge_rendered"
            url = str(st.get("url") or "")
            if any(k in url for k in ("discord.com/app", "discord.com/channels", "/verify")):
                return "url:" + url[:60]
            if not st.get("form"):
                return "form_gone:" + url[:60]
            await asyncio.sleep(0.4)
        return ""

    async def _real_click_create_button(self) -> bool:
        """REAL engine-humanized mouse click at the Create Account button's
        center (trusted input via page.mouse.click - the same coords-first
        pattern that fixed the DOB dropdowns). A JS btn.click() can be
        swallowed by native validation / overlays; a physical click triggers
        Discord's own handler and surfaces any inline validation error.
        Returns True when a click was sent."""
        try:
            # Never fire a coordinate click while Discord's challenge modal is
            # up: the button is BEHIND the modal, so the click lands on the
            # overlay (its close X or the backdrop) and dismisses the
            # challenge. Fall through to the safe JS click strategies instead.
            if (await self._challenge_iframe()) is not None:
                self._log("[Form] hCaptcha challenge present - skipping physical Create Account click", level="debug")
                return False
            pos = await self._page.evaluate("""() => {
                __LOGIN_LINK_GUARD__
                const _norm = (s) => (s || '').toLowerCase().replace(/\\s+/g, ' ').trim();
                const btns = document.querySelectorAll('button, [role="button"], [type="submit"]');
                for (const btn of btns) {
                    if (btn.offsetParent === null) continue;
                    if (btn.disabled || btn.getAttribute('aria-disabled') === 'true') continue;
                    if (__isLoginLink(btn)) continue;
                    const t = _norm(btn.textContent) + ' ' + _norm(btn.value);
                    if (!RegExp(__SUBMIT_TEXT_RE__).test(t)) continue;
                    btn.scrollIntoView({ block: 'center' });
                    const r = btn.getBoundingClientRect();
                    if (!r || r.width < 4 || r.height < 4) continue;
                    return { x: r.left + r.width / 2, y: r.top + r.height / 2, text: t.slice(0, 24) };
                }
                return null;
            }""".replace('__LOGIN_LINK_GUARD__', _LOGIN_LINK_GUARD)
                .replace('__SUBMIT_TEXT_RE__', json.dumps(_SUBMIT_TEXT_RE)))
            if not pos or not pos.get("x"):
                self._log("[Form] No enabled Create Account button found for real click", level="warn")
                return False
            await self._page.mouse.click(float(pos["x"]), float(pos["y"]))
            self._log(f"[Form] Real mouse click on Create Account ({pos.get('text')})")
            return True
        except Exception as e:
            self._log_exception("[Form] Real Create Account click failed", e)
            return False

    async def _log_form_state(self, tag: str) -> None:
        """Dump the register form's live state (values, inline validation
        errors, DOB reads, checkbox count, submit button states) so a
        non-submitting form is diagnosable instead of a silent stall."""
        try:
            dump = await self._page.evaluate("""() => {
                const errs = [];
                for (const e of document.querySelectorAll('[class*="error" i], [class*="warning" i], [role="alert"], [data-reactid*="error" i]')) {
                    if (e.offsetParent === null) continue;
                    const t = (e.textContent || '').trim().replace(/\\s+/g, ' ');
                    if (t && t.length < 160) errs.push(t);
                }
                const inputs = Array.from(document.querySelectorAll('input'))
                    .filter(e => e.offsetParent !== null)
                    .map(e => ({ name: e.name || '', type: e.type || '', val: (e.value || '').slice(0, 50) }));
                const dob = {};
                for (const el of document.querySelectorAll('[data-dob-target]')) {
                    const t = (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 40);
                    if (t && t.length <= 40) dob[el.getAttribute('data-dob-target')] = t;
                }
                const boxes = document.querySelectorAll('input[type="checkbox"]:checked, [role="checkbox"][aria-checked="true"], [role="checkbox"][data-state="checked"]').length;
                const btns = Array.from(document.querySelectorAll('button'))
                    .filter(e => e.offsetParent !== null)
                    .slice(0, 10)
                    .map(e => ({ t: (e.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 24),
                                dis: e.disabled || e.getAttribute('aria-disabled') === 'true' }));
                return JSON.stringify({ errors: errs.slice(0, 6), inputs, dob, checkboxes: boxes, buttons: btns });
            }""")
            self._log(f"[Form] POST-CLICK {tag}: {dump}", level="warn")
        except Exception as e:
            self._log_exception("[Form] POST-CLICK dump failed", e)

    async def _tos_checked_count(self) -> int:
        """How many real checkbox controls are currently checked."""
        try:
            return int(await self._page.evaluate("""() => document.querySelectorAll(
                'input[type="checkbox"]:checked, [role="checkbox"][aria-checked="true"], [role="checkbox"][data-state="checked"]').length""") or 0)
        except Exception:
            return 0

    async def _tos_continue_enabled(self) -> bool:
        """True when the submit button is enabled — a genuinely checked ToS
        is what enables it (also true on builds with no checkbox at all)."""
        try:
            return bool(await self._page.evaluate("""() => {
                for (const b of document.querySelectorAll('button')) {
                    const t = (b.textContent || '').toLowerCase().replace(/\s+/g, ' ').trim();
                    if (!RegExp(__SUBMIT_TEXT_RE__).test(t)) continue;
                    if (b.disabled || b.getAttribute('aria-disabled') === 'true') continue;
                    return true;
                }
                return false;
            }""".replace('__SUBMIT_TEXT_RE__', json.dumps(_SUBMIT_TEXT_RE))))
        except Exception:
            return False

    async def _click_tos_checkboxes(self) -> int:
        """Click Discord's REQUIRED ToS checkbox and VERIFY React registered
        it (the check must survive a re-render).

        Every pass: locate the ToS box, click its center with a trusted mouse
        click, then VERIFY (a real checkbox became checked, or the submit
        button enabled) before doing anything else. If the click didn't
        register, the click is dispatched on the element itself via JS
        (bypasses any overlay that swallowed the mouse events) and native
        inputs are force-checked. Only the Terms-of-Service box is clicked —
        never the optional marketing/email-updates box (any locale) and never
        styled container divs that also match [class*="checkbox"].
        """
        clicked = 0
        for _attempt in range(4):
            if self._stopped.is_set():
                break
            if await self._tos_checked_count() > 0 or await self._tos_continue_enabled():
                break
            try:
                target = await self._page.evaluate(_TOS_TARGET_JS)
            except Exception:
                target = None
            if not target:
                # No checkbox matched the standard selectors. Discord
                # renders the ToS box differently in some layouts (styled
                # div without role/data-state, button element, etc.) — dump
                # the real DOM so the next failure is diagnosable instead of
                # a silent skip. After the position fallback we continue to
                # the next pass (target is None — never fall through to the
                # coordinate click below, which would hit a None target).
                try:
                    dump = await self._page.evaluate("""() => {
                        const seen = new Set();
                        const out = [];
                        const textOf = (el) => ((el && el.innerText) || '')
                            .replace(/\s+/g, ' ').trim().slice(0, 90);
                        const info = (el) => {
                            const r = el.getBoundingClientRect();
                            return {
                                tag: el.tagName.toLowerCase(),
                                cls: (el.className || '').toString().slice(0, 50),
                                role: el.getAttribute('role') || '',
                                ds: el.getAttribute('data-state') || '',
                                ac: el.getAttribute('aria-checked') || '',
                                checked: !!el.checked,
                                w: Math.round(r.width), h: Math.round(r.height),
                                vis: r.width > 0 && r.height > 0,
                                txt: textOf(el),
                                parent: textOf(el.parentElement),
                            };
                        };
                        // Every element that could BE or CONTAIN the ToS box
                        for (const el of document.querySelectorAll(
                            'input[type="checkbox"], [role="checkbox"], [data-state], [aria-checked], ' +
                            '[class*="checkbox" i], [class*="checkBox" i], [class*="tos" i], ' +
                            '[class*="terms" i], [class*="agree" i], label, button')) {
                            if (seen.has(el)) continue;
                            seen.add(el);
                            const r = el.getBoundingClientRect();
                            const t = textOf(el).toLowerCase();
                            if (r.width < 4 || r.height < 4) continue;
                            if (!/checkbox|terms|tos|agree|nutzung|datenschutz|gelesen|akzeptier|service|conditions|label|button/i.test(
                                    (el.className || '') + ' ' + (el.getAttribute('role') || '') + ' ' + t)) continue;
                            out.push(info(el));
                        }
                        // Also: the 3 visible elements directly ABOVE the
                        // submit button (ToS row sits right above it).
                        const btns = Array.from(document.querySelectorAll('button'));
                        const submit = btns.filter(b => b.offsetParent !== null)
                            .sort((a, b) => b.getBoundingClientRect().top - a.getBoundingClientRect().top)[0];
                        if (submit) {
                            const row = submit.parentElement;
                            if (row) {
                                let prev = row.previousElementSibling || row.previousSibling;
                                for (let i = 0; prev && i < 3; i++) {
                                    if (prev.nodeType === 1 && !seen.has(prev)) {
                                        seen.add(prev);
                                        out.push(info(prev));
                                    }
                                    prev = prev.previousElementSibling || prev.previousSibling;
                                }
                            }
                        }
                        return JSON.stringify(out);
                    }""")
                    self._log(f"[Form] ToS NO TARGET — checkbox DOM dump: {dump}", level="warn")
                except Exception as _e:
                    self._log_exception("[Form] ToS DOM dump failed", _e)
                # ── Position-based fallback ──
                # No standard checkbox matched, but Discord always renders
                # the ToS row directly above the Create Account button.
                # Click the box-like element in that row.
                try:
                    fb = await self._page.evaluate(_TOS_FALLBACK_JS)
                except Exception:
                    fb = None
                if not fb:
                    break
                try:
                    await hm.click(self._page, fb["x"], fb["y"])
                    clicked += 1
                    self._log(
                        f"[Form] ToS clicked via position fallback (tag={fb.get('tag')}, "
                        f"label={fb.get('label') or '?'})")
                except Exception:
                    pass
                await asyncio.sleep(0.25)
                if await self._tos_checked_count() > 0 or await self._tos_continue_enabled():
                    break
                try:
                    r = await self._page.evaluate(_TOS_CLICK_JS)
                    if r:
                        self._log(f"[Form] ToS JS-dispatch fallback: {r}")
                except Exception:
                    pass
                await asyncio.sleep(0.25)
                continue
            # 1) trusted (humanized) mouse click at the box/label centre.
            #    Prefer the label/row point returned by _TOS_TARGET_JS —
            #    clicking the label toggles the box and lands on a bigger,
            #    more reliable target.
            cx = target.get("lx", target["x"])
            cy = target.get("ly", target["y"])
            try:
                await hm.click(self._page, cx, cy)
                clicked += 1
            except Exception:
                try:
                    await self._page.mouse.click(cx, cy)
                    clicked += 1
                except Exception:
                    pass
            await asyncio.sleep(0.3)
            if await self._tos_checked_count() > 0 or await self._tos_continue_enabled():
                break
            # 2) JS dispatch on the element itself (a transparent overlay or
            #    a moving page can swallow the trusted click)
            try:
                r = await self._page.evaluate(_TOS_CLICK_JS)
                if r:
                    self._log(f"[Form] ToS JS-dispatch fallback: {r}")
            except Exception:
                pass
            await asyncio.sleep(0.25)
        verified = await self._tos_checked_count()
        continue_enabled = await self._tos_continue_enabled()
        self._log(f"[Form] ToS checkboxes: clicked {clicked}, verified {verified}, continue_enabled={continue_enabled}")
        if verified > 0 or continue_enabled:
            return max(verified, 1)
        return 0

    async def _fill_missing_fields(self, display_name: str) -> None:
        """Robust fallback for fields a write didn't keep.

        Delegates to the self-healing credential fill: fill order ends with
        email (so any keystroke leak is overwritten by the email write) and
        every field is re-verified + healed after each write.
        """
        await self._fill_credential_fields(display_name)

    async def _human_pause(self) -> None:
        await asyncio.sleep(random.uniform(0.08, 0.2))

    async def live_camera_loop(self, interval: int = 4) -> None:
        while True:
            await self.capture_screenshot()
            await asyncio.sleep(interval)

    async def _extract_token(self, attempts: int = 4,
                             poll_rounds: int = 10) -> str:
        """Login to Discord with the created account and grab the FULL token
        from localStorage. Discord stores it under 'token'.

        poll_rounds x 2s bounds the wait (20s default); pass a larger value
        (e.g. 30 = 60s) when a custom email needs manual verification before
        the login unlocks."""
        if not (self._email and self._password):
            return ""
        try:
            for i in range(attempts):
                try:
                    await self._page.goto("https://discord.com/login",
                                          wait_until="domcontentloaded",
                                          timeout=NAV_TIMEOUT_MS)
                    break
                except Exception:
                    await asyncio.sleep(2)
            await asyncio.sleep(1.5)
            try:
                email_input = self._page.locator('input[name="email"]').first
                await email_input.fill(self._email, timeout=8000)
                pw_input = self._page.locator('input[name="password"]').first
                await pw_input.fill(self._password, timeout=8000)
                await pw_input.press("Enter")
                self._log("[Token] Submitted login form")
            except Exception as e:
                self._log(f"[Token] Login fill error: {e}", level="warn")
                return ""
            # Wait for token to appear (abort early if phone-gated at login).
            # The React login form can eat the first Enter on a cold load, so
            # re-submit once after ~8s if the token still hasn't landed.
            resubmitted = False
            for round_i in range(poll_rounds):
                await asyncio.sleep(2.0)
                try:
                    if await self._detect_phone_verification():
                        self.phone_verify_detected = True
                        self._log("[Phone] [DETECTED] login gated by phone verification", level="warn")
                        return ""
                except Exception:
                    pass
                try:
                    token = await self._page.evaluate(
                        "() => localStorage.getItem('token') || ''"
                    )
                    if token and len(token) > 20:
                        return token.strip()
                except Exception:
                    pass
                if not resubmitted and round_i >= 3:
                    resubmitted = True
                    self._log("[Token] No token yet - re-submitting login form", level="warn")
                    try:
                        await self._page.evaluate("""() => {
                            const f = document.querySelector('form');
                            const btn = document.querySelector('button[type="submit"]');
                            if (f && f.requestSubmit) { f.requestSubmit(); return 'submitted'; }
                            if (btn) { btn.click(); return 'clicked'; }
                            return 'none';
                        }""")
                    except Exception:
                        pass
            return ""
        except Exception as e:
            self._log(f"[Token] extract error: {e}", level="warn")
            return ""

    def get_account(self) -> dict:
        """Return the generated account info (email, user, pass, full token)."""
        return {
            "email": self._email,
            "username": self._username,
            "password": self._password,
            "token": self._token,
            "proxy": self.proxy,
            "worker_id": self.worker_id,
            "user_id": self._user_id,
            "avatar": self._avatar_data,
            "bio": self._bio,
            "humanized": self._humanized,
            "domain": self._domain,
        }

    async def close(self) -> None:
        if self._mail:
            try:
                await self._mail.close()
            except Exception:
                pass
            self._mail = None
        if self._page:
            try:
                await self._page.close()
            except:
                pass
            self._page = None
        if self._context:
            try:
                await self._context.close()
            except:
                pass
            self._context = None
        if self._browser:
            try:
                await self._browser.close()
            except:
                pass
            self._browser = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except:
                pass
            self._playwright = None

    def get_screenshots(self) -> list:
        return self._screenshots

    def get_latest_screenshot(self) -> str:
        if self._screenshots:
            return self._screenshots[-1]
        return ""


async def run_discord_automation():
    # Standalone CLI path — use a residential session when available
    # (vaultproxies.txt / VAULTPROXY_* env), TOR otherwise.
    proxy = None
    try:
        from proxies import pool as _proxy_pool
        if _proxy_pool.count == 0:
            await _proxy_pool.refresh()
        if _proxy_pool.count > 0:
            proxy = _proxy_pool.take()
            print(f"[CLI] Using proxy session: {proxy.get('key', '?')[:48]}...", flush=True)
    except Exception as e:
        print(f"[CLI] Proxy pool unavailable ({e}) — using TOR", flush=True)
    bot = DiscordAutomation(headless=True, proxy=proxy)
    try:
        await bot.initialize()
        success = await bot.start_discord_signup()
        if success:
            print("[OK] Discord automation completed")
        else:
            print("[FAIL] Discord automation failed")
        await asyncio.sleep(5)
    finally:
        await bot.close()


if __name__ == "__main__":
    asyncio.run(run_discord_automation())
