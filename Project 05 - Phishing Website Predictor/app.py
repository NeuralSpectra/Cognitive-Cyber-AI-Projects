import streamlit as st
import pandas as pd
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import tldextract

st.set_page_config(page_title="Phishing Website Predictor (Auto Extract)", page_icon="🔎", layout="centered")
st.title("🔎 Phishing Website Predictor — Auto Feature Extraction")
st.write("Paste a URL and the app will extract features automatically, then predict whether the URL is phishing or legitimate.")

st.caption("**⚠️ Note: Some Websites May Block Automated Requests. In Such Cases, The App Cannot Fetch The Page Content, And The Prediction Is Based Only On URL Features. For Best Results, Use URLs That Are Publicly Accessible.**")

try:
    def extract_domain(hostname):
        ext = tldextract.extract(hostname)
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        return domain
    
except Exception:
    def extract_domain(hostname):
        hostname = hostname.split(":")[0].lower()
        if hostname.startwith("www."):
            hostname = hostname[4:]
        return hostname
        
@st.cache_resource
def load_model(path="Phishing-Website-Predictor.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
model = load_model()

def safe_request(url, timeout=6):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PhishChecker/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text, r.url
    except  Exception:
        return None, None
    
def count_sensitive_words(url):
    sensitive = ['login', 'secure', 'account', 'update', 'verify', 'bank', 'password', 'confirm', 'signin', 'click', 'ebay', 'paypal']
    u = url.lower()
    count = 0
    for word in sensitive:
        count+= u.count(word)
    return count
    
def domain_of(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ''
    return extract_domain(hostname)

def calc_path_level(path):
    if not path or path == "/":
        return 0
    segs = [s for s in path.split("/") if s]
    return len(segs)

def percentage(part, total):
    try:
        if total == 0:
            return 0
        return round((part/total)*100, 2)
    except Exception:
        return 0
    
def extract_features_from_url(url):
    if not re.match(r'^https?://', url):
        url = "http://" + url
    
    parsed = urlparse(url)
    base_domain = domain_of(url)
    hostname = parsed.hostname or ''
    path = parsed.path or ""

    NumDots = hostname.count(".")
    PathLevel = calc_path_level(path)
    NumDash = url.count("-")
    NumSensitiveWords = count_sensitive_words(url)

    html, final_url = safe_request(url)
    if final_url:
        base_domain = domain_of(final_url)

    # default values for all features
    NumDots = hostname.count('.')
    PathLevel = calc_path_level(path)
    NumDash = url.count('-')
    NumSensitiveWords = count_sensitive_words(url)
    PctExtHyperlinks = 0
    PctExtResourceUrls = 0
    InsecureForms = 0
    PctNullSelfRedirectHyperlinks = 0
    FrequentDomainNameMismatch = 0
    SubmitInfoToEmail = 0
    IframeOrFrame = 0
    total_a = 0
    total_res = 0

    if html:
        soup = BeautifulSoup(html, "html.parser")

        anchors = [a for a in soup.find_all("a", href=True)]
        total_a = len(anchors)
        ext_a = 0
        null_a = 0
        for a in anchors:
            href = a.get("href").strip()
            if href in ['', '#'] or href.lower().startswith("javascript:") or href == parsed.path:
                null_a += 1
                continue

            if re.match(r'^https?://', href):
                link_domain = domain_of(href)
            else:
                link_domain = base_domain
            
            if link_domain and link_domain!= base_domain:
                ext_a += 1

        PctExtHyperlinks = percentage(ext_a, total_a)
        PctNullSelfRedirectHyperlinks = percentage(null_a, total_a)

        resources = []
        resources += [img.get('src') for img in soup.find_all("img") if img.get("src")]
        resources += [script.get("src") for script in soup.find_all("script") if script.get("src")]
        resources += [link.get("href") for link in soup.find_all("link") if link.get("href")]
        resources = [r for r in resources if r]
        total_res = len(resources)
        ext_res = 0
        for r in resources:
            r = r.strip()
            if re.match(r"^https?://", r):
                rdom = domain_of(r)
            else:
                rdom = base_domain
            if rdom and rdom != base_domain:
                ext_res += 1

        PctExtResourceUrls = percentage(ext_res, total_res)

        forms = soup.find_all("form")
        insecure_forms_count = 0
        mailto_found = False
        for f in forms:
            action = (f.get("action") or "").strip()
            if action.lower().startswith("mailto:"):
                mailto_found = True
            if action and re.match(r"^https?://", action):
                if not action.lower().startswith("https://"):
                    insecure_forms_count += 1

            if action == "":
                insecure_forms_count += 0

            for inp in f.find_all(["input", "textarea"]):
                itype = (inp.get("type") or '').lower()
                if itype == "email":
                    mailto_found = mailto_found or False

        InsecureForms = 1 if insecure_forms_count > 0 else 0
        SubmitInfoToEmail = 1 if mailto_found > 0 else 0

        if soup.find("iframe") or soup.find("frame"):
            IframeOrFrame = 1
        
        if total_a > 0 and PctExtHyperlinks > 30:
            FrequentDomainNameMismatch = 1
        else:
            FrequentDomainNameMismatch = 0

    features = {
        'NumDots': int(NumDots),
        'PathLevel': int(PathLevel),
        'NumDash': int(NumDash),
        'NumSensitiveWords': int(NumSensitiveWords),
        'PctExtHyperlinks': int(round(PctExtHyperlinks)),
        'PctExtResourceUrls': int(round(PctExtResourceUrls)),
        'InsecureForms': int(InsecureForms),
        'PctNullSelfRedirectHyperlinks': int(round(PctNullSelfRedirectHyperlinks)),
        'FrequentDomainNameMismatch': int(FrequentDomainNameMismatch),
        'SubmitInfoToEmail': int(SubmitInfoToEmail),
        'IframeOrFrame': int(IframeOrFrame)
    }

    return features, {
        'total_links': total_a if html else 0,
        'total_resources': total_res if html else 0,
        'fetched': True if html else False,
        'final_url': final_url or url,
        'base_domain': base_domain
    }


input_url = st.text_input("Enter URL (with our without http/https)", placeholder="example.com/login")

if st.button("🔎 Extract & Predict"):
    if not input_url:
        st.warning("Please Provide A URL.")
    else:
        with st.spinner("Fetching & Extracting Features...."):
            features, meta = extract_features_from_url(input_url)
        
        st.markdown('### Extract Features (auto)')
        st.table(pd.DataFrame([features]))

        st.markdown("**Meta**")
        st.write(meta)

        df = pd.DataFrame([features])

        try:
            prediction = model.predict(df)
            prob = None
            try:
                prob = model.predict_proba(df)[0]
            except Exception:
                prob = None
            
            st.markdown("### Prediction")
            if int(prediction[0]) == 1:
                st.error(f"⚠️ Model says: **Phishing** (label=1)")
            else:
                st.success(f"✅ Model says: **Legitimate** (label=0)")

            if prob is not None:
                if len(prob) >= 2:
                    phish_prob = round(prob[1]*100, 2)
                    st.info(f"Model Confidence (phishing): {phish_prob} %")
                else:
                    st.info(f"Model Probabilities: {prob}")

        except Exception as e:
            st.error("Prediction Failed. See Details Below.")
            st.exception(e)

st.markdown("---")
st.caption("Approximate Feature Extraction Is Used For Some Features (Externel Link/Resource Percentages, Insecure Forms Detection). For Production Use, Adapt Heuristics To Match The Exact Feature Definitions From Your Training Data.")

