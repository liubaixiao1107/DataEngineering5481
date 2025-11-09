import argparse, os, json, re, time, logging, threading, socket, random
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("topic2")

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def load_json(p):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)
def domain_of(url: str) -> str:
    try: return urlparse(url).netloc or ""
    except: return ""
def is_refusal(text: str) -> bool:
    if not text: return True
    pats = [r"很抱歉", r"无法提供", r"不予(?:显示|提供|回答)", r"违反(?:内容|安全|合规)", r"不支持的请求", r"违规"]
    return any(re.search(p, text) for p in pats)

_SENT_SPLIT = re.compile(r'(?<=[。！？!?\.])\s+(?=[A-Z“"(\[]|[A-Z][a-z])')
def split_sentences(text: str):
    t = re.sub(r'\s+', ' ', (text or "").strip())
    parts = _SENT_SPLIT.split(t)
    if len(parts) <= 1:
        parts = re.split(r'(?<=[\.!?])\s+', t)
    return [s.strip() for s in parts if s.strip()]

def refactor_to_three_paragraphs(text: str):
    sents = split_sentences(text)
    if not sents: return (text or "").strip()
    n = len(sents)
    if n <= 3:
        p1 = " ".join(sents[:1]); p2 = " ".join(sents[1:2]); p3 = " ".join(sents[2:])
        return "\n\n".join([p for p in (p1,p2,p3) if p]).strip()
    p1_len = min(5, max(3, n//6 or 3))
    rem = n - p1_len
    p2_len = min(10, max(6, rem//2 or 6))
    p3_len = n - p1_len - p2_len
    if p3_len < 3 and n >= 12:
        move = min(3 - p3_len, p2_len - 6)
        if move > 0:
            p2_len -= move
            p3_len += move
    p1 = " ".join(sents[:p1_len]).strip()
    p2 = " ".join(sents[p1_len:p1_len+p2_len]).strip()
    p3 = " ".join(sents[p1_len+p2_len:]).strip()
    return "\n\n".join([p for p in (p1,p2,p3) if p]).strip()

_NOBEL_PAT = re.compile(r"\bNobel\b|诺贝尔|诺貝爾|诺奖|諾獎", re.IGNORECASE)
_YEAR_2025_PAT = re.compile(r"\b2025\b|2025年")

def is_nobel_related(title: str, text: str = "") -> bool:
    t = (title or "").strip()
    if not t: return False
    return bool(_NOBEL_PAT.search(t) or _NOBEL_PAT.search(text or ""))

def is_year_2025(title: str, date_str: str, text: str = "") -> bool:
    if _YEAR_2025_PAT.search(title or "") or _YEAR_2025_PAT.search(text or ""):
        return True
    if (date_str or "").startswith("2025-"):
        return True
    return False

class AdaptiveLimiter:
    def __init__(self, qps: float = 0.5, min_qps: float = 0.15, max_qps: float = 1.2):
        self.lock = threading.Lock(); self.qps=qps; self.min_qps=min_qps; self.max_qps=max_qps
        self.last = 0.0; self.cool_until = 0.0
    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.cool_until:
                time.sleep(self.cool_until - now); now = time.monotonic()
            interval = 1.0 / max(self.qps, self.min_qps)
            delta = interval - (now - self.last)
            if delta > 0: time.sleep(delta); now = time.monotonic()
            self.last = now
    def punish_429(self):
        with self.lock:
            self.qps = max(self.min_qps, self.qps * 0.6)
            cool = 6.0 + random.random()*6.0
            self.cool_until = time.monotonic() + cool
            log.warning(f"触发 429：降速到 {self.qps:.2f} QPS，并强冷却 {cool:.1f}s")
    def punish_timeout(self):
        with self.lock:
            self.qps = max(self.min_qps, self.qps * 0.8)
            log.warning(f"请求超时：降速到 {self.qps:.2f} QPS")

#prompts
TIMELINE_PREAMBLE = (
    "Here is an authoritative timeline (date + event) that MUST be treated as ground truth. "
    "When headlines conflict, resolve in favor of the timeline. "
    "Do NOT invent prizewinners or entities not supported by the timeline/headlines."
)

OVERALL_PROMPT = """
You will receive:
1) An authoritative timeline of the 2025 Nobel Prizes (ground truth).
2) Optionally, a list of headlines (title, publisher domain, date).

Your task: write a **single English narrative summary** ONLY about the **2025 Nobel Prizes**.

STRICT FORMAT & STYLE:
• **Output EXACTLY THREE PARAGRAPHS**, with a blank line between paragraphs.
• Paragraph 1 (3–5 sentences): concise highlights in flowing prose — e.g.,
  “the Chemistry Prize honored developers of metal-organic frameworks (MOFs), highlighting their impact on materials science
   with applications in gas storage, separation, and catalysis. The Nobel Committee then recognized Hungarian author László
   Krasznahorkai with the Literature Prize… The Peace Prize was awarded to Venezuelan opposition leader María Corina Machado…”.
  Use such phrasing **only if these facts are supported**; otherwise use generic wording without adding specifics.
• Paragraph 2 (~200–300 words): an integrative overview linking scientific breakthroughs and societal meaning.
• Paragraph 3 (~180–260 words): synthesize 3–5 cross-cutting themes with transitions (meanwhile, in turn, as a result, by contrast…).
  Do **not** enumerate by date or outlet.

HARD CONSTRAINTS:
• **NO FABRICATION**: do not invent winners, categories, dates, affiliations, numbers, or methods.
• **TIMELINE-ALIGNED**: if any conflict arises, prefer the timeline; otherwise generalize.
• **FOCUS**: Ignore any non-Nobel or non-2025 items entirely.
• Output plain text only (no JSON, no headers, no lists).
"""

def make_session():
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=2)
    s.mount("https://", adapter); s.mount("http://", adapter)
    s.headers.update({"Accept":"application/json", "Connection":"close"})
    return s

def call_glm(session: requests.Session, limiter: AdaptiveLimiter, api_key: str, messages,
             model="glm-4.5-flash", temperature=0.26, max_tokens=1400,
             timeout=120, max_retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    last_err = None
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            resp = session.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                j = resp.json()
                txt = j.get("choices", [{}])[0].get("message", {}).get("content", "")
                return txt
            if resp.status_code == 429:
                limiter.punish_429()
                time.sleep(1.2 + random.random()); continue
            if resp.status_code in (408, 500, 502, 503, 504):
                wait = (1.8 + random.random()) * (2 ** attempt)
                log.warning(f"临时错误 {resp.status_code}：{resp.text[:160]}...，{wait:.1f}s 后重试")
                time.sleep(wait); continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        except (requests.Timeout, socket.timeout) as e:
            last_err = e; limiter.punish_timeout()
            if attempt < max_retries:
                wait = (1.2 + random.random()) * (2 ** attempt)
                log.warning(f"请求超时，{wait:.1f}s 后重试")
                time.sleep(wait); continue
            break
        except requests.RequestException as e:
            last_err = e; break
    raise RuntimeError(f"模型调用失败：{last_err or 'unknown error'}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="cleaned_news.json")
    ap.add_argument("--timeline", default="timeline.json")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--model", default="glm-4.5-flash")
    ap.add_argument("--api-key", default="d0b8bc52cf6b4c368982dfdd32384757.UcWBjZr72H7AWgyN")
    ap.add_argument("--qps", type=float, default=0.5)
    ap.add_argument("--use-headlines", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    # load timeline
    if not args.timeline or not os.path.exists(args.timeline):
        log.warning("找不到 timeline 文件，只基于 headlines")
        timeline = []
    else:
        timeline = load_json(args.timeline)
        if not isinstance(timeline, list):
            log.warning(" "); timeline = []
        else:
            log.info(f"已加载 timeline：{args.timeline}（{len(timeline)} 条）")
    # load news
    if not os.path.exists(args.input):
        log.error(f"找不到输入文件：{args.input}"); return
    data = load_json(args.input)
    log.info(f"读取数据：{args.input}，总条目：{len(data)}")

    nobel_items = []
    for it in data:
        title = (it.get("title") or "").strip()
        text  = (it.get("text")  or "")
        date  = (it.get("date")  or "unknown").strip() or "unknown"
        if not title: continue
        if is_nobel_related(title, text) and is_year_2025(title, date, text):
            nobel_items.append({"date": date, "title": title, "source": domain_of(it.get("link","") or "")})

    if not timeline and not nobel_items:
        out = os.path.join(args.output_dir, "final_summary.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("No 2025 Nobel Prize–related timeline or headlines were found.")
        log.info(f"已写出：{out}"); return

    system_prompt = OVERALL_PROMPT
    blocks = []

    if timeline:
        blocks.append("TIMELINE (authoritative):\n" + json.dumps(timeline, ensure_ascii=False, indent=2))

    if args.use_headlines and nobel_items:
        items_blob = json.dumps({"items": sorted(nobel_items, key=lambda x: x['date'])}, ensure_ascii=False)
        blocks.append("HEADLINES (secondary evidence):\n" + items_blob)

    user_msg = (
        "Use the timeline as ground truth. If any conflict arises, prefer the timeline.\n\n"
        + ("\n\n".join(blocks) if blocks else "No timeline provided; rely on headlines without fabrication.")
        + "\n\nRemember: Output EXACTLY THREE PARAGRAPHS separated by a blank line. "
          "Do not list dates or outlets; write flowing prose."
    )

    session = make_session()
    limiter = AdaptiveLimiter(qps=args.qps)

    try:
        raw = call_glm(session, limiter, args.api_key or os.getenv("ZHIPU_API_KEY",""),
                       [{"role":"system","content":system_prompt},
                        {"role":"user","content":user_msg}],
                       model=args.model, temperature=0.26, max_tokens=2400, timeout=120)
    except Exception as e:
        log.warning(f"生成失败：{e}")
        raw = ""

    if is_refusal(raw):
        final = ("could not generate")
    else:
        paras = [p for p in re.split(r'\n\s*\n', raw.strip()) if p.strip()]
        final = ("\n\n".join(re.sub(r'\s+',' ', p).strip() for p in paras[:3])
                 if len(paras) >= 3 else refactor_to_three_paragraphs(raw))

    out_path = os.path.join(args.output_dir, "final_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final)
    log.info(f"已写出：{out_path}\n完成。")

if __name__ == "__main__":
    main()
