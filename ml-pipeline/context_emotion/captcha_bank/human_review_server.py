#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from context_emotion.captcha_bank.common import EMOTIONS, read_csv

_HERE = Path(__file__).resolve().parent

EMOTION_KO = {
    "happiness": "행복",
    "calm": "평온",
    "anticipation": "기대",
    "affection": "애정",
    "anger": "분노",
    "fear": "두려움",
    "sadness": "슬픔",
    "disconnection": "불편/힘듦/지침",
    "suffering": "고통",
    "aversion": "혐오/거부감",
    "embarrassment": "당혹/수치심",
    "confidence": "자신감",
    "confusion": "혼란",
    "yearning": "그리움/갈망",
}


def emotion_label(value: str) -> str:
    return f"{EMOTION_KO.get(value, value)} ({value})"


def parse_json_list(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def format_predictions(text: str) -> str:
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        return f"<pre>{html.escape(text)}</pre>"
    labels = {"qwen25_vl_3b": "Qwen", "smolvlm2_2b": "SmolVLM"}
    lines: list[str] = []
    for key, value in values.items():
        emotion = str(value.get("emotion", ""))
        confidence = value.get("confidence", "")
        ambiguous = "예" if value.get("ambiguous") else "아니오"
        evidence = str(value.get("evidence", "")).strip()
        lines.append(f"{labels.get(key, key)}: {emotion_label(emotion)} / 확신도 {confidence} / 애매함 {ambiguous}")
        if evidence:
            lines.append(f"근거: {evidence}")
    return f"<pre>{html.escape(chr(10).join(lines))}</pre>"


def load_completed_reviews(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = value.get("sample_id")
            if not sid:
                continue
            sid = str(sid)
            if value.get("human_decision") == "__undo__":
                completed.pop(sid, None)
            else:
                completed[sid] = value
    return completed


def last_completed_sample(path: Path, reviewer_id: str | None = None) -> str | None:
    if not path.exists():
        return None
    completed = load_completed_reviews(path)
    if not completed:
        return None
    with path.open(encoding="utf-8") as f:
        lines = list(f)
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = value.get("sample_id")
        if not sid or value.get("human_decision") == "__undo__" or str(sid) not in completed:
            continue
        if reviewer_id is not None and value.get("reviewer_id") != reviewer_id:
            continue
        return str(sid)
    return None


def parse_cookies(text: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in (text or "").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        cookies[key.strip()] = value.strip()
    return cookies


def main() -> None:
    ap = argparse.ArgumentParser(description="AGAMI 감정 CAPTCHA 인간 검수 UI 서버")
    ap.add_argument("--queue", type=Path, default=_HERE / "artifacts" / "human_review_queue.csv")
    ap.add_argument("--data-root", type=Path, default=Path("/workspace/data/context_emotion"),
                    help="이미지 파일 루트 디렉터리")
    ap.add_argument("--reviews", type=Path, default=_HERE / "artifacts" / "human_reviews.jsonl")
    ap.add_argument("--claims", type=Path, default=None)
    ap.add_argument("--claim-ttl-seconds", type=int, default=1800)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.claims is None:
        a.claims = a.reviews.with_name("human_review_claims.jsonl")

    rows, _ = read_csv(a.queue)
    by_id = {r["sample_id"]: r for r in rows}
    a.reviews.parent.mkdir(parents=True, exist_ok=True)
    a.claims.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_reviews(a.reviews)
    claims: dict[str, dict] = {}
    if a.claims.exists():
        with a.claims.open(encoding="utf-8") as f:
            for line in f:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = str(value.get("sample_id", ""))
                if sid and sid not in completed:
                    claims[sid] = value
    state_lock = threading.Lock()

    def prune_claims(now: float | None = None) -> None:
        now = now or time.time()
        expired = [
            sid
            for sid, claim in claims.items()
            if sid in completed or now - float(claim.get("claimed_at", 0)) > a.claim_ttl_seconds
        ]
        for sid in expired:
            claims.pop(sid, None)

    def append_claim(sid: str, reviewer_id: str) -> None:
        value = {"sample_id": sid, "reviewer_id": reviewer_id, "claimed_at": time.time()}
        claims[sid] = value
        with a.claims.open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")
            f.flush()

    def reviewer_id_from_request(handler: BaseHTTPRequestHandler) -> tuple[str, bool]:
        cookies = parse_cookies(handler.headers.get("Cookie", ""))
        reviewer_id = cookies.get("agami_reviewer_id", "")
        if reviewer_id:
            return reviewer_id, False
        return secrets.token_urlsafe(12), True

    def assign_row(reviewer_id: str, requested_id: str) -> dict | None:
        now = time.time()
        prune_claims(now)
        if requested_id and requested_id in by_id:
            if requested_id not in completed:
                claim = claims.get(requested_id)
                if not claim:
                    append_claim(requested_id, reviewer_id)
                elif claim.get("reviewer_id") != reviewer_id:
                    return None
            elif completed.get(requested_id, {}).get("reviewer_id") not in (None, "", reviewer_id):
                return None
            return by_id[requested_id]
        for sid, claim in claims.items():
            if claim.get("reviewer_id") == reviewer_id and sid not in completed:
                claim["claimed_at"] = now
                return by_id.get(sid)
        for row in rows:
            sid = row["sample_id"]
            claim = claims.get(sid)
            if sid in completed:
                continue
            if claim and claim.get("reviewer_id") != reviewer_id:
                continue
            if not claim:
                append_claim(sid, reviewer_id)
            return row
        return None

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status, body, ctype="text/html; charset=utf-8", headers=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                sid = parse_qs(parsed.query).get("id", [""])[0]
                row = by_id.get(sid)
                if not row:
                    return self.send_body(404, "missing")
                root = a.data_root.resolve()
                path = (root / row["image_path"]).resolve()
                if root not in path.parents or not path.exists():
                    return self.send_body(404, "missing")
                ctype = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
                return self.send_body(200, path.read_bytes(), ctype)

            reviewer_id, is_new_reviewer = reviewer_id_from_request(self)
            response_headers = {}
            if is_new_reviewer:
                response_headers["Set-Cookie"] = f"agami_reviewer_id={reviewer_id}; Path=/; SameSite=Lax"

            query = parse_qs(parsed.query)
            requested_id = query.get("id", [""])[0]
            with state_lock:
                row = assign_row(reviewer_id, requested_id)
                completed_count = len(completed)
                active_claims = len([sid for sid in claims if sid not in completed])
            if row is None:
                return self.send_body(
                    200,
                    "<h1>검수 완료</h1><p>모든 리뷰 큐 검수가 끝났습니다.</p>",
                    headers=response_headers,
                )

            is_edit = row["sample_id"] in completed
            sid = html.escape(row["sample_id"])
            preds = format_predictions(row["model_predictions"])
            provisional = emotion_label(row["provisional_emotion"])
            aux_saved = parse_json_list(
                completed.get(row["sample_id"], {}).get("human_aux_emotions", "[]")
            )
            primary_buttons = "".join(
                f'<button class="emotion" type="submit" name="decision" value="relabel:{e}">'
                f"{emotion_label(e)}</button>"
                for e in EMOTIONS
            )
            aux_checkboxes = "".join(
                f'<label class="aux-item"><input type="checkbox" name="aux_emotion" value="{e}"'
                f'{" checked" if e in aux_saved else ""}> {emotion_label(e)}</label>'
                for e in EMOTIONS
            )
            edit_banner = (
                '<p style="padding:8px;background:#fff2cc;border:1px solid #d6b656;border-radius:8px">'
                "<b>수정 모드:</b> 이 문제를 다시 저장하면 이전 판정을 덮어씁니다.</p>"
                if is_edit else ""
            )
            body = f"""<!doctype html><meta charset="utf-8"><title>AGAMI 사람 검수</title>
<style>
body{{font-family:sans-serif;max-width:1200px;margin:0 auto;padding:14px;background:#f7f7f7;color:#111}}
.top{{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;padding:6px 0 12px;border-bottom:1px solid #ddd;position:sticky;top:0;background:#f7f7f7;z-index:1}}
.progress{{font-weight:700}}
.badge{{display:inline-block;padding:4px 10px;border-radius:999px;background:#e9eefc;border:1px solid #cdd7ff;margin-right:6px}}
.main{{display:grid;grid-template-columns:minmax(420px,1fr) 420px;gap:18px;margin-top:14px}}
img{{max-width:100%;max-height:760px;background:#fff;border:1px solid #ddd}}
.panel{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}}
.actions{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin:10px 0}}
button{{font-size:15px;font-weight:700;border:1px solid #999;border-radius:8px;padding:12px;cursor:pointer;background:#f2f2f2}}
button:hover{{background:#e6e6e6}}
.accept{{background:#dff5e6}}.bad{{background:#fde2e2}}.amb{{background:#fff2cc}}
.emotions{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}
.emotion{{font-weight:600;background:#eef3ff}}
.aux-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}}
.aux-item{{display:block;padding:8px 10px;border:1px solid #ddd;border-radius:8px;background:#fafafa}}
textarea{{width:100%;box-sizing:border-box;margin-top:8px}}
pre{{white-space:pre-wrap;font-size:12px;margin:0}}
.details{{margin-top:12px;padding:10px;border:1px dashed #c9c9c9;border-radius:8px;background:#fafafa}}
summary{{cursor:pointer;font-weight:700}}
.small{{color:#555;font-size:13px}}
@media(max-width:900px){{.main{{grid-template-columns:1fr}}.actions,.emotions,.aux-grid{{grid-template-columns:1fr}}}}
</style>
<div class="top"><div><span class="badge">팀 검수</span><span class="badge">중복 배정 방지</span><span class="badge">대표+보조 감정</span></div><div class="progress">{completed_count} / {len(rows)} 완료 · 작업 중 {active_claims}</div></div>
<div class="main"><div><img src="/image?id={quote(row['sample_id'])}"></div><div class="panel"><p><b>문제 ID:</b> {sid}</p><p><b>현재 임시 정답:</b> <strong>{provisional}</strong> · <b>공격 난이도:</b> {row['attack_hardness']}</p>{edit_banner}
<form method="post" action="/undo"><button type="submit">이전 판정 되돌리기</button></form>
<form method="post" action="/review"><input type="hidden" name="sample_id" value="{sid}">
<div class="actions"><button class="accept" type="submit" name="decision" value="accept">맞음: 임시 정답 승인</button><button class="amb" type="submit" name="decision" value="exclude_ambiguous">애매함: 제외</button><button class="bad" type="submit" name="decision" value="exclude_invalid">이미지/대상 문제: 제외</button></div>
<p><b>대표 정답을 하나만 선택:</b></p><div class="emotions">{primary_buttons}</div>
<p><b>보조 감정(복수 선택 가능):</b></p><div class="aux-grid">{aux_checkboxes}</div>
<p>사람 판단 확신도: <select name="confidence"><option value="high">높음</option><option value="medium">보통</option><option value="low">낮음</option></select></p>
<textarea name="note" rows="3" placeholder="선택 사항: 판단 근거 또는 제외 사유"></textarea>
<details class="details"><summary>상세 정보 보기</summary><div class="small"><p><b>공격 모델 판정</b></p>{preds}</div></details>
<p><button type="submit">저장 후 다음</button></p></form></div></div>"""
            self.send_body(200, body, headers=response_headers)

        def do_POST(self):
            if self.path == "/undo":
                reviewer_id, _ = reviewer_id_from_request(self)
                with state_lock:
                    sid = last_completed_sample(a.reviews, reviewer_id)
                if not sid:
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
                value = {
                    "sample_id": sid,
                    "human_decision": "__undo__",
                    "human_emotion": "",
                    "human_aux_emotions": "[]",
                    "human_confidence": "",
                    "human_note": "undo own last review",
                    "reviewer_id": reviewer_id,
                }
                with state_lock:
                    with a.reviews.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(value, ensure_ascii=False) + "\n")
                        f.flush()
                    completed.pop(sid, None)
                    claims.pop(sid, None)
                self.send_response(303)
                self.send_header("Location", f"/?id={quote(sid)}")
                self.end_headers()
                return

            if self.path != "/review":
                return self.send_body(404, "missing")

            reviewer_id, _ = reviewer_id_from_request(self)
            form = parse_qs(
                self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
            )
            sid = form.get("sample_id", [""])[0]
            if sid not in by_id:
                return self.send_body(400, "bad sample")
            with state_lock:
                existing = completed.get(sid)
                if existing and existing.get("reviewer_id") not in (None, "", reviewer_id):
                    claims.pop(sid, None)
                    self.send_response(303)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
            decision = form.get("decision", [""])[0]
            emotion = form.get("emotion", [""])[0]
            aux_emotions = [e for e in form.get("aux_emotion", []) if e in EMOTIONS]
            if decision.startswith("relabel:"):
                emotion = decision.split(":", 1)[1]
                decision = "relabel"
            if decision == "accept":
                emotion = by_id[sid]["provisional_emotion"]
            if decision == "relabel" and emotion not in EMOTIONS:
                return self.send_body(400, "relabel requires emotion")
            aux_emotions = [e for e in dict.fromkeys(aux_emotions) if e != emotion]
            value = {
                "sample_id": sid,
                "human_decision": decision,
                "human_emotion": emotion,
                "human_aux_emotions": json.dumps(aux_emotions, ensure_ascii=False),
                "human_confidence": form.get("confidence", [""])[0],
                "human_note": form.get("note", [""])[0],
                "reviewer_id": reviewer_id,
            }
            with state_lock:
                with a.reviews.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(value, ensure_ascii=False) + "\n")
                    f.flush()
                completed[sid] = value
                claims.pop(sid, None)
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def log_message(self, *_):
            pass

    print(
        f"review UI: http://{a.host}:{a.port}  queue={len(rows)} completed={len(completed)}",
        flush=True,
    )
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
