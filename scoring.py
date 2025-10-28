# scoring.py
from __future__ import annotations
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple

# =========================
# Utilities
# =========================

HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(x: str) -> str:
    return HTML_TAG_RE.sub(" ", x or "")


# =========================
# Optional vector backends (robust fallbacks)
# =========================

# TF-IDF fallback
_HAS_SK = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine

    _HAS_SK = True
except Exception:
    _HAS_SK = False

# Sentence-Transformers embeddings (guard with NumPy availability)
_USE_EMBEDDINGS = True
_HAS_ST = False
try:
    import numpy as _np  # noqa: F401
except Exception:
    _USE_EMBEDDINGS = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _HAS_ST = True
except Exception:
    _HAS_ST = False
    _USE_EMBEDDINGS = False

_EMBEDDER = None


def _get_embedder() -> "SentenceTransformer | None":
    """
    Load & cache the embeddings model (CPU). Controlled by SENTENCE_MODEL env.
    Defaults to 'all-MiniLM-L6-v2' (small & fast). Returns None if unavailable.
    """
    global _EMBEDDER
    if not _USE_EMBEDDINGS or not _HAS_ST:
        return None
    if _EMBEDDER is not None:
        return _EMBEDDER

    model_name = os.getenv("SENTENCE_MODEL") or "all-MiniLM-L6-v2"
    try:
        _EMBEDDER = SentenceTransformer(model_name, device="cpu")
        return _EMBEDDER
    except Exception:
        # try a safer small fallback once
        if model_name != "all-MiniLM-L6-v2":
            try:
                _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                return _EMBEDDER
            except Exception:
                pass
        _EMBEDDER = None
        return None


def _cosine_dense(a, b) -> float:
    # sentence-transformers util import here to avoid hard fail at module import
    from sentence_transformers import util as st_util  # type: ignore
    return float(st_util.cos_sim(a, b).item())


def _cosine_tfidf(t1: str, t2: str) -> float:
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z0-9\+\#\.\-]+",
        min_df=1,
        max_df=0.9,
        lowercase=True,
    )
    X = vec.fit_transform([t1, t2])
    return float(_sk_cosine(X[0], X[1])[0, 0])


# ---- fuzzy matching (optional) ----
_HAS_RAPIDFUZZ = False
try:
    from rapidfuzz.fuzz import partial_ratio as _fz_partial_ratio  # type: ignore

    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False


# =========================
# Field builders
# =========================

def _job_fields(job: Dict[str, Any]) -> Dict[str, str]:
    company = job.get("company") or {}
    industry = (company.get("industryType") or {}).get("name", "")
    area = (job.get("area") or {}).get("name", "")
    return {
        "title": job.get("title", ""),
        "industry": industry,
        "area": area,
        "requirements": _strip_html(job.get("jobRequirements", "")),
        "duties": _strip_html(job.get("dutiesAndResponsibilities", "")),
        "summary": _strip_html(job.get("roleSummary", "")),
    }


def _job_core_text(job: Dict[str, Any]) -> str:
    """Compact job text for cross-factor comparisons (experience/cert relevance)."""
    company = job.get("company") or {}
    industry = (company.get("industryType") or {}).get("name", "")
    parts = [
        job.get("title", ""),
        industry,
        (job.get("area") or {}).get("name", ""),
        _strip_html(job.get("jobRequirements", "")),
        _strip_html(job.get("dutiesAndResponsibilities", "")),
        _strip_html(job.get("roleSummary", "")),
    ]
    return " ".join(p for p in parts if p)


def _applicant_fields(app: Dict[str, Any]) -> Dict[str, str]:
    # skills (both shapes)
    skills: List[str] = []
    for s in app.get("skills", []) or []:
        if isinstance(s, dict):
            if isinstance(s.get("skill"), dict) and s["skill"].get("name"):
                skills.append(str(s["skill"]["name"]))
            elif s.get("name"):
                skills.append(str(s["name"]))

    bio = app.get("bio") or (app.get("profile") or {}).get("bio") or ""

    # functional areas: tolerate snake_case and camelCase, and two shapes
    fa: List[str] = []
    cand = app.get("functional_areas")
    if cand is None:
        cand = app.get("functionalAreas")
    for a in (cand or []):
        if isinstance(a, dict):
            if isinstance(a.get("area"), dict) and a["area"].get("name"):
                nm = a["area"]["name"]
            else:
                nm = a.get("name")
            if nm:
                fa.append(str(nm))

    # experiences
    exp_chunks: List[str] = []
    for e in app.get("experiences", []) or []:
        if not isinstance(e, dict):
            continue
        exp_chunks.append(
            " ".join(
                [
                    str(e.get("title") or ""),
                    str(e.get("companyName") or ""),
                    str(e.get("description") or ""),
                ]
            ).strip()
        )

    # education
    edu_chunks: List[str] = []
    for ed in app.get("educations", []) or []:
        if not isinstance(ed, dict):
            continue
        edu_chunks.append(
            " ".join(
                [
                    str(ed.get("level") or ""),
                    str(ed.get("fieldOfStudy") or ""),
                    str(ed.get("instituteName") or ""),
                ]
            ).strip()
        )

    # certs/awards
    certs = [
        c.get("title")
        for c in (app.get("certificates") or [])
        if isinstance(c, dict) and c.get("title")
    ]
    awards = [
        a.get("title")
        for a in (app.get("awards") or [])
        if isinstance(a, dict) and a.get("title")
    ]

    return {
        "skills": ", ".join(skills),
        "bio": bio,
        "functional_areas": ", ".join(fa),
        "experiences": ". ".join(exp_chunks),
        "education": ". ".join(edu_chunks),
        "certs_awards": ", ".join([*certs, *awards]),
    }


# =========================
# Semantic similarity (field-wise max)
# =========================

# Weights (tune here)
_JOB_WEIGHTS = {
    "title": 3.0,
    "requirements": 4.0,  # emphasize explicit requirements
    "duties": 1.0,        # de-emphasize generic prose
    "summary": 0.5,       # de-emphasize generic prose
    "industry": 1.0,
    "area": 1.0,
}
_APP_FIELDS_ORDER = ["skills", "experiences", "education", "functional_areas", "certs_awards", "bio"]


def _best_sim(job_txts: List[str], app_txts: List[str]) -> float:
    """Return the max similarity between any job text and any applicant text (0..1)."""
    # Embeddings path
    embedder = _get_embedder()
    if embedder is not None:
        try:
            ej = embedder.encode(job_txts, normalize_embeddings=True)
            ea = embedder.encode(app_txts, normalize_embeddings=True)
            sims: List[float] = []
            for i in range(len(job_txts)):
                row_max = max(_cosine_dense(ej[i], ea[j]) for j in range(len(app_txts))) if app_txts else 0.0
                sims.append(max(0.0, min(1.0, float(row_max))))
            return sum(sims) / max(1, len(sims))
        except Exception:
            pass  # fall through to TF-IDF

    # TF-IDF fallback
    if _HAS_SK:
        sims: List[float] = []
        for jtxt in job_txts:
            candidates = [jtxt] + app_txts
            try:
                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    token_pattern=r"[A-Za-z0-9\+\#\.\-]+",
                    min_df=1,
                    max_df=0.9,
                    lowercase=True,
                )
                X = vec.fit_transform(candidates)
                jv = X[0]
                av = X[1:]
                row = _sk_cosine(jv, av)[0]
                sims.append(float(max(row)) if row.size else 0.0)
            except Exception:
                sims.append(0.0)
        return sum(sims) / max(1, len(sims))

    # Absolute fallback (no vector backend)
    return 0.0


def _weighted_semantic_similarity(job: Dict[str, str], app: Dict[str, str]) -> Tuple[float, str]:
    """
    For each non-empty job field, compute similarity vs ALL applicant sub-fields and take the MAX.
    Aggregate with job field weights.
    """
    j_pairs = [(k, v.strip()) for k, v in job.items() if v and str(v).strip()]
    a_pairs = [(k, v.strip()) for k, v in app.items() if v and str(v).strip()]

    if not j_pairs:
        has_app = bool(a_pairs)
        return (0.7 if has_app else 0.0), "No explicit job text"

    # preserve applicant field priority order (skills > experiences > ...)
    a_texts = [t for k, t in a_pairs if k in _APP_FIELDS_ORDER]
    # if nothing matched, just take all
    if not a_texts:
        a_texts = [t for _, t in a_pairs]

    details: List[str] = []
    num = 0.0
    den = 0.0
    for jf, jtxt in j_pairs:
        jw = _JOB_WEIGHTS.get(jf, 1.0)
        field_sim = _best_sim([jtxt], a_texts) if a_texts else 0.0
        num += field_sim * jw
        den += jw
        details.append(f"{jf}:{field_sim:.2f}")

    ratio = num / max(1e-9, den)
    ratio = max(0.0, min(1.0, ratio))
    return ratio, "Embeddings cosine (field-max): " + " + ".join(details)


# =========================
# Other factors
# =========================

def _years_of_experience(applicant: Dict[str, Any]) -> float:
    exps = applicant.get("experiences", []) or []
    total_days = 0
    for e in exps:
        if not isinstance(e, dict):
            continue
        sd = e.get("startDate")
        ed = e.get("endDate") or datetime.utcnow().isoformat()
        try:
            from_iso = lambda s: datetime.fromisoformat(str(s).replace("Z", ""))
            sd_dt = from_iso(sd) if isinstance(sd, str) else None
            ed_dt = from_iso(ed) if isinstance(ed, str) else None
            if sd_dt and ed_dt and ed_dt >= sd_dt:
                total_days += (ed_dt - sd_dt).days
        except Exception:
            continue
    return round(total_days / 365.25, 2)


def _education_match(applicant: Dict[str, Any], job: Dict[str, Any]) -> Tuple[float, str]:
    required = (job.get("educationLevel") or "").lower()
    educs = applicant.get("educations") or []
    levels = [(e.get("level") or "").lower() for e in educs if isinstance(e, dict)]
    rank = {
        "high_school": 0, "high school": 0,
        "associate": 1, "diploma": 1,
        "bachelor": 2, "bachelors": 2,
        "masters": 3, "master": 3, "msc": 3, "ma": 3,
        "phd": 4, "doctorate": 4, "md": 4,
    }
    req_rank = rank.get(required, 0)
    best = max([rank.get(l, 0) for l in levels], default=0)
    if best >= req_rank and req_rank > 0:
        return 1.0, f"Meets required level ({required})."
    if req_rank == 0 and levels:
        return 0.8, "No explicit requirement; applicant has education."
    if not levels:
        return 0.2, "No education records."
    return 0.5, f"Below required level ({required})."


def _location_match(applicant: Dict[str, Any], job: Dict[str, Any]) -> Tuple[float, str]:
    job_country = (job.get("country", {}) or {}).get("code") or ""
    job_province = (job.get("province", {}) or {}).get("name") or job.get("provinceName") or ""
    prof = applicant.get("profile") or {}
    a_country = (
        (prof.get("country") or {}).get("code")
        if isinstance(prof.get("country"), dict)
        else (prof.get("country") or "")
    )
    a_province = (
        (prof.get("province") or {}).get("name")
        if isinstance(prof.get("province"), dict)
        else (prof.get("province") or "")
    )
    score = 0.0
    detail: List[str] = []
    if a_country and job_country and str(a_country).lower() == str(job_country).lower():
        score += 0.6
        detail.append("Same country")
    if a_province and job_province and str(a_province).lower() == str(job_province).lower():
        score += 0.4
        detail.append("Same province")
    if score == 0 and (job.get("company", {}) or {}).get("profile", {}).get("workPolicy") == "remote":
        score = 0.7
        detail.append("Remote-friendly role")
    return min(score, 1.0), ", ".join(detail) or "No location alignment"


def _languages_match(applicant: Dict[str, Any], job: Dict[str, Any]) -> Tuple[float, str]:
    prof = applicant.get("profile") or {}
    sl = prof.get("speakingLanguages") or prof.get("languages") or []
    if not isinstance(sl, list):
        sl = [sl]
    langs = {str(x).lower().strip() for x in sl if x}
    job_lang = (job.get("language") or "").lower().strip()

    if job_lang and job_lang != "any":
        ok = (job_lang in langs) or (job_lang == "english" and ("en" in langs or "english" in langs))
        detail = f"Job requires {job_lang}; applicant has: {', '.join(sorted(langs)) or 'none'}"
        return (1.0 if ok else 0.0), detail

    return (0.7 if langs else 0.0), (
        f"Applicant languages: {', '.join(sorted(langs))}" if langs else "No languages provided"
    )


def _functional_area_match(applicant: Dict[str, Any], job: Dict[str, Any]) -> Tuple[float, str]:
    job_area = (job.get("area") or {}).get("name", "").lower().strip()
    areas = set()

    cand = applicant.get("functional_areas")
    if cand is None:
        cand = applicant.get("functionalAreas")

    for a in (cand or []):
        if isinstance(a, dict):
            if isinstance(a.get("area"), dict) and a["area"].get("name"):
                nm = a["area"]["name"]
            else:
                nm = a.get("name")
            if nm:
                areas.add(str(nm).lower().strip())

    if not job_area and not areas:
        return 0.5, "No areas provided"
    if job_area in areas:
        return 1.0, f"Area matches ({job_area})."
    for a in areas:
        if job_area and (job_area in a or a in job_area):
            return 0.8, f"Partial area match: {a} ~ {job_area}"
    return 0.2, f"No area overlap (job: {job_area or 'n/a'}, applicant: {', '.join(sorted(areas)) or 'n/a'})"


def _awards_certs_bonus(applicant: Dict[str, Any], job: Dict[str, Any] | None = None) -> Tuple[float, str]:
    """
    Extras factor (0..1):
      - awards component: +0.5 if any awards
      - certificates component: up to +0.5, scaled by fuzzy/semantic relevance to job text
        (fallback to +0.5 flat if we can't compute relevance)
    """
    awards = applicant.get("awards") or []
    certs = applicant.get("certificates") or []

    # Awards: simple presence bonus
    awards_component = 0.5 if awards else 0.0
    details: List[str] = []
    if awards:
        details.append(f"{len(awards)} award(s)")

    # Certificates: relevance-aware up to +0.5
    cert_component = 0.0
    cert_titles = [c.get("title") for c in certs if isinstance(c, dict) and c.get("title")]
    if cert_titles:
        job_text = _job_core_text(job or {})
        if job_text.strip() and _HAS_RAPIDFUZZ:
            # best partial_ratio across all applicant certs against the job text
            best = 0
            for title in cert_titles:
                try:
                    score = _fz_partial_ratio(str(title).lower(), job_text.lower())
                    if score > best:
                        best = score
                except Exception:
                    continue
            rel = best / 100.0  # 0..1
            cert_component = 0.5 * rel
            details.append(f"{len(certs)} certificate(s), relevance {rel:.2f}")
        else:
            # fallback: flat +0.5 if any certs (keeps backward compatibility)
            cert_component = 0.5
            details.append(f"{len(certs)} certificate(s)")
    else:
        details.append("0 certificate(s)")

    total = min(1.0, awards_component + cert_component)
    return total, "; ".join(details)


# =========================
# Public API
# =========================

def score(job: Dict[str, Any], applicant: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]], str]:
    """
    Compute a 0..100 total score, factor breakdown, and verdict string.
    Factors:
      - Skill Match (semantic cosine, field-wise max)
      - Experience (+ semantic relevance bonus)
      - Education
      - Functional Area
      - Location
      - Languages
      - Awards & Certificates (relevance-aware)
    """
    weights = {
        "skills": 40.0,
        "experience": 25.0,
        "education": 10.0,
        "functional_area": 8.0,
        "location": 7.0,
        "languages": 5.0,
        "extras": 5.0,
    }

    factors: List[Dict[str, Any]] = []

    # 1) Skill Match (semantic)
    j = _job_fields(job)
    a = _applicant_fields(applicant)
    skill_ratio, skill_detail = _weighted_semantic_similarity(j, a)
    factors.append(
        {"name": "Skill Match", "score": round(skill_ratio * weights["skills"], 2), "details": skill_detail}
    )

    # 2) Experience (years) + semantic relevance bonus (up to +5 pts)
    years = _years_of_experience(applicant)
    try:
        min_exp = float(job.get("minimumExperience") or 0)
    except Exception:
        min_exp = 0.0
    try:
        max_exp = float(job.get("maximumExperience") or (min_exp if min_exp > 0 else 0.0))
    except Exception:
        max_exp = 0.0

    # Base ratio from years vs. requirement
    if min_exp == 0 and max_exp == 0:
        exp_ratio = 0.7 if years > 0 else 0.0
        exp_detail = f"{years} yrs exp; no explicit requirement"
    else:
        if years < min_exp:
            exp_ratio = max(0.0, years / max(1.0, min_exp))
            gap = round(min_exp - years, 2)
            exp_detail = f"{years} yrs vs required min {min_exp} (short by {gap} yrs)"
        else:
            if max_exp > 0:
                exp_ratio = min(1.0, years / max_exp)
                exp_detail = f"{years} yrs vs range {min_exp}-{max_exp}"
            else:
                exp_ratio = 1.0
                exp_detail = f"{years} yrs (meets/exceeds requirement)"

    base_points = exp_ratio * weights["experience"]

    # Relevance bonus (0..+5) based on semantic similarity of job text vs applicant experience text
    job_text = _job_core_text(job)
    exp_texts = []
    for e in (applicant.get("experiences") or []):
        if isinstance(e, dict):
            exp_texts.append(" ".join([
                str(e.get("title") or ""),
                str(e.get("companyName") or ""),
                str(e.get("description") or "")
            ]).strip())
    applicant_exp_text = ". ".join([t for t in exp_texts if t])

    relevance_bonus_pts = 0.0
    relevance_detail = ""
    if job_text.strip() and applicant_exp_text.strip():
        relevance = _best_sim([job_text], [applicant_exp_text])  # 0..1
        relevance_bonus_pts = round(5.0 * max(0.0, min(1.0, relevance)), 2)  # cap at +5 pts
        relevance_detail = f"; relevance bonus +{relevance_bonus_pts:.2f} (sim={relevance:.2f})"

    exp_points = min(weights["experience"], round(base_points + relevance_bonus_pts, 2))
    factors.append({
        "name": "Experience",
        "score": exp_points,
        "details": exp_detail + (relevance_detail if relevance_detail else "")
    })

    # 3) Education
    edu_ratio, edu_detail = _education_match(applicant, job)
    factors.append({"name": "Education", "score": round(edu_ratio * weights["education"], 2), "details": edu_detail})

    # 4) Functional Area
    area_ratio, area_detail = _functional_area_match(applicant, job)
    factors.append(
        {"name": "Functional Area", "score": round(area_ratio * weights["functional_area"], 2), "details": area_detail}
    )

    # 5) Location
    loc_ratio, loc_detail = _location_match(applicant, job)
    factors.append({"name": "Location", "score": round(loc_ratio * weights["location"], 2), "details": loc_detail})

    # 6) Languages
    lang_ratio, lang_detail = _languages_match(applicant, job)
    factors.append({"name": "Languages", "score": round(lang_ratio * weights["languages"], 2), "details": lang_detail})

    # 7) Awards & Certificates (relevance-aware)
    extra_ratio, extra_detail = _awards_certs_bonus(applicant, job)
    factors.append(
        {"name": "Awards & Certificates", "score": round(extra_ratio * weights["extras"], 2), "details": extra_detail}
    )

    total = round(sum(f["score"] for f in factors), 2)
    verdict = "Strong match" if total >= 75 else ("Moderate match" if total >= 55 else "Weak match")
    return total, factors, verdict
