import re
import os
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
import random

# a small skills lexicon for demo; expand as needed
COMMON_SKILLS = [
    "python","java","c++","c#","sql","nosql","mongodb","postgresql","mysql",
    "aws","azure","gcp","docker","kubernetes","git","linux","tensorflow",
    "pytorch","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "nlp","computer vision","machine learning","deep learning","react",
    "javascript","node.js","excel","tableau","powerbi","spark","hadoop",
    "rest api","microservices","ci/cd"
]

def normalize_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t.strip().lower())

def extract_skills_from_text(text: str, skills_list=None, fuzz_threshold=80):
    """
    Return set of skills found in text by exact or fuzzy matching.
    fuzz_threshold: 0-100 (higher is stricter)
    """
    if skills_list is None:
        skills_list = COMMON_SKILLS
    found = set()
    t = normalize_text(text)
    for skill in skills_list:
        skill_norm = skill.lower()
        if skill_norm in t:
            found.add(skill)
        else:
            # fuzzy check on words around occurrences
            # check fuzzy ratio between skill and any sliding token phrase
            ratio = fuzz.partial_ratio(skill_norm, t)
            if ratio >= fuzz_threshold:
                found.add(skill)
    return found

def parse_jd(jd_text: str):
    """
    Try to extract:
      - title (first non-empty line)
      - required skills (after 'must have'/'required' patterns)
      - optional skills (after 'good to have' patterns)
    Fallback: find skills by scanning the whole JD.
    """
    t = jd_text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0] if lines else "Job"
    # patterns
    lower = t.lower()
    required = set()
    optional = set()
    # find phrases after 'must have' or 'required'
    m = re.search(r'(must have|must-have|required|requirements:)(.*?)(?:\n|$)', lower, flags=re.I | re.S)
    if m:
        required_text = m.group(2)
        required |= extract_skills_from_text(required_text)
    # good to have
    g = re.search(r'(good to have|nice to have|preferred)(.*?)(?:\n|$)', lower, flags=re.I | re.S)
    if g:
        optional_text = g.group(2)
        optional |= extract_skills_from_text(optional_text)
    # fallback: if none found, scan entire JD
    if not required:
        required |= extract_skills_from_text(lower)
    # Heuristic: optional = required ∩ (COMMON_SKILLS) minus a few top ones
    optional = optional - required
    return {"title": title, "required_skills": sorted(required), "optional_skills": sorted(optional)}

def tfidf_similarity(text1: str, text2: str) -> float:
    vec = TfidfVectorizer(stop_words="english").fit_transform([text1, text2])
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(sim)

def compute_scores(resume_text: str, jd_text: str,
                   hard_weight=0.6, soft_weight=0.4,
                   fuzz_threshold=80):
    """
    Returns a dict:
      hard_score (0-100), soft_score (0-100), final_score (0-100), missing_skills, matched_skills, verdict
    """
    resume_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)
    parsed = parse_jd(jd_text)
    required = set(parsed["required_skills"])
    optional = set(parsed["optional_skills"])

    # Hard match: fraction of required skills present (use fuzzy matching)
    matched = set()
    for skill in required:
        # check fuzzy if skill appears in resume
        r = fuzz.partial_ratio(skill.lower(), resume_norm)
        if r >= fuzz_threshold or skill.lower() in resume_norm:
            matched.add(skill)
    # Required-skill coverage
    hard_score = 0.0
    if required:
        hard_score = 100.0 * (len(matched) / len(required))
    else:
        # if no required listed, treat hard score as 100 if some overlap found
        # compute overlap with common skills
        s = extract_skills_from_text(resume_text)
        hard_score = 100.0 if s else 0.0

    # Soft match: semantic similarity via TF-IDF
    try:
        sim = tfidf_similarity(resume_text, jd_text)
        soft_score = sim * 100.0
    except Exception:
        soft_score = 0.0

    # weighted
    final = (hard_weight * hard_score + soft_weight * soft_score)
    final_score = max(0, min(100, round(final)))

    # Verdict thresholds
    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"

    missing = sorted(list(required - matched))
    return {
        "title": parsed["title"],
        "required_skills": sorted(list(required)),
        "optional_skills": sorted(list(optional)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": missing,
        "hard_score": round(hard_score, 2),
        "soft_score": round(soft_score, 2),
        "final_score": final_score,
        "verdict": verdict
    }

def generate_suggestions(resume_text: str, missing_skills: list, job_title: str, openai_api_key=None):
    """
    If openai_api_key is provided (and openai installed), call OpenAI to generate a short suggestion blurb.
    Otherwise return templated suggestions.
    """
    if openai_api_key:
        try:
            import openai
            openai.api_key = openai_api_key
            prompt = (
                f"You are a helpful career coach. A candidate is applying for '{job_title}'. "
                f"They are missing these skills: {', '.join(missing_skills) or 'none'}. "
                "Given the resume text (kept short), provide 3 concise, actionable improvements the candidate can make "
                "— focus on projects, certificates, and resume phrasing (3 bullets). Resume text:\n\n"
                f"{resume_text[:2000]}"
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # note: if not available in their account, fallback below
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception:
            # fallback: templated suggestions
            pass

    # templated suggestions
    bullets = []
    if missing_skills:
        top = missing_skills[:5]
        bullets.append("Consider adding short projects or a 'Key Skills' section that shows hands-on use of: " + ", ".join(top) + ".")
        bullets.append("If possible, complete a short online certificate or mini-project (1–4 weeks) demonstrating one of the missing skills, and add it under 'Projects'.")
    else:
        bullets.append("Resume already lists key skills. Emphasize measurable outcomes (numbers) for each project/role (e.g. reduced latency by 30%).")
        bullets.append("Add links to GitHub or portfolio for quick verification.")
    bullets.append("Ensure top 3 skills required by the JD are on the first page/at the top under Summary/Skills.")

    return "\n".join("- " + b for b in bullets)
# scoring.py
import re
import os
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
import random

# a small skills lexicon for demo; expand as needed
COMMON_SKILLS = [
    "python","java","c++","c#","sql","nosql","mongodb","postgresql","mysql",
    "aws","azure","gcp","docker","kubernetes","git","linux","tensorflow",
    "pytorch","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "nlp","computer vision","machine learning","deep learning","react",
    "javascript","node.js","excel","tableau","powerbi","spark","hadoop",
    "rest api","microservices","ci/cd"
]

def normalize_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t.strip().lower())

def extract_skills_from_text(text: str, skills_list=None, fuzz_threshold=80):
    """
    Return set of skills found in text by exact or fuzzy matching.
    fuzz_threshold: 0-100 (higher is stricter)
    """
    if skills_list is None:
        skills_list = COMMON_SKILLS
    found = set()
    t = normalize_text(text)
    for skill in skills_list:
        skill_norm = skill.lower()
        if skill_norm in t:
            found.add(skill)
        else:
            # fuzzy check on words around occurrences
            # check fuzzy ratio between skill and any sliding token phrase
            ratio = fuzz.partial_ratio(skill_norm, t)
            if ratio >= fuzz_threshold:
                found.add(skill)
    return found

def parse_jd(jd_text: str):
    """
    Try to extract:
      - title (first non-empty line)
      - required skills (after 'must have'/'required' patterns)
      - optional skills (after 'good to have' patterns)
    Fallback: find skills by scanning the whole JD.
    """
    t = jd_text or ""
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0] if lines else "Job"
    # patterns
    lower = t.lower()
    required = set()
    optional = set()
    # find phrases after 'must have' or 'required'
    m = re.search(r'(must have|must-have|required|requirements:)(.*?)(?:\n|$)', lower, flags=re.I | re.S)
    if m:
        required_text = m.group(2)
        required |= extract_skills_from_text(required_text)
    # good to have
    g = re.search(r'(good to have|nice to have|preferred)(.*?)(?:\n|$)', lower, flags=re.I | re.S)
    if g:
        optional_text = g.group(2)
        optional |= extract_skills_from_text(optional_text)
    # fallback: if none found, scan entire JD
    if not required:
        required |= extract_skills_from_text(lower)
    # Heuristic: optional = required ∩ (COMMON_SKILLS) minus a few top ones
    optional = optional - required
    return {"title": title, "required_skills": sorted(required), "optional_skills": sorted(optional)}

def tfidf_similarity(text1: str, text2: str) -> float:
    vec = TfidfVectorizer(stop_words="english").fit_transform([text1, text2])
    sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(sim)

def compute_scores(resume_text: str, jd_text: str,
                   hard_weight=0.6, soft_weight=0.4,
                   fuzz_threshold=80):
    """
    Returns a dict:
      hard_score (0-100), soft_score (0-100), final_score (0-100), missing_skills, matched_skills, verdict
    """
    resume_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)
    parsed = parse_jd(jd_text)
    required = set(parsed["required_skills"])
    optional = set(parsed["optional_skills"])

    # Hard match: fraction of required skills present (use fuzzy matching)
    matched = set()
    for skill in required:
        # check fuzzy if skill appears in resume
        r = fuzz.partial_ratio(skill.lower(), resume_norm)
        if r >= fuzz_threshold or skill.lower() in resume_norm:
            matched.add(skill)
    # Required-skill coverage
    hard_score = 0.0
    if required:
        hard_score = 100.0 * (len(matched) / len(required))
    else:
        # if no required listed, treat hard score as 100 if some overlap found
        # compute overlap with common skills
        s = extract_skills_from_text(resume_text)
        hard_score = 100.0 if s else 0.0

    # Soft match: semantic similarity via TF-IDF
    try:
        sim = tfidf_similarity(resume_text, jd_text)
        soft_score = sim * 100.0
    except Exception:
        soft_score = 0.0

    # weighted
    final = (hard_weight * hard_score + soft_weight * soft_score)
    final_score = max(0, min(100, round(final)))

    # Verdict thresholds
    if final_score >= 75:
        verdict = "High"
    elif final_score >= 50:
        verdict = "Medium"
    else:
        verdict = "Low"

    missing = sorted(list(required - matched))
    return {
        "title": parsed["title"],
        "required_skills": sorted(list(required)),
        "optional_skills": sorted(list(optional)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": missing,
        "hard_score": round(hard_score, 2),
        "soft_score": round(soft_score, 2),
        "final_score": final_score,
        "verdict": verdict
    }

def generate_suggestions(resume_text: str, missing_skills: list, job_title: str, openai_api_key=None):
    """
    If openai_api_key is provided (and openai installed), call OpenAI to generate a short suggestion blurb.
    Otherwise return templated suggestions.
    """
    if openai_api_key:
        try:
            import openai
            openai.api_key = openai_api_key
            prompt = (
                f"You are a helpful career coach. A candidate is applying for '{job_title}'. "
                f"They are missing these skills: {', '.join(missing_skills) or 'none'}. "
                "Given the resume text (kept short), provide 3 concise, actionable improvements the candidate can make "
                "— focus on projects, certificates, and resume phrasing (3 bullets). Resume text:\n\n"
                f"{resume_text[:2000]}"
            )
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # note: if not available in their account, fallback below
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception:
            # fallback: templated suggestions
            pass

    # templated suggestions
    bullets = []
    if missing_skills:
        top = missing_skills[:5]
        bullets.append("Consider adding short projects or a 'Key Skills' section that shows hands-on use of: " + ", ".join(top) + ".")
        bullets.append("If possible, complete a short online certificate or mini-project (1–4 weeks) demonstrating one of the missing skills, and add it under 'Projects'.")
    else:
        bullets.append("Resume already lists key skills. Emphasize measurable outcomes (numbers) for each project/role (e.g. reduced latency by 30%).")
        bullets.append("Add links to GitHub or portfolio for quick verification.")
    bullets.append("Ensure top 3 skills required by the JD are on the first page/at the top under Summary/Skills.")

    return "\n".join("- " + b for b in bullets)
