import os
import re

import requests


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {hit['source']}]\n{hit['text']}" for hit in hits
    )
    return (
        "Answer the question only from the supplied context. "
        "If the context does not contain the answer, say that it does not contain it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def _openai_compatible_answer(prompt: str) -> str | None:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_url = os.getenv(
        "LLM_API_URL",
        "https://api.openai.com/v1/chat/completions",
    )
    if not api_key:
        return None
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    response = requests.post(
        api_url,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _ollama_answer(prompt: str) -> str | None:
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.RequestException:
        return None


def _extractive_answer(question: str, hits: list[dict]) -> str:
    question_lower = question.lower()
    context = "\n".join(hit["text"] for hit in hits)

    if "skill" in question_lower:
        skills_match = re.search(
            r"SKILLS\s*\n(.*?)(?=\nPROJECTS\s*\n|\n[A-Z][A-Z /&]{2,}\s*\n|\Z)",
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if skills_match:
            lines = [
                line.strip().strip("\u2022")
                for line in skills_match.group(1).splitlines()
                if line.strip()
            ]
            lines = [
                line for line in lines
                if any(label in line.lower() for label in ("programming", "ai/", "tools"))
            ]
            if lines:
                return "\n".join(lines)

    if "who is" in question_lower or "about nandushree" in question_lower:
        profile_match = re.search(
            r"PROFILE\s*\n(.*?)(?=\nEDUCATION\s*\n|\n[A-Z][A-Z /&]{2,}\s*\n)",
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if profile_match:
            return " ".join(profile_match.group(1).split())

    words = [
        word.lower()
        for word in re.findall(r"[a-z0-9]+", question)
        if len(word) > 3
    ]
    candidates = []
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?])\s+|(?<=\n)", hit["text"]):
            sentence = sentence.replace("\n", " ").strip()
            if not sentence:
                continue
            sentence_lower = sentence.lower()
            score = sum(
                (
                    sentence_lower.count(word)
                    or sentence_lower.count(word[:-1] if word.endswith("s") else word)
                ) * 2
                for word in words
            )
            matched_words = sum(
                1 for word in words
                if word in sentence_lower
                or (word.endswith("s") and word[:-1] in sentence_lower)
            )
            score += matched_words * matched_words
            if any(marker in sentence_lower for marker in (" is ", " are ", "defined", "means")):
                score += 1
            candidates.append((score, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates and candidates[0][0] > 0:
        return candidates[0][1]
    return "The supplied documents do not contain a clear answer to this question."


def generate(question: str, hits: list[dict] | None = None) -> str:
    prompt = build_prompt(question, hits or [])
    return (
        _openai_compatible_answer(prompt)
        or _ollama_answer(prompt)
        or _extractive_answer(question, hits or [])
    )
