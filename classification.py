"""Parenthood-disclosure classification using GPT-5.2.

Input: a CSV or ZIP containing id, Campaign story, and creator_bio.
Output: classification_reproduced.csv. Requires OPENAI_API_KEY.

"""

import csv
import json
import os
import re
import sys
import zipfile
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
INPUT = HERE / "kickstarter_narrative_and_bios_classified.zip"
OUTPUT = HERE / "classification_reproduced.csv"
MODEL = "gpt-5.2"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 3500
MIN_WORDS = 50
MAX_STORY_CHARS = 20000
MAX_BIO_CHARS = 8000
MAX_QUOTE_CHARS = 320

SYSTEM_PROMPT = "You are a careful research coder. Follow definitions strictly. Never guess."
PROMPT = """Task: Detect founder parenthood disclosure.

You have:
- NARRATIVE (Campaign story)
- CREATOR_BIO

PARENTHOOD (may use narrative and/or bio)
Labels:
- father = explicit "dad/father" self-ID (or bio says father).
- mother = explicit "mom/mother" self-ID (or bio says mother).
- parent_unspecified = explicit parenthood WITHOUT mom/dad wording.

IMPORTANT CLARIFICATION:
- If the narrative/bio says "my son", "my daughter", "my kids", "our children", "our daughters/sons"
  or mentions "the birth of our child/children", then parent_unspecified MUST be YES
  (unless father/mother is explicitly stated).

Mutual exclusivity:
- If father=yes OR mother=yes, then parent_unspecified MUST be no.

Evidence rules for father/mother/parent_unspecified:
- If NO: evidence_source="" and quote fields blank.
- If YES:
  * evidence_source="narrative": provide exactly ONE quote_narrative (1–2 sentences of context); quote_bio=""
  * evidence_source="bio": provide exactly ONE quote_bio (1–2 sentences of context); quote_narrative=""
  * evidence_source="both": provide one contextual quote from each.

Do NOT infer gender from names, writing style, stereotypes, or child's gender.
Do NOT count metaphors like "this project is my baby".

Manual review:
- needs_manual_review=true only for ambiguous/contradictory cases.
- review_reason="" unless needs_manual_review=true.

Return JSON matching the schema and echo the id exactly."""

LABELS = ("father", "mother", "parent_unspecified")
SOURCES = ("", "narrative", "bio", "both")


def label_schema():
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "yes": {"type": "boolean"},
            "evidence_source": {"type": "string", "enum": list(SOURCES)},
            "quote_narrative": {"type": "string"},
            "quote_bio": {"type": "string"},
        },
        "required": ["yes", "evidence_source", "quote_narrative", "quote_bio"],
    }


ITEM_PROPERTIES = {"id": {"type": "string"}, **{name: label_schema() for name in LABELS},
                   "needs_manual_review": {"type": "boolean"}, "review_reason": {"type": "string"}}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"items": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "properties": ITEM_PROPERTIES, "required": list(ITEM_PROPERTIES),
    }}},
    "required": ["items"],
}

COLUMNS = ["id", "skipped_short_narrative", "skipped_non_english"]
for name in LABELS:
    COLUMNS.extend((f"{name}_yes", f"{name}_evidence_source",
                    f"{name}_quote_narrative", f"{name}_quote_bio"))
COLUMNS.extend(("needs_manual_review", "review_reason", "error"))

CHILD_PATTERNS = [
    r"\bmy\s+(?:beloved\s+|beautiful\s+|talented\s+|young\s+|little\s+)?(son|daughter|kid|child)\b",
    r"\bour\s+(?:beloved\s+|beautiful\s+|talented\s+|young\s+|little\s+)?(sons|daughters|kids|children)\b",
    r"\bthe\s+birth\s+of\s+our\s+(son|daughter|sons|daughters|child|children)\b",
    r"\bafter\s+the\s+birth\s+of\s+our\s+(son|daughter|sons|daughters|child|children)\b",
    r"\breconnected\s+after\s+the\s+birth\s+of\s+our\s+(son|daughter|sons|daughters|child|children)\b",
]
EXCLUSIONS = (r"\bdaughter\s+company\b", r"\bdaughter\s+subsidiary\b", r"\bdaughter\s+corporation\b")


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def normalized(text):
    text = (text or "").lower().translate(str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "…": "..."}))
    return clean(text)


def expand_quote(phrase, source):
    found = re.search(re.escape(phrase.strip()), source, flags=re.IGNORECASE)
    if not found:
        return ""
    before = list(re.finditer(r"[.!?]", source[:found.start()]))
    start = before[-1].end() if before else 0
    end_mark = re.search(r"[.!?]", source[found.end():])
    end = found.end() + (end_mark.end() if end_mark else len(source) - found.end())
    passage = clean(source[start:end])
    if len(passage) < 30:
        next_mark = re.search(r"[.!?]", source[end:])
        if next_mark:
            passage = clean(source[start:end + next_mark.end()])
    return passage[:520].rstrip() + ("…" if len(passage) > 520 else "")


def validate_label(label, story, bio):
    if not label["yes"]:
        return empty_label(), ""
    source = label["evidence_source"]
    qn, qb = label["quote_narrative"], label["quote_bio"]
    if source not in SOURCES[1:]:
        return empty_label(), "invalid evidence source"
    for required, quote, text in ((source in ("narrative", "both"), qn, story),
                                  (source in ("bio", "both"), qb, bio)):
        if required != bool(quote.strip()):
            return empty_label(), "missing or unexpected quote"
        if required:
            quote = expand_quote(quote, text) if len(quote) < 30 else clean(quote)
            quote = quote[:MAX_QUOTE_CHARS].rstrip() + ("…" if len(quote) > MAX_QUOTE_CHARS else "")
            if normalized(quote) not in normalized(text):
                return empty_label(), "evidence quote not found verbatim"
            if text is story:
                qn = quote
            else:
                qb = quote
    return {"yes": True, "evidence_source": source,
            "quote_narrative": qn if source in ("narrative", "both") else "",
            "quote_bio": qb if source in ("bio", "both") else ""}, ""


def empty_label():
    return {"yes": False, "evidence_source": "", "quote_narrative": "", "quote_bio": ""}


def backstop(story, bio):
    matches = []
    for source_name, text in (("narrative", story), ("bio", bio)):
        if any(re.search(p, text, re.I) for p in EXCLUSIONS):
            continue
        phrase = next((m.group(0) for pat in CHILD_PATTERNS if (m := re.search(pat, text, re.I))), None)
        if phrase:
            matches.append((source_name, expand_quote(phrase, text)))
    if not matches:
        return empty_label()
    sources = {s: q for s, q in matches}
    return {"yes": True, "evidence_source": "both" if len(sources) == 2 else matches[0][0],
            "quote_narrative": sources.get("narrative", ""), "quote_bio": sources.get("bio", "")}


def output_record(identifier, flags=None):
    record = dict.fromkeys(COLUMNS, "")
    record["id"] = identifier
    record.update(flags or {})
    return record


def classify(client, identifier, story, bio):
    request_story = clean(story)
    request_bio = clean(bio)
    request_story = request_story[:MAX_STORY_CHARS] + (" [TRUNCATED]" if len(request_story) > MAX_STORY_CHARS else "")
    request_bio = request_bio[:MAX_BIO_CHARS] + (" [TRUNCATED]" if len(request_bio) > MAX_BIO_CHARS else "")
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": f"{PROMPT}\n\nID: {identifier}\nNARRATIVE:\n{request_story or '[EMPTY]'}\nCREATOR_BIO:\n{request_bio or '[EMPTY]'}"}],
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": "low", "format": {"type": "json_schema", "name": "parenthood_disclosure",
                                            "strict": True, "schema": SCHEMA}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    item = json.loads(response.output_text)["items"][0]
    if item.get("id") != identifier:
        raise ValueError("Model response ID does not match request ID")
    record = output_record(identifier)
    issues = []
    for name in LABELS:
        value, problem = validate_label(item[name], story, bio)
        item[name] = value
        if problem:
            issues.append(f"{name}: {problem}")
    if item["father"]["yes"] or item["mother"]["yes"]:
        item["parent_unspecified"] = empty_label()
    elif not item["parent_unspecified"]["yes"]:
        item["parent_unspecified"] = backstop(story, bio)
    for name in LABELS:
        for field, value in item[name].items():
            record[f"{name}_{field}"] = value
    record["needs_manual_review"] = bool(item["needs_manual_review"] or issues)
    record["review_reason"] = "; ".join(filter(None, [item["review_reason"], *issues])) if record["needs_manual_review"] else ""
    return record


def source_rows(path):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected one CSV inside the ZIP archive")
            with archive.open(names[0]) as stream:
                import io
                yield from csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline=""))
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Put the classification ZIP next to this script: {INPUT}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment before running")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    completed = set()
    if OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8-sig", newline="") as stream:
            completed = {row["id"] for row in csv.DictReader(stream) if not row.get("error")}
    client = OpenAI()
    with OUTPUT.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        if OUTPUT.stat().st_size == 0:
            writer.writeheader()
        for row in source_rows(INPUT):
            identifier = row["id"]
            if identifier in completed:
                continue
            story, bio = row.get("Campaign story", ""), row.get("creator_bio", "")
            flags = {flag: row.get(flag, "") for flag in ("skipped_short_narrative", "skipped_non_english")}
            if flags["skipped_short_narrative"].lower() == "true" or flags["skipped_non_english"].lower() == "true" or len(re.findall(r"\b\w+\b", story)) < MIN_WORDS:
                result = output_record(identifier, flags)
            else:
                try:
                    result = classify(client, identifier, story, bio)
                except Exception as exc:
                    result = output_record(identifier)
                    result["error"] = str(exc)
            result.update(flags)
            writer.writerow(result)
            stream.flush()
            if result["error"]:
                print(f"{identifier}: {result['error']}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
