"""Prompt and model sensitivity analysis of entrepreneur parenthood disclosure.

Authors: See repository README.

Set the OPENAI_API_KEY environment variable before running.

"""

import argparse
import csv
import json
import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

MODELS = {"mini": "gpt-5-mini", "advanced": "gpt-5.2"}
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 1800
NARRATIVE_MAX_CHARS = 20000
BIO_MAX_CHARS = 8000
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10
PAUSE_BETWEEN_REQUESTS_SECONDS = 0.5

SYSTEM_ROLE = "You are a careful research coder. Follow definitions strictly. Never guess."
SYSTEM_NEUTRAL = "Classify the text using the definitions provided. Do not guess."

DEFINITIONS = """Identify PARENTHOOD DISCLOSURE BY THE CAMPAIGN CREATOR using the
NARRATIVE and CREATOR_BIO. This task does NOT classify product target audiences.

- father=yes only when the creator explicitly identifies as a father/dad
  (including unambiguous equivalent self-descriptions).
- mother=yes only when the creator explicitly identifies as a mother/mom
  (including unambiguous equivalent self-descriptions).
- parent_unspecified=yes when the creator explicitly describes being a parent
  or clearly describes their own child without stating mother/father identity.
  Examples of evidence include 'my son', 'my daughter', 'our kids', and
  'the birth of our child'.
- If father=yes or mother=yes, parent_unspecified MUST be no.
- Do NOT infer gender from the creator's name, child's gender, stereotypes,
  or any other indirect cues.
- Do NOT code other people's parenting, a product intended for children or
  parents, generic words about family, hypothetical examples within a campaign,
  or metaphorical language such as 'this project is my baby'.
- If there is uncertainty about whose relationship is described, mark for
  manual review instead of inventing an inference.

For each YES label, supply a verbatim supporting quote from narrative, bio,
or both, and indicate the evidence source. For each NO label, use blank quotes
and evidence_source=''. Flag genuinely ambiguous cases for manual review.
Return only the structured JSON response, with the same id as the input."""

ROLE_PROMPT = """Task: Detect founder parenthood disclosure.
You have NARRATIVE (campaign story) and CREATOR_BIO.
Carefully apply the following coding instructions as an expert research coder.
""" + DEFINITIONS

FEWSHOT_PROMPT = """Task: Classify founder parenthood disclosure.
Use the following labeled, hypothetical examples as demonstrations.

Example 1
NARRATIVE: 'I am a mother of two, and I designed this puzzle after work.'
CREATOR_BIO: ''
Correct classification: mother=yes, father=no, parent_unspecified=no.
Evidence: 'I am a mother of two' (narrative).

Example 2
NARRATIVE: ''
CREATOR_BIO: 'I'm a dad to three wonderful kids and a photographer.'
Correct classification: father=yes, mother=no, parent_unspecified=no.
Evidence: "I'm a dad to three wonderful kids" (bio).

Example 3
NARRATIVE: 'Our daughter inspired us to make this project.'
CREATOR_BIO: ''
Correct classification: parent_unspecified=yes, mother=no, father=no.
Evidence: 'Our daughter inspired us to make this project.' (narrative).

Example 4
NARRATIVE: 'This toy helps parents teach their children to read.'
CREATOR_BIO: 'I am a toy designer.'
Correct classification: mother=no, father=no, parent_unspecified=no.
No parenthood evidence: references to customers, not the creator's children.

Now apply the SAME general coding definitions below to the actual campaign.
""" + DEFINITIONS

COT_PROMPT = """Task: Classify founder parenthood disclosure.
Apply the common rules below, using this explicit decision sequence internally:
(1) Identify who is being described (creator vs. customers/third parties).
(2) Locate an explicit statement about the creator's own parental identity
    or own child, if one exists.
(3) Check whether that statement names mother or father. If neither, assess
    whether it clearly indicates the creator's own parenthood.
(4) Exclude metaphors, hypothetical claims, product targeting, and ambiguity.
(5) Return the final labels and verbatim evidence, without an explanation of
    your internal reasoning steps.

Apply the following SAME construct definitions:
""" + DEFINITIONS

PROMPTS = {"role": ROLE_PROMPT, "fewshot": FEWSHOT_PROMPT, "cot": COT_PROMPT}

CONDITIONS = {
    "role_advanced": ("role", "advanced"),
    "role_mini": ("role", "mini"),
    "fewshot_mini": ("fewshot", "mini"),
    "fewshot_advanced": ("fewshot", "advanced"),
    "cot_mini": ("cot", "mini"),
    "cot_advanced": ("cot", "advanced"),
}

LABEL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "yes": {"type": "boolean"},
        "evidence_source": {"type": "string", "enum": ["", "narrative", "bio", "both"]},
        "quote_narrative": {"type": "string"},
        "quote_bio": {"type": "string"},
    },
    "required": ["yes", "evidence_source", "quote_narrative", "quote_bio"],
}

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "mother": LABEL_SCHEMA,
        "father": LABEL_SCHEMA,
        "parent_unspecified": LABEL_SCHEMA,
        "needs_manual_review": {"type": "boolean"},
        "review_reason": {"type": "string"},
    },
    "required": ["id", "mother", "father", "parent_unspecified",
                 "needs_manual_review", "review_reason"],
}


FIELDNAMES = [
    "id", "condition", "prompt_type", "model", "status",
    "mother_yes", "father_yes", "parent_unspecified_yes", "any_parent_yes",
    "mother_evidence_source", "mother_quote_narrative", "mother_quote_bio",
    "father_evidence_source", "father_quote_narrative", "father_quote_bio",
    "parent_unspecified_evidence_source",
    "parent_unspecified_quote_narrative", "parent_unspecified_quote_bio",
    "needs_manual_review", "review_reason", "qc_notes",
    "input_tokens", "output_tokens", "model_returned", "completed_utc", "error",
]


def load_sample(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "id" not in rows[0]:
        raise ValueError("Validation file must contain a nonempty id column")
    ids = [row["id"].strip() for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Validation IDs must be nonempty and unique")
    return ids


def load_campaigns(path, ids):
    required = {"id", "Campaign story", "creator_bio"}
    selected = {}

    def collect(handle):
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Classification data must contain {sorted(required)}")
        for row in reader:
            study_id = row["id"].strip()
            if study_id in ids:
                if study_id in selected:
                    raise ValueError(f"Duplicate campaign ID: {study_id}")
                selected[study_id] = row

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError("Dataset ZIP must contain exactly one CSV")
            import io
            with archive.open(members[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
                    collect(handle)
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            collect(handle)

    missing = ids - selected.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} sampled IDs (e.g., {sorted(missing)[0]})")
    return selected


def shorten(value, limit):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value[:limit] + " [TRUNCATED]" if len(value) > limit else value


def normalize(value):
    value = str(value or "").lower().replace("’", "'").replace("‘", "'")
    value = value.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value).strip()


def validate_and_flatten(study_id, obj, narrative, bio):
    if obj.get("id") != study_id:
        raise ValueError("Model returned a mismatched study ID")
    result, qc = {}, []
    for label in ("mother", "father", "parent_unspecified"):
        item = obj[label]
        yes = item["yes"]
        source = item["evidence_source"]
        narrative_quote = item["quote_narrative"]
        bio_quote = item["quote_bio"]
        if not isinstance(yes, bool):
            raise ValueError(f"{label}: expected a boolean label")
        result.update({
            f"{label}_yes": yes,
            f"{label}_evidence_source": source,
            f"{label}_quote_narrative": narrative_quote,
            f"{label}_quote_bio": bio_quote,
        })
        if yes:
            if not source or (source in ("narrative", "both") and not narrative_quote) or (source in ("bio", "both") and not bio_quote):
                qc.append(f"{label}: missing evidence")
            if narrative_quote and normalize(narrative_quote) not in normalize(narrative):
                qc.append(f"{label}: narrative quote not found")
            if bio_quote and normalize(bio_quote) not in normalize(bio):
                qc.append(f"{label}: bio quote not found")
        elif source or narrative_quote or bio_quote:
            qc.append(f"{label}: negative label has evidence")
    if (result["father_yes"] or result["mother_yes"]) and result["parent_unspecified_yes"]:
        qc.append("Named parental role and unspecified parenthood both marked yes")
    result["any_parent_yes"] = any(result[f"{label}_yes"] for label in ("mother", "father", "parent_unspecified"))
    result["needs_manual_review"] = bool(obj["needs_manual_review"] or qc)
    result["review_reason"] = obj["review_reason"]
    result["qc_notes"] = " | ".join(qc)
    return result


def completed_pairs(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {(row["id"], row["condition"]) for row in csv.DictReader(handle)
                if row["status"] == "success"}


def append_result(path, result):
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(result)
        handle.flush()
        os.fsync(handle.fileno())


def classify(client, study_id, campaign, condition):
    prompt_type, model_type = CONDITIONS[condition]
    model = MODELS[model_type]
    narrative = shorten(campaign["Campaign story"], NARRATIVE_MAX_CHARS)
    bio = shorten(campaign["creator_bio"], BIO_MAX_CHARS)
    system_message = SYSTEM_ROLE if prompt_type == "role" else SYSTEM_NEUTRAL
    user_message = (PROMPTS[prompt_type] + "\n\nID: " + study_id
                    + "\nNARRATIVE:\n" + (narrative or "[EMPTY]")
                    + "\nCREATOR_BIO:\n" + (bio or "[EMPTY]"))
    response = client.responses.create(
        model=model,
        input=[{"role": "system", "content": system_message},
               {"role": "user", "content": user_message}],
        reasoning={"effort": REASONING_EFFORT},
        text={"verbosity": "low", "format": {
            "type": "json_schema", "name": "parenthood_only_validation",
            "strict": True, "schema": SCHEMA}},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    if not response.output_text:
        raise ValueError(f"No JSON output; response status={response.status}")
    result = validate_and_flatten(study_id, json.loads(response.output_text), narrative, bio)
    usage = getattr(response, "usage", None)
    result.update({
        "id": study_id, "condition": condition, "prompt_type": prompt_type,
        "model": model, "status": "success",
        "input_tokens": getattr(usage, "input_tokens", "") if usage else "",
        "output_tokens": getattr(usage, "output_tokens", "") if usage else "",
        "model_returned": getattr(response, "model", model),
        "completed_utc": datetime.now(timezone.utc).isoformat(), "error": "",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("kickstarter_narrative_and_bios_classified.zip"))
    parser.add_argument("--validation", type=Path, default=Path("validation_200.csv"))
    parser.add_argument("--output", type=Path, default=Path("sensitivity_results_new.csv"))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set the OPENAI_API_KEY environment variable")
    ids = load_sample(args.validation)
    campaigns = load_campaigns(args.data, set(ids))
    completed = completed_pairs(args.output)
    client = OpenAI()
    for condition in args.conditions:
        for index, study_id in enumerate(ids, start=1):
            if (study_id, condition) in completed:
                continue
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    result = classify(client, study_id, campaigns[study_id], condition)
                    append_result(args.output, result)
                    completed.add((study_id, condition))
                    print(f"{condition}: {index}/{len(ids)} complete")
                    break
                except Exception as exc:
                    if attempt == RETRY_ATTEMPTS:
                        append_result(args.output, {
                            "id": study_id, "condition": condition,
                            "prompt_type": CONDITIONS[condition][0],
                            "model": MODELS[CONDITIONS[condition][1]],
                            "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                        })
                    else:
                        time.sleep(RETRY_DELAY_SECONDS * attempt)
            time.sleep(PAUSE_BETWEEN_REQUESTS_SECONDS)


if __name__ == "__main__":
    main()
