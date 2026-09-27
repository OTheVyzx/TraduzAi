"""Source-grounded casing for new quality recipes only."""
from __future__ import annotations

import re

_CONFUSABLE = re.compile(r"(?i)\b(?:[a-z]*[015][a-z]+|[a-z]+[015][a-z]*)\b")


def profile(source: str) -> dict:
    tokens = re.findall(r"[A-Za-z0-9]+", source)
    suspicious = [word for word in tokens if _CONFUSABLE.fullmatch(word)]
    letters = [word for word in tokens if word.isalpha()]
    anomalous = [word for word in letters if len(word) >= 3 and
                 word[0].islower() and word[1:].isupper()]
    evidence = [word for word in letters if word not in anomalous]
    if not evidence or (len(evidence) < 2 and (suspicious or anomalous)):
        kind,confidence,reason = 'UNCERTAIN',0.0,'insufficient_reliable_case_evidence'
    elif all(word.isupper() for word in evidence):
        kind,confidence,reason = 'ALL_UPPER',len(evidence)/max(1,len(letters)),'reliable_letters_upper'
    elif all(word.islower() for word in evidence):
        kind,confidence,reason = 'ALL_LOWER',len(evidence)/max(1,len(letters)),'reliable_letters_lower'
    else:
        kind,confidence,reason = 'SENTENCE_OR_MIXED',len(evidence)/max(1,len(letters)),'mixed_reliable_letters'
    return dict(source_case_profile=kind,case_evidence=dict(reliable_tokens=evidence,
                ignored_ocr_confusions=suspicious,ignored_case_anomalies=anomalous),
                case_confidence=round(confidence,3),case_reason=reason)


def apply(source: str, target: str) -> tuple[str,dict]:
    decision=profile(source)
    kind=decision['source_case_profile']
    transform={'ALL_UPPER':'upper','ALL_LOWER':'lower'}.get(kind,'none')
    decision['target_case_transform']=transform
    return (target.upper() if transform=='upper' else
            target.lower() if transform=='lower' else target),decision
