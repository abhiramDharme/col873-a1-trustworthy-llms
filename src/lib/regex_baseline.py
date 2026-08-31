"""Rule baseline: field-aware extractors plus generic contact/network/date patterns.

Labels are emitted in the ai4privacy-300k vocabulary unless `ood=True`, in which
case the 200k names (PHONENUMBER, FIRSTNAME, …) are used when they exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .align import Span

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.I)
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")
IPV6_RE = re.compile(
    r"\b(?:[0-9A-F]{1,4}:){7}[0-9A-F]{1,4}\b"
    r"|\b(?:[0-9A-F]{1,4}:){1,7}:\b"
    r"|\b(?:[0-9A-F]{1,4}:){1,6}:[0-9A-F]{1,4}\b",
    re.I,
)
URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{2,4}[\s.\-]\d{2,4}(?:[\s.\-]\d{2,4})?(?!\w)"
)
SSN_RE = re.compile(r"\b\d{3}[-.\s]\d{2}[-.\s]\d{4}\b|\b\d{9,10}\b")
PASSPORT_RE = re.compile(r"\b[A-Z]{0,2}\d{7,9}\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"[a-z]*,?\s+\d{4})\b",
    re.I,
)
TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3])[:.][0-5]\d(?:\s?(?:AM|PM|am|pm))?\b|\b(?:[1-9]|1[0-2])\s?(?:AM|PM|am|pm)\b")
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
GEO_RE = re.compile(r"\[?\s*-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+\s*\]?")

FIELD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"(?i)(?:e-?mail(?:\s+address)?|contact information for feedback)\s*[:=]\s*(\S+@\S+)")),
    ("TEL", re.compile(r"(?i)(?:telephone(?:\s+number)?|phone(?:\s+number)?|tel|mobile)\s*[:=]\s*([+\d][\d.\-() ]{6,})")),
    ("SOCIALNUMBER", re.compile(r"(?i)(?:social(?:\s+security)?\s+number|ssn|social number)\s*[:=]\s*([\d.\-]{8,})")),
    ("PASSPORT", re.compile(r"(?i)passport(?:\s*(?:id|number|#))?\s*[:=]\s*([A-Za-z0-9]{6,12})")),
    ("IDCARD", re.compile(r"(?i)(?:identification card number|id(?:card)?(?:\s+number)?|national id)\s*[:=]\s*([A-Za-z0-9\-]{6,20})")),
    ("DRIVERLICENSE", re.compile(r"(?i)(?:driver'?s?\s+license|dl(?:\s+number)?)\s*[:=]\s*([A-Za-z0-9\-]{5,20})")),
    ("IP", re.compile(r"(?i)ip(?:v[46])?(?:\s+address)?\s*[:=]\s*([0-9A-Fa-f:.]+)")),
    ("PASS", re.compile(r"(?i)(?:pass(?:code|word)|pwd)\s*[:=]\s*([^\s<]{4,})")),
    ("USERNAME", re.compile(r"(?i)user(?:name)?\s*[:=]\s*([A-Za-z0-9._\-]{3,})")),
    ("DATE", re.compile(r"(?i)(?:date|assessment date)\s*[:=]\s*([0-9A-Za-z/.\- ]{4,20})")),
    ("BOD", re.compile(r"(?i)(?:d\.?o\.?b\.?|date of birth|bodily origin date|birth(?:day| date))\s*[:=]\s*([0-9A-Za-z/.\- ]{4,20})")),
    ("GIVENNAME1", re.compile(r"(?i)(?:given name|first name|fullname)\s*[:=]?\s*>?\s*([A-Z][A-Za-z\-']+)")),
    ("LASTNAME1", re.compile(r"(?i)(?:last name|surname)\s*[:=]\s*([A-Z][A-Za-z\-']+)")),
    ("CITY", re.compile(r"(?i)\bcity\s*[:=]\s*([A-Za-z .'-]{2,40})")),
    ("STATE", re.compile(r"(?i)\bstate\s*[:=]\s*([A-Za-z .'-]{2,40})")),
    ("POSTCODE", re.compile(r"(?i)(?:post(?:al)?\s*code|zip(?:\s*code)?)\s*[:=]\s*([A-Za-z0-9\- ]{3,12})")),
    ("STREET", re.compile(r"(?i)(?:street|address)\s*[:=]\s*([^\n<]{4,60})")),
    ("BUILDING", re.compile(r"(?i)building(?:\s+number)?\s*[:=]\s*([A-Za-z0-9\- ]{1,20})")),
    ("COUNTRY", re.compile(r"(?i)\bcountry\s*[:=]\s*([A-Za-z .'-]{2,40})")),
    ("SEX", re.compile(r"(?i)\b(?:sex|gender)\s*[:=]\s*([A-Za-z]{3,20})")),
    ("TIME", re.compile(r"(?i)(?:best time to connect|time|timestamp)\s*[:=]\s*([0-9A-Za-z:. ]{1,12})")),
    ("TITLE", re.compile(r"(?i)\btitle\s*[:=]\s*([A-Za-z .'-]{2,40})")),
    ("GEOCOORD", re.compile(r"(?i)(?:geo(?:coordinates)?|coordinates)\s*[:=]\s*(\[?-?\d.+,?\s*-?\d.+\]?)")),
]


OOD_LABEL = {
    "TEL": "PHONENUMBER",
    "PASS": "PASSWORD",
    "GIVENNAME1": "FIRSTNAME",
    "LASTNAME1": "LASTNAME",
    "BOD": "DOB",
    "POSTCODE": "ZIPCODE",
    "SECADDRESS": "SECONDARYADDRESS",
    "BUILDING": "BUILDINGNUMBER",
    "SOCIALNUMBER": "SSN",
    "IP": "IPV4",
}


def _luhn_ok(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class RegexSanitizer:
    ood: bool = False

    def _label(self, label: str) -> str:
        if self.ood:
            return OOD_LABEL.get(label, label)
        return label

    def spans(self, text: str) -> list[Span]:
        found: list[Span] = []

        def add(start: int, end: int, label: str) -> None:
            if start < 0 or end <= start or end > len(text):
                return
            found.append(Span(start, end, self._label(label)))

        for label, pattern in FIELD_PATTERNS:
            for match in pattern.finditer(text):
                if match.lastindex:
                    add(match.start(1), match.end(1), label)
                else:
                    add(match.start(), match.end(), label)

        for match in EMAIL_RE.finditer(text):
            add(match.start(), match.end(), "EMAIL")
        for match in IPV4_RE.finditer(text):
            add(match.start(), match.end(), "IP" if not self.ood else "IPV4")
        for match in IPV6_RE.finditer(text):
            add(match.start(), match.end(), "IP" if not self.ood else "IPV6")
        for match in URL_RE.finditer(text):
            add(match.start(), match.end(), "URL" if self.ood else "USERNAME")
        for match in PHONE_RE.finditer(text):
            span_text = match.group(0)
            if sum(ch.isdigit() for ch in span_text) >= 8:
                add(match.start(), match.end(), "TEL")
        for match in SSN_RE.finditer(text):
            add(match.start(), match.end(), "SOCIALNUMBER")
        for match in PASSPORT_RE.finditer(text):
            add(match.start(), match.end(), "PASSPORT")
        for match in DATE_RE.finditer(text):
            add(match.start(), match.end(), "DATE")
        for match in TIME_RE.finditer(text):
            add(match.start(), match.end(), "TIME")
        for match in GEO_RE.finditer(text):
            add(match.start(), match.end(), "GEOCOORD")
        for match in CARD_RE.finditer(text):
            raw = match.group(0)
            if _luhn_ok(raw):
                add(match.start(), match.end(), "IDCARD" if not self.ood else "CREDITCARDNUMBER")

        found.sort(key=lambda s: (s.start, -(s.end - s.start)))
        kept: list[Span] = []
        occupied = [False] * len(text)
        for span in found:
            if any(occupied[span.start : span.end]):
                continue
            for i in range(span.start, span.end):
                occupied[i] = True
            kept.append(span)
        kept.sort(key=lambda s: (s.start, s.end))
        return kept

    def sanitize(self, text: str) -> tuple[str, list[Span]]:
        spans = self.spans(text)
        pieces: list[str] = []
        cursor = 0
        for span in spans:
            pieces.append(text[cursor : span.start])
            pieces.append(f"[{span.label}]")
            cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces), spans
