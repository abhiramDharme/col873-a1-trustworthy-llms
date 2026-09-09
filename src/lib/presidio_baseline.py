from __future__ import annotations

from .align import Span

_PRESIDIO_TO_300K = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "TEL",
    "IP_ADDRESS": "IP",
    "DATE_TIME": "DATE",
    "URL": "USERNAME",
    "US_SSN": "SOCIALNUMBER",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "CREDIT_CARD": "IDCARD",
    "PERSON": "GIVENNAME1",
    "LOCATION": "CITY",
    "NRP": "COUNTRY",
    "IBAN_CODE": "IDCARD",
    "CRYPTO": "IDCARD",
    "MEDICAL_LICENSE": "IDCARD",
}

_PRESIDIO_TO_OOD = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONENUMBER",
    "IP_ADDRESS": "IPV4",
    "DATE_TIME": "DATE",
    "URL": "URL",
    "US_SSN": "SSN",
    "US_PASSPORT": "PASSPORT",
    "US_DRIVER_LICENSE": "DRIVERLICENSE",
    "CREDIT_CARD": "CREDITCARDNUMBER",
    "PERSON": "FIRSTNAME",
    "LOCATION": "CITY",
    "NRP": "COUNTRY",
    "IBAN_CODE": "IBAN",
    "CRYPTO": "BITCOINADDRESS",
}


def _offline_spacy_http() -> None:
    """SpaCy/Presidio probe GitHub for model compatibility; HPC compute has no DNS."""
    import requests

    if getattr(requests.get, "_a1_offline", False):
        return

    class _Dummy:
        status_code = 200
        text = "{}"
        content = b"{}"

        def json(self):
            return {}

        def raise_for_status(self):
            return None

    real_get = requests.get

    def fake_get(url, *args, **kwargs):
        text = str(url)
        if "githubusercontent.com" in text or "compatibility.json" in text:
            return _Dummy()
        return real_get(url, *args, **kwargs)

    fake_get._a1_offline = True  # type: ignore[attr-defined]
    requests.get = fake_get


class PresidioSanitizer:
    def __init__(self, ood: bool = False):
        _offline_spacy_http()
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        )
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        self.mapping = _PRESIDIO_TO_OOD if ood else _PRESIDIO_TO_300K

    def spans(self, text: str) -> list[Span]:
        results = self.engine.analyze(text=text, language="en")
        spans: list[Span] = []
        for item in results:
            label = self.mapping.get(item.entity_type)
            if not label:
                continue
            spans.append(Span(item.start, item.end, label))
        spans.sort(key=lambda s: (s.start, -s.end))
        occupied = [False] * len(text)
        kept: list[Span] = []
        for span in spans:
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
