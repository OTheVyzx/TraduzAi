import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def test_glossary_used_report_has_complete_summary_and_hits():
    report = main.build_glossary_used_report(
        {"obra": "Teste", "idioma_origem": "ko", "glossario": {"Bukmyeongdae": "Bukmyeongdae"}},
        {"fontes_usadas": ["manual"]},
        [
            {
                "texts": [
                    {
                        "id": "t1",
                        "glossary_hits": [
                            {"phase": "target", "source": "Bukmyeongdae", "target": "Bukmyeongdae"}
                        ],
                        "qa_flags": [],
                    }
                ]
            }
        ],
    )

    assert report["work_identity"]["title"] == "Teste"
    assert report["summary"]["terms_loaded"] == 1
    assert report["summary"]["terms_used"] == 1
    assert report["summary"]["locked_terms_used"] == 1
    assert report["hits"][0]["source_term"] == "Bukmyeongdae"


def test_glossary_used_report_explains_empty_glossary():
    report = main.build_glossary_used_report(
        {"obra": ""},
        {},
        [{"texts": []}],
    )

    assert report["summary"]["terms_loaded"] == 0
    assert report["summary"]["empty_reason"] == "work_identity_unresolved"
