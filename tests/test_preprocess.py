import pandas as pd

from src import config
from src.preprocess import clean_text, normalize_text, prepare


def test_agency_tags_and_entities_removed():
    raw = "Wall St. Bears Claw Back (Reuters) Reuters - Short-sellers #39;s quot;rally quot;"
    out = normalize_text(raw)
    assert "Reuters" not in out
    assert "#39;" not in out and "quot;" not in out
    assert out.startswith("Wall St. Bears Claw Back")


def test_clean_text_keeps_only_lowercase_words():
    assert clean_text("Apple's iPod (AP) AP - sales up 25%!") == "apple s ipod sales up"


def test_label_mapping_matches_ag_news():
    assert config.ID2LABEL == {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    assert all(config.LABEL2ID[name] == i for i, name in config.ID2LABEL.items())


def test_prepare_produces_both_text_views():
    df = pd.DataFrame({"text": ["Oil prices rise (AFP)", "Team wins final"], "label": [2, 1]})
    out = prepare(df)
    assert list(out.columns) == ["label", "text", "clean_text", "label_name"]
    assert out["label_name"].tolist() == ["Business", "Sports"]
    assert out.loc[0, "text"] == "Oil prices rise"
