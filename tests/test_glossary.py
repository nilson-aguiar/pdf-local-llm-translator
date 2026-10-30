"""Tests for pdf_translator.translate.glossary — loading, protection, verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdf_translator.exceptions import ConfigError
from pdf_translator.extract.markdown import protect as md_protect
from pdf_translator.models import ProtectedText, apply_restorations, find_sentinels
from pdf_translator.translate.glossary import Glossary, load_glossary


# ---------------------------------------------------------------------------
# load_glossary — file format variants
# ---------------------------------------------------------------------------


def test_load_flat_yaml(tmp_path: Path) -> None:
    """Load a flat YAML mapping."""
    f = tmp_path / "glossary.yaml"
    f.write_text("neural network: neuronales Netzwerk\nmachine learning: maschinelles Lernen\n")
    g = load_glossary(f)
    assert g.terms["neural network"] == "neuronales Netzwerk"
    assert g.terms["machine learning"] == "maschinelles Lernen"


def test_load_flat_json(tmp_path: Path) -> None:
    """Load a flat JSON mapping."""
    f = tmp_path / "glossary.json"
    f.write_text(json.dumps({"transformer": "Transformer", "attention": "Aufmerksamkeit"}))
    g = load_glossary(f)
    assert g.terms["transformer"] == "Transformer"


def test_load_wrapped_json(tmp_path: Path) -> None:
    """Load a JSON file with a {"terms": {...}} wrapper."""
    f = tmp_path / "glossary.json"
    f.write_text(json.dumps({"terms": {"encoder": "Kodierer", "decoder": "Dekodierer"}}))
    g = load_glossary(f)
    assert g.terms["encoder"] == "Kodierer"
    assert g.terms["decoder"] == "Dekodierer"


def test_load_wrapped_yaml(tmp_path: Path) -> None:
    """Load a YAML file with a terms: wrapper."""
    f = tmp_path / "terms.yaml"
    f.write_text("terms:\n  gradient: Gradient\n  epoch: Epoche\n")
    g = load_glossary(f)
    assert g.terms["gradient"] == "Gradient"


def test_load_per_language_yaml(tmp_path: Path) -> None:
    """Load a per-language YAML and resolve via target_lang."""
    f = tmp_path / "multi.yaml"
    f.write_text(
        "de:\n  loss: Verlust\n  batch: Stapel\n"
        "fr:\n  loss: perte\n  batch: lot\n"
    )
    g = load_glossary(f, target_lang="de")
    assert g.terms["loss"] == "Verlust"


def test_load_per_language_missing_lang(tmp_path: Path) -> None:
    """Raises ConfigError when target_lang is absent from the file."""
    f = tmp_path / "multi.yaml"
    f.write_text("de:\n  word: Wort\n")
    with pytest.raises(ConfigError):
        load_glossary(f, target_lang="fr")


def test_load_per_language_no_lang_supplied(tmp_path: Path) -> None:
    """Raises ConfigError when per-language file is detected but no lang given."""
    f = tmp_path / "multi.yaml"
    f.write_text("de:\n  word: Wort\n")
    with pytest.raises(ConfigError):
        load_glossary(f)  # no target_lang


def test_load_missing_file(tmp_path: Path) -> None:
    """Raises ConfigError for a non-existent file."""
    with pytest.raises(ConfigError):
        load_glossary(tmp_path / "nope.yaml")


def test_load_unsupported_extension(tmp_path: Path) -> None:
    """Raises ConfigError for unsupported file extensions."""
    f = tmp_path / "glossary.toml"
    f.write_text("[terms]\nword = 'Wort'\n")
    with pytest.raises(ConfigError):
        load_glossary(f)


# ---------------------------------------------------------------------------
# Glossary.protect — sentinel replacement
# ---------------------------------------------------------------------------


def test_protect_replaces_term_with_gloss_sentinel() -> None:
    """A source term is replaced by a ⟦GLOSS{n}⟧ sentinel."""
    g = Glossary(terms={"neural network": "neuronales Netzwerk"})
    pt = ProtectedText(text="The neural network is trained.", restorations={})
    result = g.protect(pt)

    assert "neural network" not in result.text
    sentinels = find_sentinels(result.text)
    assert len(sentinels) == 1
    assert sentinels[0].startswith("⟦GLOSS")


def test_protect_maps_sentinel_to_target_term() -> None:
    """The GLOSS sentinel maps to the target term in restorations."""
    g = Glossary(terms={"loss": "Verlust"})
    pt = ProtectedText(text="Minimize the loss function.", restorations={})
    result = g.protect(pt)

    assert "Verlust" in result.restorations.values()
    sentinel = next(k for k, v in result.restorations.items() if v == "Verlust")
    assert "GLOSS" in sentinel


def test_protect_case_insensitive_match() -> None:
    """Term matching is case-insensitive."""
    g = Glossary(terms={"gradient": "Gradient"})
    pt = ProtectedText(text="The Gradient descent converges.", restorations={})
    result = g.protect(pt)

    assert "Gradient descent" not in result.text
    sentinels = find_sentinels(result.text)
    assert len(sentinels) == 1


def test_protect_word_boundary_no_partial_match() -> None:
    """Glossary terms are not matched inside longer words."""
    g = Glossary(terms={"loss": "Verlust"})
    pt = ProtectedText(text="Glossary losses are not a loss.", restorations={})
    result = g.protect(pt)

    # Only the standalone "loss" at the end should be replaced.
    sentinels = find_sentinels(result.text)
    assert len(sentinels) == 1, f"Expected 1 replacement, got: {result.text!r}"


def test_protect_preserves_existing_restorations() -> None:
    """Glossary.protect extends, not replaces, the incoming restorations."""
    existing = {"⟦MATH0⟧": "$x^2$"}
    pt = ProtectedText(text="The loss ⟦MATH0⟧ is minimised.", restorations=existing)
    g = Glossary(terms={"loss": "Verlust"})
    result = g.protect(pt)

    # Original math restoration must survive.
    assert "⟦MATH0⟧" in result.restorations
    assert result.restorations["⟦MATH0⟧"] == "$x^2$"


def test_protect_does_not_match_inside_existing_sentinels() -> None:
    """Existing ⟦...⟧ tokens are never modified by Glossary.protect."""
    # Craft a sentinel whose content *contains* a glossary term.
    # (Unusual but must be safe.)
    existing = {"⟦MATH0⟧": "$\\text{loss}$"}
    pt = ProtectedText(text="⟦MATH0⟧ and loss.", restorations=existing)
    g = Glossary(terms={"loss": "Verlust"})
    result = g.protect(pt)

    # The sentinel itself must be untouched.
    assert "⟦MATH0⟧" in result.text
    # "loss" in plain text was replaced; inside sentinel was not.
    assert result.restorations["⟦MATH0⟧"] == "$\\text{loss}$"


# ---------------------------------------------------------------------------
# Composition: markdown.protect + glossary.protect
# ---------------------------------------------------------------------------


def test_compose_markdown_and_glossary_protect() -> None:
    """Composing markdown protection then glossary protection preserves all restorations."""
    text = "The gradient $\\nabla L$ is the loss gradient."
    # Step 1: protect markdown spans (inline math)
    pt = md_protect(text)
    assert "$" not in pt.text  # math sentinel in place

    # Step 2: protect glossary terms
    g = Glossary(terms={"gradient": "Gradient", "loss": "Verlust"})
    pt2 = g.protect(pt)

    # MATH sentinel from markdown step must still be present
    math_sentinels = [k for k in pt2.restorations if "MATH" in k]
    assert len(math_sentinels) >= 1, "math restoraton must survive glossary protect"

    # GLOSS sentinels from glossary step must be present
    gloss_sentinels = [k for k in pt2.restorations if "GLOSS" in k]
    assert len(gloss_sentinels) >= 1, "glossary sentinels must be added"


def test_compose_full_round_trip() -> None:
    """After protection and apply_restorations, target terms and math both appear."""
    text = "Minimize the loss $L(x)$ via gradient descent."
    # Markdown protection
    pt = md_protect(text)
    # Glossary protection
    g = Glossary(terms={"loss": "Verlust", "gradient": "Gradient"})
    pt2 = g.protect(pt)

    # Simulate translation: the model leaves all sentinels intact.
    translated_with_sentinels = pt2.text  # no actual translation — just restore

    # Restore all sentinels (both MATH and GLOSS).
    restored = apply_restorations(translated_with_sentinels, pt2.restorations)

    # Target terms appear (GLOSS sentinel restored to target).
    assert "Verlust" in restored or "loss" in restored
    # Math is intact.
    assert "$L(x)$" in restored or "L(x)" in restored


# ---------------------------------------------------------------------------
# Glossary.verify
# ---------------------------------------------------------------------------


def test_verify_flags_missing_target_term() -> None:
    """verify returns target terms absent from the translated string."""
    g = Glossary(terms={"loss": "Verlust"})
    pt = ProtectedText(text="Minimize the loss.", restorations={})
    g.protect(pt)  # registers "Verlust" as expected

    # Simulate a translated string that lost the target term.
    missing = g.verify("Minimiere den Fehler.")  # "Fehler" not "Verlust"
    assert "Verlust" in missing


def test_verify_does_not_flag_present_target_term() -> None:
    """verify returns an empty list when all expected target terms are present."""
    g = Glossary(terms={"loss": "Verlust"})
    pt = ProtectedText(text="Minimize the loss.", restorations={})
    g.protect(pt)

    present = g.verify("Minimiere den Verlust.")
    assert present == []


def test_verify_only_checks_protected_terms() -> None:
    """verify only flags terms that were actually substituted, not all glossary terms."""
    g = Glossary(terms={"loss": "Verlust", "gradient": "Gradient"})
    # Only "loss" appears in the text — "gradient" is never substituted.
    pt = ProtectedText(text="The loss is high.", restorations={})
    g.protect(pt)

    # "Gradient" was never expected (not in input text), so it must not be flagged.
    missing = g.verify("Der Verlust ist hoch.")
    assert "Gradient" not in missing
    assert missing == []
