"""A contrast assembled from verified sides, never written as new prose.

A comparison's core requirement is a statement of how the subjects
differ, and the honest form of such a statement asserts two things:

    Fine-tuning changes a model's behavioural patterns, while
    retrieval-augmented generation changes the information it can access

Atomicity refuses that, correctly -- every other guard reasons about
"the sentence that supports this claim". Across nine hosted runs the
synthesiser produced one `direct_contrast` claim and atomicity refused
it; a later run published three supported, relevant, cited claims about
both subjects and reported that it had not answered.

So the contrast is structure over verified claims, not a new sentence.
Nothing in a pair was generated to fill it.
"""

from __future__ import annotations

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.answer_coverage import assess_coverage
from agentic_research.comparison import (
    SideClaim,
    build_comparison_pairs,
    render_pairs,
)

RAG = "How does retrieval-augmented generation differ from fine-tuning for language models?"


def contract():
    return build_contract(
        RAG,
        QuestionType.COMPARISON,
        entities=["retrieval-augmented generation", "fine-tuning", "language models"],
        comparison_subjects=["retrieval-augmented generation", "fine-tuning"],
        # A named axis, which the analyst now proposes for every
        # comparison. The generic `dimension` slot cannot complete a
        # contrast: two claims declaring it may address different
        # properties, which is not a comparison of anything.
        dimensions=["knowledge_update"],
    )


AXIS = "knowledge_update"


def side(text: str, slot: str = AXIS, evidence: tuple[str, ...] = ()) -> SideClaim:
    return SideClaim(subject="", text=text, answer_slot=slot, evidence_ids=evidence)


class TestAPairNeedsEverySubjectOnOneDimension:
    def test_one_verified_claim_per_side_forms_a_contrast(self) -> None:
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Keeping a fine-tuning approach current requires retraining."),
                side("Retrieval-augmented generation retrieves passages at query time."),
            ],
        )
        assert len(pairs) == 1
        assert set(pairs[0].subjects) == {
            "retrieval-augmented generation",
            "fine-tuning",
        }

    def test_a_missing_side_forms_no_pair(self) -> None:
        """The defect the contract was built for: supported claims about
        one of two subjects, presented as an answer to how they differ.
        An incomplete pair is not a partial contrast; it is not one."""
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Retrieval-augmented generation retrieves passages at query time."),
                side("Retrieval-augmented generation can cite its sources."),
            ],
        )
        assert pairs == ()

    def test_claims_on_different_slots_are_not_a_contrast(self) -> None:
        """Mentioning both subjects is not comparing them. This is the
        rule `_spans_every_entity` did not have: it pooled every
        published claim and asked only whether the subjects appeared
        somewhere between them."""
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights.", slot="dimension"),
                side(
                    "Retrieval-augmented generation is a kind of language model pipeline.",
                    slot="relationship",
                ),
            ],
        )
        assert pairs == ()

    def test_a_context_noun_is_not_required_in_a_pair(self) -> None:
        """ "language models" is the setting. A pair that had to mention
        it would be unformable, which is the live failure."""
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights."),
                side("Retrieval-augmented generation retrieves passages at query time."),
            ],
        )
        assert len(pairs) == 1

    def test_a_non_comparison_never_produces_pairs(self) -> None:
        definition = build_contract(
            "What is retrieval-augmented generation?",
            QuestionType.DEFINITION,
            entities=["retrieval-augmented generation"],
        )
        assert build_comparison_pairs(definition, [side("RAG retrieves passages.")]) == ()

    def test_named_dimensions_are_preferred_over_the_placeholder(self) -> None:
        c = build_contract(
            "How does A differ from B on latency?",
            QuestionType.COMPARISON,
            entities=["A", "B"],
            comparison_subjects=["A", "B"],
            dimensions=["latency"],
        )
        pairs = build_comparison_pairs(
            c,
            [
                side("A responds in ten milliseconds.", slot="latency"),
                side("B responds in forty milliseconds.", slot="latency"),
                side("A is written in Rust.", slot="dimension"),
                side("B is written in Go.", slot="dimension"),
            ],
        )
        assert next(p.dimension for p in pairs) == "latency"


class TestTheContrastIsRenderedNotAsserted:
    def test_it_renders_side_by_side(self) -> None:
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights.", evidence=("S4-e1",)),
                side("RAG retrieves passages at query time.", evidence=("S3-e2",)),
            ],
        )
        rendered = render_pairs(pairs)
        assert "| Subject |" in rendered
        assert "fine-tuning" in rendered
        assert "[S4]" in rendered

    def test_nothing_is_invented_in_the_rendering(self) -> None:
        """Every cell is a published claim's own text. A sentence like
        "X does this while Y does that" would be new prose no quote was
        checked against -- the laundering this avoids."""
        claims = [
            side("Fine-tuning updates model weights."),
            side("RAG retrieves passages at query time."),
        ]
        rendered = render_pairs(build_comparison_pairs(contract(), claims))
        for claim in claims:
            assert claim.text in rendered
        assert "while" not in rendered

    def test_no_pairs_renders_nothing(self) -> None:
        assert render_pairs(()) == ""


class TestCoverageUsesPairsForTheCoreSlot:
    def test_a_complete_pair_answers_the_question(self) -> None:
        coverage = assess_coverage(
            contract(),
            [AXIS, AXIS],
            claims=[
                side("Fine-tuning updates model weights."),
                side("RAG retrieves passages at query time."),
            ],
        )
        assert coverage.answered
        assert "direct_contrast" in coverage.satisfied

    def test_an_incomplete_pair_does_not(self) -> None:
        coverage = assess_coverage(
            contract(),
            [AXIS],
            claims=[side("RAG retrieves passages at query time.")],
        )
        assert not coverage.answered

    def test_the_pairs_are_serialised_for_the_client(self) -> None:
        coverage = assess_coverage(
            contract(),
            [AXIS, AXIS],
            claims=[
                side("Fine-tuning updates model weights.", evidence=("S4-e1",)),
                side("RAG retrieves passages at query time.", evidence=("S3-e2",)),
            ],
        )
        payload = coverage.to_dict()["comparison_pairs"]
        assert isinstance(payload, list) and len(payload) == 1
        assert payload[0]["dimension"] == AXIS
        assert len(payload[0]["sides"]) == 2

    def test_the_reporting_node_passes_slots_with_texts(self) -> None:
        """They were two lists built with different filters, so they
        were not index-aligned. Pairing requires that they are."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting.verify_citations)
        assert "claims=[" in body
        assert "SideClaim(" in body


class TestRepairCannotLaunderAnUnsupportedHalf:
    """The prohibition, tested rather than implied.

    `atomicity` is a repairable guard, so repair legitimately reduces a
    compound claim to one proposition. What must never happen is that
    the proposition it drops was the *unsupported* one, leaving a
    narrower claim presented as a repair of the original. That is
    prevented at eligibility, not at rewrite time, and this pins it.
    """

    def test_an_unsupported_proposition_makes_the_claim_unrepairable(self) -> None:
        from agentic_research.citations.repair import is_repairable

        eligible, why = is_repairable(
            "every cited quote failed a deterministic guard: atomicity",
            every_proposition_supported=False,
        )
        assert not eligible
        assert "does not support every proposition" in why

    def test_a_fully_supported_compound_claim_may_still_be_repaired(self) -> None:
        """Non-vacuity: the rule above must not be refusing everything."""
        from agentic_research.citations.repair import is_repairable

        eligible, _ = is_repairable(
            "every cited quote failed a deterministic guard: atomicity",
            every_proposition_supported=True,
        )
        assert eligible

    def test_evidence_failures_are_never_repairable(self) -> None:
        from agentic_research.citations.repair import is_repairable

        eligible, _ = is_repairable(
            "the quote does not entail the claim", every_proposition_supported=True
        )
        assert not eligible


class TestThePairsReachTheRenderedReport:
    """Computed, serialised, and then actually rendered.

    Six defects in this repository were a value derived correctly and
    never handed to the thing that needed it, and four passed every
    test. So the path is checked at each hop: coverage produces pairs,
    state carries them, the node rehydrates them, the renderer prints
    them.
    """

    def test_state_is_rehydrated_into_pairs(self) -> None:
        from agentic_research.graph.nodes.reporting import _pairs_from_state

        pairs = _pairs_from_state(
            {
                "answer_coverage": {
                    "comparison_pairs": [
                        {
                            "dimension": "latency",
                            "sides": [
                                {
                                    "subject": "A",
                                    "text": "A responds in 10ms.",
                                    "answer_slot": "latency",
                                    "evidence_ids": ["S1-e1"],
                                },
                                {
                                    "subject": "B",
                                    "text": "B responds in 40ms.",
                                    "answer_slot": "latency",
                                    "evidence_ids": ["S2-e1"],
                                },
                            ],
                        }
                    ]
                }
            }  # type: ignore[arg-type]
        )
        assert len(pairs) == 1
        assert pairs[0].dimension == "latency"
        assert pairs[0].subjects == ("A", "B")

    def test_an_absent_assessment_rehydrates_to_nothing(self) -> None:
        from agentic_research.graph.nodes.reporting import _pairs_from_state

        assert _pairs_from_state({}) == ()  # type: ignore[arg-type]
        assert _pairs_from_state({"answer_coverage": None}) == ()  # type: ignore[arg-type]

    def test_a_side_with_no_entries_is_not_a_pair(self) -> None:
        from agentic_research.graph.nodes.reporting import _pairs_from_state

        assert (
            _pairs_from_state(
                {"answer_coverage": {"comparison_pairs": [{"dimension": "d", "sides": []}]}}  # type: ignore[arg-type]
            )
            == ()
        )

    def test_the_renderer_prints_the_table(self) -> None:
        from agentic_research.models import ResearchReport
        from agentic_research.report import render_markdown

        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights.", evidence=("S4-e1",)),
                side("RAG retrieves passages at query time.", evidence=("S3-e2",)),
            ],
        )
        markdown = render_markdown(
            ResearchReport(title="T", summary_claims=[]),
            [],
            None,
            comparison_pairs=pairs,
        )
        assert "## How they differ" in markdown
        assert "Fine-tuning updates model weights." in markdown
        # The disclaimer matters: a reader must not take the table for a
        # conclusion the engine drew.
        assert "no sentence here was written" in markdown

    def test_no_pairs_prints_no_section(self) -> None:
        from agentic_research.models import ResearchReport
        from agentic_research.report import render_markdown

        markdown = render_markdown(ResearchReport(title="T"), [], None)
        assert "## How they differ" not in markdown

    def test_the_final_node_passes_them(self) -> None:
        import inspect

        from agentic_research.graph.nodes import reporting

        source = inspect.getsource(reporting)
        assert "comparison_pairs=_pairs_from_state(state)" in source


class TestTheRelationshipEscapeHatchIsNarrowed:
    """`relationship` used to discharge `direct_contrast` statically.

    The comment justifying it admitted the cost: for a genuine
    comparison of two unrelated subjects, a vague relationship claim
    discharged the core slot too. A slot name cannot express "only when
    the claim asserts a particular kind of thing", so the decision
    moved to coverage time and reads the claim.
    """

    def _llm_contract(self):
        return build_contract(
            "How does a large language model differ from a neural network?",
            QuestionType.COMPARISON,
            entities=["large language model", "neural network"],
            comparison_subjects=["large language model", "neural network"],
        )

    def test_a_subtype_relation_answers_the_comparison(self) -> None:
        """The case this exists to keep. Asked how an LLM differs from a
        neural network, the honest answer is that one is a kind of the
        other -- there is no contrast to find, and demanding one makes
        the engine wrong about itself."""
        from agentic_research.answer_coverage import assess_coverage

        coverage = assess_coverage(
            self._llm_contract(),
            ["relationship"],
            claims=[
                side(
                    "A large language model is a kind of neural network trained on text.",
                    slot="relationship",
                )
            ],
        )
        assert coverage.answered
        assert coverage.relationship_discharge == "subtype"

    def test_a_relationship_mentioning_both_without_explaining_does_not(self) -> None:
        from agentic_research.answer_coverage import assess_coverage

        coverage = assess_coverage(
            self._llm_contract(),
            ["relationship"],
            claims=[
                side(
                    "A large language model and a neural network are both widely "
                    "used in production systems.",
                    slot="relationship",
                )
            ],
        )
        assert not coverage.answered
        assert coverage.relationship_discharge == ""

    def test_a_dependency_relation_does_not_discharge(self) -> None:
        """Two things can depend on one another and still need
        contrasting. Only containment and identity remove the
        contrast."""
        from agentic_research.answer_coverage import assess_coverage
        from agentic_research.comparison import RelationshipKind, relationship_kind

        text = "Retrieval-augmented generation uses a large language model."
        assert relationship_kind(text) is RelationshipKind.DEPENDENCY
        coverage = assess_coverage(
            build_contract(
                "How does retrieval-augmented generation differ from a large language model?",
                QuestionType.COMPARISON,
                entities=["retrieval-augmented generation", "large language model"],
                comparison_subjects=[
                    "retrieval-augmented generation",
                    "large language model",
                ],
            ),
            ["relationship"],
            claims=[side(text, slot="relationship")],
        )
        assert not coverage.answered

    def test_a_subtype_claim_about_one_subject_only_does_not_discharge(self) -> None:
        """A subtype claim naming one of two subjects establishes
        nothing about the pair."""
        from agentic_research.answer_coverage import assess_coverage

        coverage = assess_coverage(
            self._llm_contract(),
            ["relationship"],
            claims=[
                side("A large language model is a kind of statistical model.", slot="relationship")
            ],
        )
        assert not coverage.answered


class TestUnrelatedFactsAreNotAComparison:
    def test_two_true_descriptions_on_the_generic_slot_do_not_pair(self) -> None:
        """The shortcut the audit rejected. Both claims are true, both
        mention their subject, and they address different properties --
        task versus input domain -- so there is no axis on which they
        meet."""
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights.", slot="dimension"),
                side("Retrieval-augmented generation reduces hallucination.", slot="dimension"),
            ],
        )
        assert pairs == ()

    def test_the_same_two_claims_pair_on_a_named_axis(self) -> None:
        """Non-vacuity: the rule is about the axis being named, not
        about these claims being unusable."""
        pairs = build_comparison_pairs(
            contract(),
            [
                side("Fine-tuning updates model weights.", slot=AXIS),
                side("Retrieval-augmented generation retrieves passages at query time.", slot=AXIS),
            ],
        )
        assert len(pairs) == 1

    def test_one_supported_side_and_one_missing_side_do_not_pair(self) -> None:
        """Only published claims reach this function, so an unsupported
        half is simply absent -- and an incomplete pair is not a partial
        contrast, it is not one."""
        pairs = build_comparison_pairs(
            contract(),
            [side("Fine-tuning updates model weights.", slot=AXIS)],
        )
        assert pairs == ()

    def test_a_pair_needs_both_sides_on_the_same_axis(self) -> None:
        """Each side on its own named axis is two half-comparisons."""
        c = build_contract(
            RAG,
            QuestionType.COMPARISON,
            entities=["retrieval-augmented generation", "fine-tuning"],
            comparison_subjects=["retrieval-augmented generation", "fine-tuning"],
            dimensions=["knowledge_update", "cost"],
        )
        pairs = build_comparison_pairs(
            c,
            [
                side(
                    "Fine-tuning requires retraining to update knowledge.", slot="knowledge_update"
                ),
                side("Retrieval-augmented generation has a lower upfront cost.", slot="cost"),
            ],
        )
        assert pairs == ()
