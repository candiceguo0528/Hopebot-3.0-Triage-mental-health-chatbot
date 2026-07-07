from screening import (
    PHASE_ASSESSMENT,
    PHASE_CONVERSATION,
    PHASE_SUMMARY,
    choose_conversation_transition,
    ScreeningSession,
    classify_scale_signal,
    extract_scale_answer,
    get_scale,
    is_minimal_followup_response,
    is_reluctant_response,
    is_scale_consent_question,
    is_scale_decline,
    interpret_score,
)


def test_extracts_scale_answers_from_natural_language():
    assert extract_scale_answer("Nearly every day, honestly.") == 3
    assert extract_scale_answer("I think more than half the days") == 2
    assert extract_scale_answer("several days maybe") == 1
    assert extract_scale_answer("not at all") == 0
    assert extract_scale_answer("2") == 2


def test_does_not_score_unclear_answers_or_unconfirmed_summaries():
    assert extract_scale_answer("I don't know how to answer that") is None
    assert extract_scale_answer("maybe yes and no") is None
    assert extract_scale_answer("so you think my answer is several days?") is None


def test_screening_session_advances_questions_and_scores_in_order():
    session = ScreeningSession(scale_id="phq9")

    first_prompt = session.current_question_prompt()
    assert "1/9" in first_prompt

    result = session.record_answer("several days")
    assert result.accepted is True
    assert result.score == 1
    assert session.current_index == 1
    assert session.phase == PHASE_ASSESSMENT

    for _ in range(8):
        session.record_answer("not at all")

    assert session.total_score == 1
    assert session.phase == PHASE_SUMMARY
    assert len(session.answers) == 9


def test_scale_signal_recommends_clear_tendency_and_keeps_unclear_neutral():
    assert classify_scale_signal("I feel hopeless, low, and cannot enjoy anything") == "phq9"
    assert classify_scale_signal("I worry all the time and feel tense every day") == "gad7"
    assert classify_scale_signal("Sometimes I sleep very little but feel full of energy and impulsive") == "mdq"
    assert classify_scale_signal("\u6211\u6700\u8fd1\u597d\u6291\u90c1") == "phq9"
    assert classify_scale_signal("\u6211\u6700\u8fd1\u597d\u7126\u8651") == "gad7"
    assert classify_scale_signal("I feel depressed and anxious") is None
    assert classify_scale_signal("I just feel strange lately") is None


def test_short_symptom_disclosure_is_not_treated_as_reluctance():
    assert is_reluctant_response("hi") is False
    assert is_reluctant_response("hello") is False
    assert is_reluctant_response("I feel depressed") is False
    assert is_reluctant_response("\u6211\u5f88\u7126\u8651") is False
    assert is_reluctant_response("thank you") is False
    assert is_reluctant_response("nothing to share") is True
    assert is_reluctant_response("I don't want to talk") is True
    assert is_reluctant_response("all done") is True
    assert is_reluctant_response("that's all") is True
    assert is_reluctant_response("\u4e0d\u60f3\u8bf4") is True
    assert is_reluctant_response("\u6ca1\u4ec0\u4e48\u60f3\u8bf4\u7684") is True
    assert is_minimal_followup_response("\u55ef") is True
    assert is_minimal_followup_response("?") is True
    assert is_minimal_followup_response("okay") is True
    assert is_minimal_followup_response("I feel a bit better") is False


def test_reluctant_user_gets_one_guidance_turn_before_scale_offer():
    assert choose_conversation_transition(user_turns=1, reluctance_count=1) == "gentle_prompt"
    assert choose_conversation_transition(user_turns=2, reluctance_count=2) == "offer_scale"
    assert choose_conversation_transition(user_turns=14, reluctance_count=0) == "continue"
    assert choose_conversation_transition(user_turns=15, reluctance_count=0) == "offer_scale"


def test_scale_selection_handles_consent_questions_and_declines():
    assert is_scale_consent_question("i must do it?") is True
    assert is_scale_consent_question("do I have to take this test?") is True
    assert is_scale_decline("I don't want to do a test") is True
    assert is_scale_decline("not now") is True


def test_interpretation_uses_scale_specific_ranges():
    assert interpret_score(get_scale("phq9"), 22).severity == "severe"
    assert interpret_score(get_scale("gad7"), 16).severity == "severe"
    assert interpret_score(get_scale("mdq"), 7).severity == "positive screen"
