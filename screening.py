from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


PHASE_CONVERSATION = "conversation"
PHASE_SCALE_SELECTION = "scale_selection"
PHASE_ASSESSMENT = "assessment"
PHASE_SUMMARY = "summary"

TRANSITION_CONTINUE = "continue"
TRANSITION_GENTLE_PROMPT = "gentle_prompt"
TRANSITION_OFFER_SCALE = "offer_scale"


@dataclass(frozen=True)
class Scale:
    scale_id: str
    name: str
    description: str
    questions: List[str]
    max_item_score: int
    option_text: str


@dataclass(frozen=True)
class ScoreInterpretation:
    severity: str
    guidance: str


@dataclass
class AnswerResult:
    accepted: bool
    score: Optional[int] = None
    normalized_answer: Optional[str] = None
    message: str = ""


PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself, or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading or watching television",
    "Moving or speaking so slowly that other people could have noticed, or being so fidgety or restless that you move around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself in some way",
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

MDQ_QUESTIONS = [
    "You felt so good or so hyper that other people thought you were not your normal self, or you were so hyper that you got into trouble",
    "You were so irritable that you shouted at people or started fights or arguments",
    "You felt much more self-confident than usual",
    "You got much less sleep than usual and found you did not really miss it",
    "You were much more talkative or spoke faster than usual",
    "Thoughts raced through your head or you could not slow your mind down",
    "You were so easily distracted by things around you that you had trouble concentrating or staying on track",
    "You had much more energy than usual",
    "You were much more active or did many more things than usual",
    "You were much more social or outgoing than usual",
    "You were much more interested in sex than usual",
    "You did things that were unusual for you or that other people might have thought were excessive, foolish, or risky",
    "Spending money got you or your family into trouble",
]


SCALES: Dict[str, Scale] = {
    "phq9": Scale(
        scale_id="phq9",
        name="PHQ-9",
        description="a 9-question depression screening tool",
        questions=PHQ9_QUESTIONS,
        max_item_score=3,
        option_text="0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day",
    ),
    "gad7": Scale(
        scale_id="gad7",
        name="GAD-7",
        description="a 7-question anxiety screening tool",
        questions=GAD7_QUESTIONS,
        max_item_score=3,
        option_text="0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day",
    ),
    "mdq": Scale(
        scale_id="mdq",
        name="MDQ",
        description="a screening tool for manic or hypomanic symptoms",
        questions=MDQ_QUESTIONS,
        max_item_score=1,
        option_text="0 = no, 1 = yes",
    ),
}


FREQUENCY_PATTERNS = [
    (3, r"\b(nearly every day|almost every day|every day|daily|most days|always)\b"),
    (2, r"\b(more than half|over half|half the days|often|frequently)\b"),
    (1, r"\b(several days|some days|a few days|occasionally|sometimes|a little|rarely)\b"),
    (0, r"\b(not at all|never|none|no days|zero)\b"),
]

YES_PATTERNS = r"\b(yes|yeah|yep|true|i have|i did|definitely|probably yes)\b"
NO_PATTERNS = r"\b(no|nope|false|i have not|i haven't|not really|probably not)\b"

UNCLEAR_PATTERNS = r"\b(don't know|do not know|not sure|unsure|maybe|hard to say|can't tell|cannot tell)\b"
SUMMARY_CONFIRMATION_PATTERNS = r"\b(you think|you mean|so my answer|so you think|is that)\b"


def get_scale(scale_id: str) -> Scale:
    return SCALES[scale_id]


def extract_scale_answer(text: str, max_item_score: int = 3) -> Optional[int]:
    normalized = text.strip().lower()
    if not normalized:
        return None
    if re.search(SUMMARY_CONFIRMATION_PATTERNS, normalized):
        return None

    numeric = re.search(r"\b([0-3])\b", normalized)
    if numeric:
        value = int(numeric.group(1))
        if value <= max_item_score:
            return value

    if max_item_score == 1:
        if re.search(YES_PATTERNS, normalized):
            return 1
        if re.search(NO_PATTERNS, normalized):
            return 0
        return None

    for score, pattern in FREQUENCY_PATTERNS:
        if re.search(pattern, normalized):
            return score
    if re.search(UNCLEAR_PATTERNS, normalized):
        return None
    return None


def classify_scale_signal(text: str) -> Optional[str]:
    normalized = text.lower()
    signals = {
        "phq9": [
            "depressed", "depression", "hopeless", "worthless", "empty",
            "low mood", "low", "sad", "no interest", "can't enjoy", "cannot enjoy",
            "\u6291\u90c1", "\u4f4e\u843d", "\u7edd\u671b", "\u6ca1\u5174\u8da3",
            "\u6ca1\u6709\u5174\u8da3", "\u5f00\u5fc3\u4e0d\u8d77\u6765",
        ],
        "gad7": [
            "anxious", "anxiety", "worry", "worried", "panic", "tense",
            "nervous", "on edge", "overthinking",
            "\u7126\u8651", "\u62c5\u5fc3", "\u7d27\u5f20", "\u6050\u614c", "\u5bb3\u6015",
        ],
        "mdq": [
            "manic", "mania", "hypomanic", "too much energy", "little sleep",
            "sleep very little", "impulsive", "racing thoughts", "hyper",
            "\u8e81\u72c2", "\u8f7b\u8e81", "\u7cbe\u529b\u65fa\u76db",
            "\u7761\u5f88\u5c11", "\u51b2\u52a8", "\u601d\u7ef4\u5954\u9038",
        ],
    }
    scores = {
        scale_id: sum(1 for keyword in keywords if keyword in normalized)
        for scale_id, keywords in signals.items()
    }
    best_score = max(scores.values())
    if best_score < 1:
        return None
    best_matches = [scale_id for scale_id, score in scores.items() if score == best_score]
    if len(best_matches) > 1:
        return None
    return best_matches[0]


def is_reluctant_response(text: str) -> bool:
    normalized = text.lower().strip()
    if classify_scale_signal(normalized):
        return False
    return bool(
        re.search(
            r"\b(nothing|nothing else|don't want to talk|do not want to talk|"
            r"no more|that's all|all done|done|not much|nothing to share|no need|skip)\b",
            normalized,
        )
        or re.search(
            r"(\u4e0d\u60f3\u8bf4|\u4e0d\u60f3\u804a|\u6ca1\u4ec0\u4e48|\u6ca1\u6709\u4ec0\u4e48|"
            r"\u4e0d\u77e5\u9053\u8bf4\u4ec0\u4e48|\u7b97\u4e86|\u8df3\u8fc7|\u4e0d\u9700\u8981)",
            normalized,
        )
    )


def is_minimal_followup_response(text: str) -> bool:
    normalized = text.lower().strip()
    return normalized in {
        "?",
        "??",
        ".",
        "...",
        "ok",
        "okay",
        "sure",
        "fine",
        "yeah",
        "yes",
        "no",
        "\u55ef",
        "\u54e6",
        "\u597d",
        "\u597d\u5427",
        "\u968f\u4fbf",
    }


def choose_conversation_transition(user_turns: int, reluctance_count: int) -> str:
    if user_turns >= 15 or reluctance_count >= 2:
        return TRANSITION_OFFER_SCALE
    if reluctance_count == 1:
        return TRANSITION_GENTLE_PROMPT
    return TRANSITION_CONTINUE


def is_scale_consent_question(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(
        re.search(
            r"\b(must|have to|required|need to|do i have to|should i|is it compulsory)\b",
            normalized,
        )
        or re.search(r"(\u5fc5\u987b|\u4e00\u5b9a\u8981|\u975e\u505a|\u9700\u8981\u505a)", normalized)
    )


def is_scale_decline(text: str) -> bool:
    normalized = text.lower().strip()
    return bool(
        re.search(
            r"\b(no|not now|don't want|do not want|rather not|skip|can we not)\b",
            normalized,
        )
        or re.search(r"(\u4e0d\u60f3|\u4e0d\u8981|\u5148\u4e0d|\u7b97\u4e86|\u8df3\u8fc7)", normalized)
    )


def interpret_score(scale: Scale, score: int) -> ScoreInterpretation:
    if scale.scale_id == "phq9":
        if score <= 4:
            return ScoreInterpretation("minimal", "Your answers suggest few depression symptoms right now.")
        if score <= 9:
            return ScoreInterpretation("mild", "Your answers suggest mild depression symptoms.")
        if score <= 14:
            return ScoreInterpretation("moderate", "Your answers suggest moderate depression symptoms.")
        if score <= 19:
            return ScoreInterpretation("moderately severe", "Your answers suggest moderately severe depression symptoms.")
        return ScoreInterpretation("severe", "Your answers suggest severe depression symptoms. Please consider speaking with a qualified professional soon.")

    if scale.scale_id == "gad7":
        if score <= 4:
            return ScoreInterpretation("minimal", "Your answers suggest few anxiety symptoms right now.")
        if score <= 9:
            return ScoreInterpretation("mild", "Your answers suggest mild anxiety symptoms.")
        if score <= 14:
            return ScoreInterpretation("moderate", "Your answers suggest moderate anxiety symptoms.")
        return ScoreInterpretation("severe", "Your answers suggest severe anxiety symptoms. Please consider speaking with a qualified professional soon.")

    if score >= 7:
        return ScoreInterpretation("positive screen", "Your answers show several manic or hypomanic symptoms. This is not a diagnosis, but it would be worth discussing with a qualified clinician.")
    return ScoreInterpretation("negative screen", "Your answers do not strongly indicate manic or hypomanic symptoms on this brief screen.")


@dataclass
class ScreeningSession:
    scale_id: str
    current_index: int = 0
    answers: List[int] = field(default_factory=list)
    phase: str = PHASE_ASSESSMENT

    @property
    def scale(self) -> Scale:
        return get_scale(self.scale_id)

    @property
    def total_score(self) -> int:
        return sum(self.answers)

    def current_question_prompt(self) -> str:
        question_number = self.current_index + 1
        total = len(self.scale.questions)
        question = self.scale.questions[self.current_index]
        return (
            f"Question {question_number}/{total}: Over the last two weeks, how often have you been bothered by this: "
            f"{question}? You can answer using: {self.scale.option_text}."
        )

    def record_answer(self, text: str) -> AnswerResult:
        score = extract_scale_answer(text, self.scale.max_item_score)
        if score is None:
            return AnswerResult(
                accepted=False,
                message=(
                    "I want to score this accurately, so could you choose the closest option: "
                    f"{self.scale.option_text}?"
                ),
            )

        self.answers.append(score)
        self.current_index += 1
        if self.current_index >= len(self.scale.questions):
            self.phase = PHASE_SUMMARY
        return AnswerResult(
            accepted=True,
            score=score,
            normalized_answer=score_to_label(self.scale, score),
        )


def score_to_label(scale: Scale, score: int) -> str:
    if scale.max_item_score == 1:
        return "yes" if score == 1 else "no"
    return {
        0: "not at all",
        1: "several days",
        2: "more than half the days",
        3: "nearly every day",
    }[score]


def build_score_summary(session: ScreeningSession) -> str:
    scale = session.scale
    interpretation = interpret_score(scale, session.total_score)
    distribution = []
    for index, score in enumerate(session.answers, start=1):
        distribution.append(f"Question {index}: {score_to_label(scale, score)} ({score} point)")
    distribution_text = "; ".join(distribution)
    extra_support = ""
    if interpretation.severity in {"severe", "positive screen"}:
        extra_support = (
            " If you feel at risk of harming yourself or unable to stay safe, please contact emergency services. "
            "In the UK, Samaritans are available at 116 123, and NHS 111 can help with urgent health advice."
        )
    return (
        f"Thank you for going through the {scale.name} with me. Here is how each answer was interpreted: "
        f"{distribution_text}. Your total score is {session.total_score}. This falls in the "
        f"{interpretation.severity} range. {interpretation.guidance}{extra_support} "
        "I am a virtual mental health assistant, not a doctor, so this is not a diagnosis or a substitute for professional medical advice."
    )


def scale_intro_text(preferred_scale_id: Optional[str] = None) -> str:
    if preferred_scale_id:
        scale = get_scale(preferred_scale_id)
        return (
            f"Based on what you have shared, {scale.name} may be the most relevant next step. "
            f"It is {scale.description}. Would you like to start it now?"
        )
    return (
        "There are three brief screening tools we can use next: PHQ-9 for depression symptoms, "
        "GAD-7 for anxiety symptoms, and MDQ for manic or hypomanic symptoms. "
        "Which one would you feel most comfortable starting with?"
    )


def parse_scale_choice(text: str) -> Optional[str]:
    normalized = text.lower()
    if "phq" in normalized or "depression" in normalized or "depressed" in normalized:
        return "phq9"
    if "gad" in normalized or "anxiety" in normalized or "anxious" in normalized or "worry" in normalized:
        return "gad7"
    if "mdq" in normalized or "mania" in normalized or "manic" in normalized or "bipolar" in normalized:
        return "mdq"
    if re.search(r"\b(yes|start|begin|okay|ok|sure)\b", normalized):
        return "yes"
    return None
