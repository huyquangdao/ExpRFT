ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant who is an expert at answering multiple-choice questions.\n"
    "You MUST first state your final answer in the format: ’The answer is X’ where X is A, B, C,\n"
    "or D. After the final answer clearly explain your reasoning.\n"
)

TEACHER_TURN1 = (
    "Question:  {question}\n"
    "Choices:\n"
    "{choices}\n"
    "\n"
    "{dataset_instruction}\n"
    "\n"
    "Please think through this step by step and explain your initial thoughts about the question."
)

TEACHER_TURN2 = (
    "That's a good start. Can you explain more about why some options might be incorrect? "
    "Also, are there any key concepts or facts that are relevant to answering this question?"
)

TEACHER_FINAL = (
    "Thank you for your detailed explanations. What is your final answer (A, B, C, or D)? "
    "You MUST first state your final answer in the format: 'The answer is X' where X is A, B, C, or D. "
    "After the final answer clearly explain your reasoning."
)

TEACHER_INTERMEDIATE = (
    "Thank you for your explanation. Let's explore this further (turn {turn_num}). "
    "Could you elaborate on the key concepts relevant to this question?"
)

DATASET_INSTRUCTIONS = {
    "openbookqa": "This question is from a science exam.",
    "arc": "This question is from a science exam.",
    "sciqa": "This question is from a science exam.",
    "cosql": (
        "This question asks for the correct SQL query to retrieve the requested information "
        "from a database. Focus on SQL syntax, correct table joins, and query efficiency."
    ),
    "mathdial": (
        "This is a math tutoring problem. "
        "Analyze the student's reasoning carefully to identify any errors."
    ),
}


def dataset_instruction(context: dict) -> str:
    if context.get("dataset") == "mmlu":
        subject = context.get("metadata", {}).get("subject", "")
        if subject:
            return f"This question is from a standardized academic test on {subject}."
        return "This question is from a standardized academic test."
    return DATASET_INSTRUCTIONS.get(context.get("dataset", ""), "")


def format_choices(choices: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in choices.items())


def teacher_prompt(turn: int, context: dict, num_turns: int = 3) -> str:
    """Scripted teacher utterance for 1-indexed ``turn``."""
    if turn == 1:
        return TEACHER_TURN1.format(
            question=context["question"],
            choices=format_choices(context["choices"]),
            dataset_instruction=dataset_instruction(context),
        )
    if turn == num_turns:
        return TEACHER_FINAL
    if turn == 2:
        return TEACHER_TURN2
    return TEACHER_INTERMEDIATE.format(turn_num=turn)


# --------------------------------------------------------------------------- #
# Judge                                                                        #
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of educational conversations about multiple-choice questions.
Your job is to rate the quality and progress of a conversation on a scale from 0 to 10 for
several metrics. For each metric, you MUST provide a numerical score (0-10) and a brief
justification of 1-2 sentences.
The metrics to evaluate are:
    1. Accuracy - Is the assistant moving toward the correct answer?
    2. Reasoning ability - Is the reasoning logical, clear, and precise?
    3. Comprehensiveness - Are key concepts being properly explained?
    4. Pedagogical value - Is this explanation helping someone learn?
    5. Confidence calibration - Is the assistant appropriately confident in their analysis?

After rating each metric, provide an overall score (0-10) that represents your holistic
assessment of the conversation, followed by a brief overall justification.

IMPORTANT: You MUST format your response EXACTLY as follows with these exact labels
and line breaks:
Accuracy:  [score]
[justification]

Reasoning ability:  [score]
[justification]

Comprehensiveness:  [score]
[justification]

Pedagogical value:  [score]
[justification]

Confidence calibration:  [score]
[justification]

Overall:  [score]
[justification]\
"""

JUDGE_USER_PROMPT = """\
Please evaluate this educational conversation about a multiple-choice question up to this point.
Original Question: {question}
Choices: {choices}
Correct Answer: {correct_answer}
--- CONVERSATION TRANSCRIPT SO FAR ---
{transcript}
--- END CONVERSATION TRANSCRIPT ---
Please provide your evaluation following the exact format specified in your system prompt.\
"""


def judge_prompt(context: dict, transcript: str) -> tuple[str, str]:
    choices = ", ".join(f"{k}: {v}" for k, v in context["choices"].items())
    user = JUDGE_USER_PROMPT.format(
        question=context["question"],
        choices=choices,
        correct_answer=context["correct_answer"],
        transcript=transcript,
    )
    return JUDGE_SYSTEM_PROMPT, user
