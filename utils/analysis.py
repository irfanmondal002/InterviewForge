
import json
import numpy as np

from utils.embeddings import (
    split_into_passages,
    generate_embeddings,
    calculate_cosine_similarity
)


# ==================================================
# 1. EXTRACT INDIVIDUAL JD REQUIREMENTS
# ==================================================

def extract_jd_requirements(
    llm,
    jd_text
):
    """
    Extract individual job requirements using the selected LLM.
    Supports Gemini, Groq, and Ollama through LLMClient.
    """

    if not jd_text or not jd_text.strip():
        raise ValueError("Job description is empty.")

    prompt = f"""
You are an expert technical recruiter and job description analyst.

Extract individual job requirements from the job description.

JOB DESCRIPTION:
{jd_text}

INSTRUCTIONS:

1. Extract each requirement separately.
2. Include programming languages, technical skills, frameworks,
   libraries, databases, tools, and platforms.
3. Include relevant responsibilities and qualifications.
4. Distinguish mandatory and preferred requirements only when
   clearly supported by the job description.
5. Otherwise, use "unspecified" for priority.
6. Do not invent requirements.
7. Avoid duplicates.
8. Keep requirements concise and specific.
9. Preserve the original meaning.
10. Do not assess the candidate's skills.

Return valid JSON:

{{
    "job_title": "Data Engineer",
    "requirements": [
        {{
            "requirement": "Python",
            "category": "technical_skill",
            "priority": "mandatory"
        }},
        {{
            "requirement": "SQL",
            "category": "technical_skill",
            "priority": "mandatory"
        }}
    ]
}}
"""

    result = llm.generate_json(
        prompt,
        temperature=0.1
    )

    requirements = result.get("requirements", [])

    if not isinstance(requirements, list):
        raise ValueError("Invalid requirements format.")

    cleaned_requirements = []
    seen = set()

    for item in requirements:

        if not isinstance(item, dict):
            continue

        name = str(
            item.get("requirement", "")
        ).strip()

        if not name or name.casefold() in seen:
            continue

        seen.add(name.casefold())

        cleaned_requirements.append({
            "requirement": name,
            "category": str(
                item.get("category", "other")
            ),
            "priority": str(
                item.get("priority", "unspecified")
            )
        })

    if not cleaned_requirements:
        raise ValueError(
            "No valid requirements were extracted."
        )

    return {
        "job_title": result.get("job_title"),
        "requirements": cleaned_requirements
    }


# ==================================================
# 2. COMPARE REQUIREMENTS WITH RESUME PASSAGES
# ==================================================

def compare_requirements_with_resume(
    requirements,
    resume_text,
    top_k=3
):
    """
    Generate embeddings for JD requirements and resume passages.
    Retrieve relevant resume evidence using cosine similarity.
    """

    resume_passages = split_into_passages(resume_text)

    if not resume_passages:
        raise ValueError("No resume passages found.")

    requirement_texts = [
        item["requirement"]
        for item in requirements
    ]

    # Generate local embeddings.
    requirement_embeddings = generate_embeddings(
        requirement_texts
    )

    resume_embeddings = generate_embeddings(
        resume_passages
    )

    # Calculate cosine similarity.
    similarity_matrix = calculate_cosine_similarity(
        requirement_embeddings,
        resume_embeddings
    )

    comparison_results = []

    for index, requirement in enumerate(requirements):

        scores = similarity_matrix[index]

        ranked_indices = np.argsort(scores)[::-1]

        evidence = []

        for passage_index in ranked_indices[:top_k]:

            score = float(scores[passage_index])

            evidence.append({
                "passage": resume_passages[
                    int(passage_index)
                ],
                "similarity_score": round(score, 4)
            })

        best_score = float(
            scores[ranked_indices[0]]
        )

        comparison_results.append({

            "requirement": requirement["requirement"],

            "category": requirement["category"],

            "priority": requirement["priority"],

            "best_similarity_score": round(
                best_score,
                4
            ),

            "evidence": evidence

        })

    return comparison_results


# ==================================================
# 3. GENERATE SKILL GAP REPORT
# ==================================================

def generate_skill_gap_report(
    llm,
    comparisons
):
    """
    Interpret the similarity results and resume evidence
    to produce an evidence-based skill-gap report.
    """

    comparison_text = json.dumps(
        comparisons,
        ensure_ascii=False
    )

    prompt = f"""
You are an expert technical interviewer and career advisor.

Analyze the comparison between job requirements
and the candidate's resume passages.

COMPARISON DATA:

{comparison_text}

INSTRUCTIONS:

1. Assess every requirement using actual resume evidence.
2. Similarity scores indicate semantic relevance,
   not proven proficiency.
3. Use "matched" when clear supporting evidence exists.
4. Use "partially_matched" when evidence is related
   but incomplete.
5. Use "no_evidence_found" when no meaningful evidence exists.
6. Use "uncertain" when the evidence is ambiguous.
7. No evidence does not prove the candidate lacks a skill.
8. Never invent candidate experience.
9. Explain each assessment briefly.
10. Suggest practical improvements.

Return only valid JSON:

{{
    "summary": "Overall candidate assessment.",
    "requirements_analysis": [
        {{
            "requirement": "Python",
            "status": "matched",
            "reasoning": "Python is mentioned in a project.",
            "recommended_action": "Revise Python fundamentals."
        }}
    ],
    "recommendations": [
        "Practice Python programming.",
        "Improve SQL knowledge."
    ]
}}
"""

    result = llm.generate_json(
        prompt,
        temperature=0.2
    )

    model_results = result.get(
        "requirements_analysis",
        []
    )

    if not isinstance(model_results, list):
        model_results = []

    model_map = {
        str(item.get("requirement", "")).casefold(): item
        for item in model_results
        if isinstance(item, dict)
    }

    valid_statuses = {
        "matched",
        "partially_matched",
        "no_evidence_found",
        "uncertain"
    }

    final_results = []

    for comparison in comparisons:

        requirement_name = comparison["requirement"]

        model_result = model_map.get(
            requirement_name.casefold(),
            {}
        )

        status = str(
            model_result.get("status", "uncertain")
        ).lower()

        if status not in valid_statuses:
            status = "uncertain"

        final_results.append({

            **comparison,

            "status": status,

            "reasoning": model_result.get(
                "reasoning",
                "No explanation was returned."
            ),

            "recommended_action": model_result.get(
                "recommended_action",
                "Review this requirement."
            )

        })

    return {
        "summary": result.get(
            "summary",
            "Analysis completed."
        ),
        "requirements_analysis": final_results,
        "recommendations": result.get(
            "recommendations",
            []
        )
    }


# ==================================================
# 4. COMPLETE RESUME-JD ANALYSIS PIPELINE
# ==================================================

def analyze_resume_against_jd(
    llm,
    resume_text,
    jd_text
):
    """
    Run the complete JD-to-resume analysis pipeline.
    """

    extracted = extract_jd_requirements(
        llm,
        jd_text
    )

    comparisons = compare_requirements_with_resume(
        extracted["requirements"],
        resume_text
    )

    report = generate_skill_gap_report(
        llm,
        comparisons
    )

    report["job_title"] = (
        extracted.get("job_title")
    )

    report["extracted_requirements"] = (
        extracted["requirements"]
    )

    return report


# ==================================================
# 5. CREATE COMPACT ANALYSIS CONTEXT
# ==================================================

def build_compact_analysis_context(
    analysis,
    max_requirements=15,
    max_evidence_per_requirement=2
):
    """
    Reduce prompt size by using relevant analysis data
    instead of repeatedly sending complete documents.
    """

    requirements_analysis = analysis.get(
        "requirements_analysis",
        []
    )

    compact_results = []

    for item in requirements_analysis[:max_requirements]:

        evidence_list = []

        for evidence in item.get(
            "evidence",
            []
        )[:max_evidence_per_requirement]:

            if isinstance(evidence, dict):

                passage = str(
                    evidence.get("passage", "")
                )[:450]

                evidence_list.append({
                    "passage": passage,
                    "similarity_score": evidence.get(
                        "similarity_score"
                    )
                })

            elif isinstance(evidence, str):

                evidence_list.append(
                    evidence[:450]
                )

        compact_results.append({

            "requirement": item.get("requirement"),

            "category": item.get("category"),

            "priority": item.get("priority"),

            "status": item.get("status"),

            "reasoning": str(
                item.get("reasoning", "")
            )[:300],

            "recommended_action": str(
                item.get("recommended_action", "")
            )[:250],

            "resume_evidence": evidence_list

        })

    return {
        "job_title": analysis.get("job_title"),
        "summary": str(
            analysis.get("summary", "")
        )[:600],
        "requirements_analysis": compact_results,
        "recommendations": analysis.get(
            "recommendations",
            []
        )[:8]
    }


# ==================================================
# 6. GENERATE INTERVIEW QUESTIONS
# ==================================================

def generate_interview_questions(
    llm,
    jd_text,
    resume_text,
    analysis,
    max_questions=8
):
    """
    Generate personalized interview questions using
    compact resume evidence and skill-gap analysis.

    The model is asked to generate up to max_questions.
    """

    max_questions = max(
        1,
        min(int(max_questions), 12)
    )

    compact_analysis = build_compact_analysis_context(
        analysis,
        max_requirements=15,
        max_evidence_per_requirement=2
    )

    # Extract only a small JD reference.
    jd_reference = jd_text[:3500]

    # Extract only a small resume reference.
    # Relevant resume evidence is already included
    # in compact_analysis.
    resume_reference = resume_text[:1500]

    prompt = f"""
You are an experienced technical interviewer.

Generate up to {max_questions} personalized interview
questions for this candidate.

JOB TITLE:

{analysis.get("job_title", "Not specified")}

JOB DESCRIPTION REFERENCE:

{jd_reference}

RESUME REFERENCE:

{resume_reference}

SKILL GAP ANALYSIS AND RELEVANT RESUME EVIDENCE:

{json.dumps(compact_analysis, ensure_ascii=False)}

INSTRUCTIONS:

1. Generate up to {max_questions} useful questions.
2. Prioritize mandatory job requirements.
3. Include questions about relevant resume projects.
4. Include questions addressing partially matched skills.
5. Include questions about important requirements
   for which no resume evidence was found.
6. Include technical and practical questions.
7. Include behavioral questions when relevant.
8. Avoid duplicate and generic questions.
9. Never invent candidate experience.
10. Keep every answer outline within 3 sentences.
11. Use Easy, Medium, or Hard difficulty.
12. Return fewer questions rather than repeating questions
    to meet the requested number.

Return only valid JSON:

{{
    "questions": [
        {{
            "question": "Explain how Python dictionaries work.",
            "topic": "Python",
            "difficulty": "Easy",
            "why_asked": "Tests Python fundamentals.",
            "answer_outline": "Explain key-value storage, hashing, and average-case lookup."
        }}
    ]
}}
"""

    result = llm.generate_json(
        prompt,
        temperature=0.2
    )

    questions = result.get(
        "questions",
        []
    )

    if not isinstance(questions, list):
        raise ValueError(
            "Invalid interview question format returned by the LLM."
        )

    # Remove invalid or duplicate questions.
    cleaned_questions = []
    seen_questions = set()

    for item in questions:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        if not question:
            continue

        if question.casefold() in seen_questions:
            continue

        seen_questions.add(question.casefold())

        cleaned_questions.append({

            "question": question,

            "topic": str(
                item.get("topic", "General")
            ),

            "difficulty": str(
                item.get("difficulty", "Medium")
            ),

            "why_asked": str(
                item.get("why_asked", "")
            ),

            "answer_outline": str(
                item.get("answer_outline", "")
            )

        })

    if not cleaned_questions:

        raise ValueError(
            "The AI provider did not generate valid questions."
        )

    result["questions"] = cleaned_questions[:max_questions]

    return result


# ==================================================
# 7. GENERATE PERSONALIZED STUDY PLAN
# ==================================================

def generate_study_plan(
    llm,
    jd_text,
    analysis,
    days,
    hours_per_day
):
    """
    Generate a personalized study plan using
    compact skill-gap information.
    """

    days = max(3, min(int(days), 30))

    hours_per_day = max(
        1,
        min(int(hours_per_day), 10)
    )

    compact_analysis = build_compact_analysis_context(
        analysis,
        max_requirements=15,
        max_evidence_per_requirement=1
    )

    prompt = f"""
You are an expert interview preparation coach.

Create a practical study plan for a fresher candidate.

JOB TITLE:

{analysis.get("job_title", "Not specified")}

DURATION:

{days} days

AVAILABLE STUDY TIME:

{hours_per_day} hours per day

JOB DESCRIPTION REFERENCE:

{jd_text[:3000]}

SKILL GAP ANALYSIS:

{json.dumps(compact_analysis, ensure_ascii=False)}

INSTRUCTIONS:

1. Create a plan for {days} days.
2. Never exceed {hours_per_day} hours per day.
3. Prioritize mandatory job requirements.
4. Address partially matched and no-evidence-found areas.
5. Include technical concepts and practical coding.
6. Include project revision.
7. Include revision and mock interview practice.
8. Keep daily tasks realistic.
9. Do not assume the candidate already knows a skill
   unless supported by the analysis.
10. Avoid repeating the same tasks unnecessarily.

Return only valid JSON:

{{
    "days": [
        {{
            "day": 1,
            "focus": "Python fundamentals",
            "hours": 3,
            "tasks": [
                "Revise Python data types.",
                "Practice coding problems.",
                "Solve interview questions."
            ]
        }}
    ]
}}
"""

    result = llm.generate_json(
        prompt,
        temperature=0.2
    )

    plan_days = result.get(
        "days",
        []
    )

    if not isinstance(plan_days, list):
        raise ValueError(
            "Invalid study plan format."
        )

    # Avoid displaying more days than requested.
    result["days"] = plan_days[:days]

    return result
