
import json
import time
import streamlit as st

from utils.llm import (
    LLMClient,
    get_configured_providers,
    list_available_models,
    DEFAULT_MODELS
)

from utils.database import (
    init_db,
    create_application,
    save_analysis,
    save_generated_result,
    get_application,
    list_applications,
    save_chat_message,
    get_chat_history
)

from utils.document_loader import extract_document_text

from utils.analysis import (
    extract_jd_requirements,
    compare_requirements_with_resume,
    generate_skill_gap_report,
    generate_interview_questions,
    generate_study_plan
)

from utils.rag import (
    index_application_documents,
    answer_question
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="InterviewForge",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    [data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    h1, h2, h3 {
        color: #202123;
        font-weight: 650;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d1d5db;
        min-height: 42px;
        font-weight: 500;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    .if-muted {
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3. INITIALIZE DATABASE
# ==================================================

init_db()


# ==================================================
# 4. CACHED RESOURCES
# ==================================================

@st.cache_resource(show_spinner=False)
def get_llm_client(provider, model):

    return LLMClient(
        provider=provider,
        model=model
    )


@st.cache_data(
    ttl=300,
    show_spinner=False
)
def get_cached_models(provider):

    return list_available_models(provider)


# ==================================================
# 5. SESSION STATE
# ==================================================

if "active_application_id" not in st.session_state:

    st.session_state.active_application_id = None

if "generation_info" not in st.session_state:

    st.session_state.generation_info = {}


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 🎯 InterviewForge")

    st.caption(
        "Your personal AI interview preparation assistant."
    )

    st.divider()

    # ----------------------------------------------
    # NAVIGATION
    # ----------------------------------------------

    st.markdown("### Navigation")

    page = st.radio(
        "Navigate",
        [
            "New Analysis",
            "Workspace",
            "Interview Prep",
            "Chat"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    # ----------------------------------------------
    # SAVED APPLICATIONS
    # ----------------------------------------------

    st.markdown("### Saved Applications")

    applications = list_applications()

    if applications:

        app_options = {}

        for application in applications:

            label = (
                f"{application['job_title']} "
                f"· #{application['id']}"
            )

            app_options[label] = application["id"]

        current_id = st.session_state.active_application_id

        selected_index = 0

        for index, app_id in enumerate(
            app_options.values()
        ):

            if app_id == current_id:

                selected_index = index
                break

        selected_label = st.selectbox(
            "Open application",
            options=list(app_options.keys()),
            index=selected_index,
            key="saved_application_selector"
        )

        st.session_state.active_application_id = (
            app_options[selected_label]
        )

    else:

        st.caption(
            "Your saved applications will appear here."
        )

    st.divider()

    # ----------------------------------------------
    # AI PROVIDER SELECTOR
    # ----------------------------------------------

    st.markdown("### AI Model Settings")

    available_providers = get_configured_providers()

    if not available_providers:

        st.error(
            "No AI provider is configured. "
            "Check your .env file."
        )

        st.stop()

    selected_provider = st.selectbox(
        "Choose AI provider",
        options=available_providers,
        key="selected_provider"
    )

    # ----------------------------------------------
    # MODEL DISCOVERY
    # ----------------------------------------------

    with st.spinner("Loading available models..."):

        available_models = get_cached_models(
            selected_provider
        )

    if not available_models:

        st.error(
            f"No models found for {selected_provider}."
        )

        st.stop()

    default_model = DEFAULT_MODELS.get(
        selected_provider,
        ""
    )

    default_index = (
        available_models.index(default_model)
        if default_model in available_models
        else 0
    )

    selected_model = st.selectbox(
        "Choose model",
        options=available_models,
        index=default_index,
        key=f"selected_model_{selected_provider}"
    )

    # ----------------------------------------------
    # REFRESH MODELS
    # ----------------------------------------------

    if st.button(
        "Refresh model list",
        use_container_width=True
    ):

        get_cached_models.clear()

        st.rerun()

    st.caption(
        f"Provider: {selected_provider}"
    )

    st.caption(
        f"Model: {selected_model}"
    )

    st.divider()

    st.caption("Embeddings: Sentence Transformers")
    st.caption("Vector database: ChromaDB")
    st.caption("Storage: SQLite")


# ==================================================
# 7. INITIALIZE SELECTED LLM
# ==================================================

llm = get_llm_client(
    selected_provider,
    selected_model
)


# ==================================================
# 8. HELPER FUNCTIONS
# ==================================================

def get_active_application():

    application_id = (
        st.session_state.active_application_id
    )

    if application_id is None:
        return None

    return get_application(application_id)


def load_analysis(application):

    if not application:
        return {}

    analysis_json = application.get(
        "analysis_json"
    )

    if not analysis_json:
        return {}

    try:

        return json.loads(analysis_json)

    except json.JSONDecodeError:

        return {}


def save_generation_info(
    application_id,
    feature,
    elapsed_time
):

    st.session_state.generation_info[
        f"{application_id}_{feature}"
    ] = {

        "provider": llm.last_provider,

        "model": llm.last_model,

        "time": round(elapsed_time, 2)

    }


def show_generation_info(
    application_id,
    feature
):

    key = f"{application_id}_{feature}"

    info = st.session_state.generation_info.get(key)

    if not info:
        return

    st.caption(
        f"Generated using {info['provider']} "
        f"· {info['model']} "
        f"· {info['time']} seconds"
    )

    if info["provider"] != selected_provider:

        st.warning(
            f"The selected provider ({selected_provider}) "
            f"was unavailable. InterviewForge switched to "
            f"{info['provider']}."
        )


# ==================================================
# 9. NEW ANALYSIS PAGE
# ==================================================

if page == "New Analysis":

    st.title("Let's get you interview-ready.")

    st.markdown(
        """
        Upload your resume and job description.

        InterviewForge will analyze the requirements,
        identify relevant resume evidence, and prepare
        a personalized interview report.
        """
    )

    st.divider()

    with st.form("analysis_form"):

        job_title = st.text_input(
            "Job title",
            placeholder="Example: Data Engineer"
        )

        resume_file = st.file_uploader(
            "Upload your resume",
            type=["pdf", "docx", "txt"],
            key="resume_upload"
        )

        jd_file = st.file_uploader(
            "Upload the job description",
            type=["pdf", "docx", "txt"],
            key="jd_upload"
        )

        submitted = st.form_submit_button(
            "Analyze My Resume",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not resume_file or not jd_file:

            st.warning(
                "Please upload both your resume and job description."
            )

        else:

            start_time = time.perf_counter()

            try:

                with st.status(
                    "Analyzing your documents...",
                    expanded=True
                ) as status:

                    # Step 1: Document extraction.

                    st.write(
                        "Step 1: Extracting document text..."
                    )

                    resume_text = extract_document_text(
                        resume_file
                    )

                    jd_text = extract_document_text(
                        jd_file
                    )

                    # Step 2: Extract JD requirements.

                    st.write(
                        "Step 2: Extracting individual JD requirements..."
                    )

                    extracted = extract_jd_requirements(
                        llm,
                        jd_text
                    )

                    # Step 3: Compare resume passages.

                    st.write(
                        "Step 3: Generating embeddings and comparing passages..."
                    )

                    comparisons = compare_requirements_with_resume(
                        extracted["requirements"],
                        resume_text
                    )

                    # Step 4: Generate skill-gap report.

                    st.write(
                        "Step 4: Generating skill-gap report..."
                    )

                    report = generate_skill_gap_report(
                        llm,
                        comparisons
                    )

                    report["job_title"] = (
                        job_title.strip()
                        or extracted.get("job_title")
                        or "Untitled Role"
                    )

                    report["extracted_requirements"] = (
                        extracted["requirements"]
                    )

                    # Step 5: Save application.

                    st.write(
                        "Step 5: Saving application..."
                    )

                    application_id = create_application(
                        report["job_title"],
                        resume_file.name,
                        jd_file.name,
                        resume_text,
                        jd_text
                    )

                    save_analysis(
                        application_id,
                        report
                    )

                    # Step 6: Index documents.

                    st.write(
                        "Step 6: Indexing documents in ChromaDB..."
                    )

                    try:

                        indexed_count = index_application_documents(
                            application_id,
                            resume_text,
                            jd_text,
                            report
                        )

                    except Exception as index_error:

                        indexed_count = 0

                        st.warning(
                            "Analysis was saved, but ChromaDB "
                            f"indexing failed: {index_error}"
                        )

                    st.session_state.active_application_id = (
                        application_id
                    )

                    status.update(
                        label="Analysis completed successfully!",
                        state="complete",
                        expanded=False
                    )

                elapsed_time = round(
                    time.perf_counter() - start_time,
                    2
                )

                st.success(
                    f"Analysis completed! Application ID: {application_id}"
                )

                st.caption(
                    f"Analysis time: {elapsed_time} seconds"
                )

                st.caption(
                    f"LLM provider: {llm.last_provider} "
                    f"· {llm.last_model}"
                )

                st.caption(
                    f"ChromaDB passages indexed: {indexed_count}"
                )

                st.info(
                    "Open Workspace to view your skill-gap report."
                )

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


# ==================================================
# 10. WORKSPACE PAGE
# ==================================================

elif page == "Workspace":

    st.title("Your Workspace")

    application = get_active_application()

    if application is None:

        st.info(
            "Create an analysis first to view your workspace."
        )

    else:

        analysis = load_analysis(application)

        if not analysis:

            st.info(
                "No skill-gap report is available for this application."
            )

        else:

            st.subheader(
                application["job_title"]
            )

            st.caption(
                f"Resume: {application['resume_filename']} "
                f"| Job Description: {application['jd_filename']}"
            )

            st.divider()

            # Overall assessment.

            st.markdown("### Overall Assessment")

            st.write(
                analysis.get("summary", "")
            )

            requirements = analysis.get(
                "requirements_analysis",
                []
            )

            matched = sum(
                1 for item in requirements
                if item.get("status") == "matched"
            )

            partial = sum(
                1 for item in requirements
                if item.get("status") == "partially_matched"
            )

            missing = sum(
                1 for item in requirements
                if item.get("status") == "no_evidence_found"
            )

            uncertain = sum(
                1 for item in requirements
                if item.get("status") == "uncertain"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Matched", matched)
            col2.metric("Partially Matched", partial)
            col3.metric("No Evidence", missing)
            col4.metric("Uncertain", uncertain)

            st.divider()

            # Requirement-by-requirement analysis.

            st.markdown(
                "### Requirement-by-Requirement Analysis"
            )

            for item in requirements:

                requirement = item.get(
                    "requirement",
                    "Unnamed requirement"
                )

                status_text = item.get(
                    "status",
                    "uncertain"
                )

                with st.expander(
                    f"{requirement} · {status_text}"
                ):

                    st.write(
                        f"**Category:** {item.get('category', 'N/A')}"
                    )

                    st.write(
                        f"**Priority:** {item.get('priority', 'N/A')}"
                    )

                    st.write(
                        "**Cosine similarity:** "
                        f"{item.get('best_similarity_score', 'N/A')}"
                    )

                    st.write(
                        f"**Assessment:** {item.get('reasoning', '')}"
                    )

                    st.write(
                        "**Recommended action:** "
                        f"{item.get('recommended_action', '')}"
                    )

                    evidence = item.get(
                        "evidence",
                        []
                    )

                    if evidence:

                        st.markdown(
                            "**Relevant resume passages**"
                        )

                        for passage in evidence:

                            if isinstance(passage, dict):

                                st.write(
                                    passage.get("passage", "")
                                )

                                st.caption(
                                    "Similarity: "
                                    f"{passage.get('similarity_score', 'N/A')}"
                                )

                            else:

                                st.write(passage)

            st.divider()

            st.markdown("### Recommendations")

            for recommendation in analysis.get(
                "recommendations",
                []
            ):

                st.write(
                    f"- {recommendation}"
                )

            st.download_button(
                "Download Skill Gap Report",
                data=json.dumps(
                    analysis,
                    indent=2,
                    ensure_ascii=False
                ),
                file_name="interviewforge_report.json",
                mime="application/json"
            )


# ==================================================
# 11. INTERVIEW PREPARATION PAGE
# ==================================================

elif page == "Interview Prep":

    st.title("Interview Preparation")

    application = get_active_application()

    if application is None:

        st.info(
            "Run a resume analysis before generating "
            "interview preparation materials."
        )

    else:

        question_tab, plan_tab = st.tabs(
            [
                "Interview Questions",
                "Study Plan"
            ]
        )

        analysis = load_analysis(application)

        # ------------------------------------------
        # INTERVIEW QUESTIONS
        # ------------------------------------------

        with question_tab:

            st.subheader(
                "Personalized Interview Questions"
            )

            question_count = st.select_slider(
                "Number of questions",
                options=[4, 6, 8, 10, 12],
                value=8
            )

            if st.button(
                "Generate Interview Questions",
                type="primary",
                use_container_width=True
            ):

                start_time = time.perf_counter()

                try:

                    with st.spinner(
                        f"Generating up to {question_count} questions..."
                    ):

                        questions = generate_interview_questions(
                            llm,
                            application["jd_text"],
                            application["resume_text"],
                            analysis,
                            max_questions=question_count
                        )

                        save_generated_result(
                            application["id"],
                            questions,
                            "questions_json"
                        )

                    elapsed_time = (
                        time.perf_counter() - start_time
                    )

                    save_generation_info(
                        application["id"],
                        "questions",
                        elapsed_time
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Question generation failed: {error}"
                    )

            # Reload application after generation.

            application = get_application(
                application["id"]
            )

            show_generation_info(
                application["id"],
                "questions"
            )

            questions_json = application.get(
                "questions_json"
            )

            if questions_json:

                questions_data = json.loads(
                    questions_json
                )

                questions = questions_data.get(
                    "questions",
                    []
                )

                st.caption(
                    f"{len(questions)} interview questions available."
                )

                for index, item in enumerate(
                    questions,
                    start=1
                ):

                    with st.expander(
                        f"Q{index}. {item.get('question', '')}"
                    ):

                        st.write(
                            f"**Topic:** {item.get('topic', 'General')}"
                        )

                        st.write(
                            f"**Difficulty:** "
                            f"{item.get('difficulty', 'Medium')}"
                        )

                        st.write(
                            f"**Why asked:** "
                            f"{item.get('why_asked', '')}"
                        )

                        st.markdown("**Answer outline**")

                        st.write(
                            item.get("answer_outline", "")
                        )

                st.download_button(
                    "Download Interview Questions",
                    data=json.dumps(
                        questions_data,
                        indent=2,
                        ensure_ascii=False
                    ),
                    file_name="interview_questions.json",
                    mime="application/json"
                )

        # ------------------------------------------
        # PERSONALIZED STUDY PLAN
        # ------------------------------------------

        with plan_tab:

            st.subheader(
                "Personalized Study Plan"
            )

            days = st.slider(
                "Plan duration",
                min_value=3,
                max_value=30,
                value=7,
                key="plan_days"
            )

            hours_per_day = st.slider(
                "Study hours per day",
                min_value=1,
                max_value=10,
                value=3,
                key="plan_hours"
            )

            if st.button(
                "Generate Study Plan",
                type="primary",
                use_container_width=True
            ):

                start_time = time.perf_counter()

                try:

                    with st.spinner(
                        "Creating your personalized study plan..."
                    ):

                        plan = generate_study_plan(
                            llm,
                            application["jd_text"],
                            analysis,
                            days,
                            hours_per_day
                        )

                        save_generated_result(
                            application["id"],
                            plan,
                            "study_plan_json"
                        )

                    elapsed_time = (
                        time.perf_counter() - start_time
                    )

                    save_generation_info(
                        application["id"],
                        "study_plan",
                        elapsed_time
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Study plan generation failed: {error}"
                    )

            # Reload application after generation.

            application = get_application(
                application["id"]
            )

            show_generation_info(
                application["id"],
                "study_plan"
            )

            study_plan_json = application.get(
                "study_plan_json"
            )

            if study_plan_json:

                plan_data = json.loads(
                    study_plan_json
                )

                for day in plan_data.get("days", []):

                    with st.expander(
                        f"Day {day.get('day', '?')} · "
                        f"{day.get('focus', 'Study')}"
                    ):

                        st.write(
                            f"**Hours:** {day.get('hours', 0)}"
                        )

                        for task in day.get("tasks", []):

                            st.write(
                                f"- {task}"
                            )

                st.download_button(
                    "Download Study Plan",
                    data=json.dumps(
                        plan_data,
                        indent=2,
                        ensure_ascii=False
                    ),
                    file_name="study_plan.json",
                    mime="application/json"
                )


# ==================================================
# 12. CHATGPT-STYLE RAG CHATBOT
# ==================================================

elif page == "Chat":

    application = get_active_application()

    if application is None:

        st.title("InterviewForge AI")

        st.info(
            "Select an existing application or create a new "
            "analysis before starting a document-based chat."
        )

    else:

        st.title("InterviewForge AI")

        st.caption(
            f"Chatting about: {application['job_title']}"
        )

        st.divider()

        # Load previous conversation.

        history = get_chat_history(
            application["id"]
        )

        # Display previous messages.

        for message in history:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # Chat input.

        user_question = st.chat_input(
            "Ask anything about your resume, JD, or interview preparation..."
        )

        if user_question:

            # Save user question.

            save_chat_message(
                application["id"],
                "user",
                user_question
            )

            # Retrieve conversation without duplicating
            # the current question in the prompt.

            updated_history = get_chat_history(
                application["id"]
            )

            try:

                start_time = time.perf_counter()

                with st.spinner(
                    "InterviewForge is thinking..."
                ):

                    answer = answer_question(
                        llm,
                        application["id"],
                        user_question,
                        updated_history[:-1]
                    )

                    save_chat_message(
                        application["id"],
                        "assistant",
                        answer
                    )

                elapsed_time = round(
                    time.perf_counter() - start_time,
                    2
                )

                st.session_state["last_chat_generation"] = {
                    "application_id": application["id"],
                    "provider": llm.last_provider,
                    "model": llm.last_model,
                    "time": elapsed_time
                }

                st.rerun()

            except Exception as error:

                st.error(
                    f"Chatbot error: {error}"
                )

        # Display most recent chat generation information.

        chat_info = st.session_state.get(
            "last_chat_generation"
        )

        if (
            chat_info
            and chat_info["application_id"] == application["id"]
        ):

            st.caption(
                f"Last response: {chat_info['provider']} "
                f"· {chat_info['model']} "
                f"· {chat_info['time']} seconds"
            )

            if chat_info["provider"] != selected_provider:

                st.warning(
                    f"Fallback provider used: {chat_info['provider']}"
                )
