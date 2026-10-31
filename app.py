
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from ddgs import DDGS

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-2.5-flash"

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Deep Research Agent")
st.write(
    "An autonomous AI research system with planning, "
    "evidence retrieval, self-reflection and revision."
)

# Initialize Gemini
if not API_KEY:
    st.error("GEMINI_API_KEY not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ==========================================
# HELPER FUNCTION: GEMINI RESPONSE
# ==========================================

def ask_gemini(prompt):

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text or ""


# ==========================================
# PHASE 1: MULTI-PHASE PLANNING
# ==========================================

def planning_agent(topic):

    st.subheader("📋 Phase 1: Research Planning")

    prompt = f"""
    You are an expert research planning agent.

    Research Topic: {topic}

    Break this topic into 5 clear research questions.

    Create a structured research plan covering:
    1. Introduction and background
    2. Major applications or concepts
    3. Benefits and opportunities
    4. Challenges and limitations
    5. Future scope and conclusion

    Return a numbered research plan.
    """

    plan = ask_gemini(prompt)

    st.markdown(plan)

    return plan


# ==========================================
# PHASE 2: EVIDENCE RETRIEVAL
# ==========================================

def evidence_retrieval(plan):

    st.subheader("🔎 Phase 2: Evidence Retrieval")

    evidence = []

    # Search using each research question
    with DDGS() as search:

        results = search.text(
            plan,
            max_results=8
        )

        for result in results:

            evidence.append({
                "title": result.get("title", "Untitled"),
                "url": result.get("href", ""),
                "summary": result.get("body", "")
            })

    if not evidence:
        st.warning("No evidence found. Try another topic.")
        return []

    for i, item in enumerate(evidence, 1):

        st.markdown(f"**Source {i}: {item['title']}**")
        st.write(item["summary"])
        st.markdown(f"[Visit Source]({item['url']})")

    return evidence


# ==========================================
# PHASE 3: INITIAL REPORT GENERATION
# ==========================================

def generate_draft(topic, plan, evidence):

    st.subheader("📝 Phase 3: Initial Report Generation")

    prompt = f"""
    You are an expert research writer.

    Research Topic:
    {topic}

    Research Plan:
    {plan}

    Retrieved Evidence:
    {evidence}

    Instructions:
    - Write a detailed research report.
    - Follow the research plan.
    - Use retrieved evidence to support claims.
    - Include source URLs next to supported claims.
    - Do not invent facts or references.
    - Clearly identify information that cannot be verified.

    Include:
    1. Introduction
    2. Main Findings
    3. Benefits
    4. Challenges
    5. Future Scope
    6. Conclusion
    7. References
    """

    draft = ask_gemini(prompt)

    st.markdown(draft)

    return draft


# ==========================================
# PHASE 4: SELF-REFLECTION AND CRITIQUE
# ==========================================

def self_critique(draft, evidence):

    st.subheader("🧠 Phase 4: Self-Reflection and Critique")

    prompt = f"""
    You are a strict research quality evaluator.

    Evaluate the following research report.

    REPORT:
    {draft}

    AVAILABLE EVIDENCE:
    {evidence}

    Check:
    1. Factual accuracy
    2. Evidence support
    3. Missing information
    4. Logical consistency
    5. Clarity and completeness
    6. Source reliability

    If improvements are needed, list them clearly.

    At the END, write exactly one of:

    VERDICT: READY
    VERDICT: NEEDS_REVISION

    READY means the report meets the quality requirements.
    NEEDS_REVISION means important improvements are required.
    """

    critique = ask_gemini(prompt)

    st.markdown(critique)

    return critique


# ==========================================
# PHASE 5: AUTONOMOUS REVISION
# ==========================================

def autonomous_revision(topic, draft, critique, evidence):

    st.subheader("🔄 Phase 5: Autonomous Revision")

    prompt = f"""
    You are an autonomous research revision agent.

    Research Topic:
    {topic}

    Previous Research Report:
    {draft}

    Critique and Feedback:
    {critique}

    Available Evidence:
    {evidence}

    Your task:
    - Correct identified weaknesses.
    - Add missing relevant information.
    - Improve clarity and logical flow.
    - Use only supported evidence.
    - Do not invent facts or references.
    - Preserve valid information from the previous report.

    Generate the improved complete research report.
    """

    revised_report = ask_gemini(prompt)

    st.markdown(revised_report)

    return revised_report


# ==========================================
# MAIN DEEP RESEARCH WORKFLOW
# ==========================================

def run_research(topic):

    try:

        # PHASE 1: PLANNING
        with st.expander("Phase 1: Planning", expanded=True):
            plan = planning_agent(topic)

        # PHASE 2: RETRIEVAL
        with st.expander("Phase 2: Evidence Retrieval", expanded=True):
            evidence = evidence_retrieval(plan)

        if not evidence:
            st.error("Research stopped because no evidence was retrieved.")
            return

        # PHASE 3: INITIAL DRAFT
        with st.expander("Phase 3: Initial Report", expanded=True):
            draft = generate_draft(
                topic, plan, evidence
            )

        # PHASE 4 & 5: CRITIQUE AND REVISION
        max_iterations = 2

        final_report = draft

        for iteration in range(max_iterations):

            st.divider()

            st.info(
                f"Review and Revision Iteration {iteration + 1}"
            )

            critique = self_critique(
                final_report, evidence
            )

            if "VERDICT: READY" in critique.upper():
                st.success("Research report passed quality review!")
                break

            if iteration == max_iterations - 1:
                st.warning(
                    "Maximum revision attempts reached. "
                    "Displaying the latest report."
                )
                break

            final_report = autonomous_revision(
                topic,
                final_report,
                critique,
                evidence
            )

        # FINAL REPORT
        st.divider()
        st.header("📄 Final Research Report")

        st.markdown(final_report)

        st.download_button(
            label="Download Research Report",
            data=final_report,
            file_name="deep_research_report.md",
            mime="text/markdown"
        )

        st.success(
            "Deep Research Agent execution completed!"
        )

    except Exception as e:

        st.error("An error occurred during research.")
        st.exception(e)


# ==========================================
# USER INTERFACE
# ==========================================

st.divider()

topic = st.text_area(
    "Enter your research topic:",
    placeholder="Example: Impact of Artificial Intelligence in Healthcare",
    height=100
)

if st.button("🚀 Start Deep Research", type="primary"):

    if topic.strip():

        with st.spinner(
            "Deep Research Agent is working..."
        ):
            run_research(topic.strip())

    else:

        st.warning("Please enter a research topic.")
