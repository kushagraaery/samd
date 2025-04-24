import streamlit as st
import pandas as pd
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile
import plotly.express as px

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Email credentials from environment variables
EMAIL_ADDRESS = "johnwickcrayons@gmail.com"
EMAIL_PASSWORD = "afpt eoyt asaq qzjh"
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# --- Streamlit UI Header ---
st.set_page_config(page_title="SaMD Launch AI Assessor")
st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: 600;
    }
    .result-box {
        border: 1px solid #DDD;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 SaMD Agentic AI Launch Assessor")
st.markdown("**Upload your full market & readiness survey file to begin assessment.**")

# --- Scoring Logic ---
def calculate_scores(row):
    risk_score = (5 - int(row['Risk_Class'])) + (int(row['Medical_Incidence'])) + (3 - int(row['Tech_Limitations']))
    readiness_score = int(row['Market_Maturity']) + int(row['Affiliate_Readiness']) + int(row['Digital_Readiness'])
    return risk_score, readiness_score

# --- Agent Nodes ---
def risk_impact_evaluator(state):
    row = state["row"]
    prompt = f"""
    You are a Regulatory and Clinical Risk Expert for Software as a Medical Device (SaMD).

    Assess the **clinical and technical risk** for launching a SaMD in {row['Country']}.

    Input:
    - Risk Class (1-5): {row['Risk_Class']}
    - Medical Incidence (1-3): {row['Medical_Incidence']}
    - Technical Limitations (1-3): {row['Tech_Limitations']}
    - Predicate Device in US: {row['Predicate_US']}
    - Population: {row['Population']}

    Provide:
    - Structured Risk Summary
    - Clinical & Technical Risk Factors
    - Final Risk Impact Level (Low / Medium / High)
    """
    response = llm.invoke([SystemMessage(content="You are a Regulatory and Clinical Risk Expert."), HumanMessage(content=prompt)])
    state["risk_summary"] = response.content
    return state

def opportunity_roi_assessor(state):
    row = state["row"]
    prompt = f"""
    You are a Market Opportunity and ROI Analyst for global med-tech products.

    Based on the below, assess the **market potential** and **economic viability** for launching a SaMD in {row['Country']}.

    Input:
    - Risk Score: {row['Risk_Score']}
    - Country Wealth: {row['Country_Wealth']}
    - Population: {row['Population']}
    - Risk Summary:
    {state['risk_summary']}

    Provide:
    - Market Opportunity Overview
    - Country Wealth and ROI Alignment
    - ROI Summary (High / Medium / Low)
    """
    response = llm.invoke([SystemMessage(content="You are a financial and opportunity analyst for healthcare markets."), HumanMessage(content=prompt)])
    state["opportunity_summary"] = response.content
    return state

def readiness_capability_analyzer(state):
    row = state["row"]
    prompt = f"""
    You are a Market Readiness & Infrastructure Specialist.

    Analyze the digital and organizational readiness to support a SaMD launch in {row['Country']}.

    Inputs:
    - Market Maturity: {row['Market_Maturity']}
    - Affiliate Readiness: {row['Affiliate_Readiness']}
    - Digital Readiness: {row['Digital_Readiness']}
    - Readiness Score: {row['Readiness_Score']}

    Provide:
    - Readiness Overview
    - Infrastructure Strengths & Gaps
    - Final Readiness Verdict (Ready / Needs Improvement / Not Ready)
    """
    response = llm.invoke([SystemMessage(content="You are a digital infrastructure and readiness analyst."), HumanMessage(content=prompt)])
    state["readiness_summary"] = response.content
    return state

def strategic_decision_coordinator(state):
    row = state["row"]
    prompt = f"""
    You are a senior strategic executive in a global med-tech company.

    Make a **go-to-market** decision for the SaMD launch in {row['Country']} using the following insights:

    1. Risk Summary:
    {state['risk_summary']}

    2. Market Opportunity:
    {state['opportunity_summary']}

    3. Readiness Overview:
    {state['readiness_summary']}

    Use all dimensions to decide.

    Respond in this strict format:
    Decision: [Launch / Hold / Reject]
    Explanation:
    - Risk-Based Justification
    - ROI Justification
    - Readiness Justification
    - Final Synthesis
    """
    response = llm.invoke([SystemMessage(content="You are responsible for SaMD global strategy."), HumanMessage(content=prompt)])
    state["final_decision"] = response.content
    return state

# --- Summary Agent ---
def ai_summary_agent(results):
    # Build input prompt with structured summaries for each country
    decision_input = ""
    for r in results:
        decision_input += f"""
        Country: {r['Country']}
        Risk Summary: {r['Risk Summary']}
        Opportunity Summary: {r['Opportunity Summary']}
        Readiness Summary: {r['Readiness Summary']}
        """

    summary_prompt = f"""
    You are a Global MedTech Strategy Expert. Based on the following country-wise summaries for SaMD launch assessment, evaluate each country and provide:
    
    - Final Decision: (Launch / Hold / Reject)
    - Reasoning and Justification (based on Risk, Opportunity, and Readiness)
    
    Respond in **this format** in HTML:
    <b>Country:</b> Country Name<br>
    <b>Final Decision:</b> Launch / Hold / Reject<br>
    <b>Justification:</b> Detailed justification using all summaries (Risk, Opportunity, Readiness)<br><br>
    
    Here is the input data:
    {decision_input}
    """
    response = llm.invoke([
        SystemMessage(content="You are a senior strategic advisor evaluating product launches for each country."),
        HumanMessage(content=summary_prompt)
    ])
    return response.content


# --- LangGraph Flow ---
graph = StateGraph(dict)
graph.add_node("risk", risk_impact_evaluator)
graph.add_node("opportunity", opportunity_roi_assessor)
graph.add_node("readiness", readiness_capability_analyzer)
graph.add_node("decision", strategic_decision_coordinator)

graph.set_entry_point("risk")

graph.add_edge("risk", "opportunity")
graph.add_edge("opportunity", "readiness")
graph.add_edge("readiness", "decision")
graph.add_edge("decision", END)
agent_graph = graph.compile()

# --- Email Helper ---
# Function to generate PDF from HTML summary
def generate_pdf_from_html(html_content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    # Convert HTML content to plain text (basic formatting)
    lines = html_content.replace("<br>", "\n").replace("<b>", "").replace("</b>", "").split("\n")
    for line in lines:
        pdf.multi_cell(0, 10, line.strip())
    
    pdf.output(filename)

# Updated email sender with PDF attachment
def send_summary_email(to_email, user_name, html_content):
    msg = MIMEMultipart("mixed")
    msg['Subject'] = "SaMD Launch Assessment Summary"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email

    # Wrap the summary inside a proper email format
    email_body = f"""
    <html>
        <body>
            <p>Hi {user_name},</p>
            <p>Please find below the AI-generated summary for your SaMD launch assessment:</p>
            {html_content}
            <p>Regards,<br>Incedo Developer's Team</p>
        </body>
    </html>
    """

    # Attach the formatted HTML body
    html_part = MIMEText(email_body, 'html')
    msg.attach(html_part)

    # Generate temp PDF and attach
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        generate_pdf_from_html(email_body, tmp_pdf.name)
        tmp_pdf.seek(0)
        pdf_data = tmp_pdf.read()

    from email.mime.application import MIMEApplication
    pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
    pdf_attachment.add_header('Content-Disposition', 'attachment', filename="SaMD_Launch_Summary.pdf")
    msg.attach(pdf_attachment)

    # Send the email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())

# --- Streamlit UI Interaction ---
uploaded_file = st.file_uploader("📤 Upload Survey Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    required_columns = [
        'User_Email', 'Country', 'Risk_Class', 'Medical_Incidence', 'Tech_Limitations',
        'Predicate_US', 'Population', 'Country_Wealth', 'Market_Maturity',
        'Affiliate_Readiness', 'Digital_Readiness'
    ]
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing columns in your Excel file. Required: {', '.join(required_columns)}")
        st.stop()

    # Calculate scores
    df[['Risk_Score', 'Readiness_Score']] = df.apply(lambda row: pd.Series(calculate_scores(row)), axis=1)

    # Preview limited data
    display_cols = ['User_Email', 'Country', 'Risk_Class', 'Medical_Incidence', 'Tech_Limitations', 'Predicate_US']
    st.write("### ✅ Parsed Input Data", df[display_cols])

    results = []
    for idx, row in df.iterrows():
        st.markdown(f"<div class='big-font'>🔍 Assessing: {row['Country']}</div>", unsafe_allow_html=True)
        with st.spinner(f"🤖 Assessing {row['Country']}... please wait"):
            state = agent_graph.invoke({"row": row.to_dict()})
        with st.expander(f"📊 Detailed Results for {row['Country']}"):
            st.markdown(f"<div class='result-box'>🧪 **Risk & Impact**<br>{state['risk_summary']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>💰 **Opportunity & ROI**<br>{state['opportunity_summary']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>🏗️ **Readiness & Capability**<br>{state['readiness_summary']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>🚦 **Final Decision**<br>{state['final_decision']}</div>", unsafe_allow_html=True)
        results.append({
            "Country": row["Country"],
            "Risk Summary": state["risk_summary"],
            "Opportunity Summary": state["opportunity_summary"],
            "Readiness Summary": state["readiness_summary"],
            "Final Decision": state["final_decision"]
        })

    user_email = df['User_Email'].iloc[0]
    user_name = df['User_Name'].iloc[0]
    with st.spinner("📧 Generating AI Summary and sending email..."):
        html_summary = ai_summary_agent(results)
        send_summary_email(user_email, user_name, html_summary)

    st.success(f"📧 AI-generated summary email sent to: {user_email}")

    result_df = pd.DataFrame(results)
    st.download_button("📥 Download Launch Report", result_df.to_csv(index=False), "launch_assessment_report.csv")
    st.markdown("## 📊 Comparative Analysis Dashboard")

    # Merge original data with results for visualization
    merged_df = df.copy()
    merged_df["Risk Summary"] = [r["Risk Summary"] for r in results]
    merged_df["Opportunity Summary"] = [r["Opportunity Summary"] for r in results]
    merged_df["Readiness Summary"] = [r["Readiness Summary"] for r in results]
    merged_df["Final Decision"] = [r["Final Decision"] for r in results]
    
    # Chart 1: Country Wealth vs Country
    fig_wealth = px.bar(merged_df, x="Country", y="Country_Wealth", title="💰 Country Wealth Comparison", color="Country_Wealth", text="Country_Wealth")
    st.plotly_chart(fig_wealth, use_container_width=True)
    
    # Chart 2: Readiness Score vs Country
    fig_readiness = px.bar(merged_df, x="Country", y="Readiness_Score", title="🏗️ Readiness Score Comparison", color="Readiness_Score", text="Readiness_Score")
    st.plotly_chart(fig_readiness, use_container_width=True)
    
    # Chart 3: Risk Score vs Country
    fig_risk = px.bar(merged_df, x="Country", y="Risk_Score", title="🧪 Risk Score Comparison", color="Risk_Score", text="Risk_Score")
    st.plotly_chart(fig_risk, use_container_width=True)