"""
LangGraph workflow construction
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from .schemas import InterviewState
from .nodes import (
    document_ingest,
    resume_parse,
    jd_parse,
    normalize_resume,
    normalize_jd,
    generate_questions,
    conduct_interview,
    collect_answers,
    evaluate_responses,
    generate_report
)


def create_interview_graph():
    """Creates and compiles the interview workflow graph"""
    
    # Create workflow
    workflow = StateGraph(InterviewState)
    
    # Add checkpointer
    checkpointer = InMemorySaver()
    
    # Add nodes
    workflow.add_node("document_ingest", document_ingest)
    workflow.add_node("resume_parse", resume_parse)
    workflow.add_node("jd_parse", jd_parse)
    workflow.add_node("normalize_resume", normalize_resume)
    workflow.add_node("normalize_jd", normalize_jd)
    workflow.add_node("generate_questions", generate_questions)
    workflow.add_node("conduct_interview", conduct_interview)
    workflow.add_node("collect_answers", collect_answers)
    workflow.add_node("evaluate_responses", evaluate_responses)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges (workflow flow)
    workflow.add_edge(START, "document_ingest")
    workflow.add_edge("document_ingest", "resume_parse")
    workflow.add_edge("resume_parse", "jd_parse")
    workflow.add_edge("jd_parse", "normalize_resume")
    workflow.add_edge("normalize_resume", "normalize_jd")
    workflow.add_edge("normalize_jd", "generate_questions")
    workflow.add_edge("generate_questions", "conduct_interview")
    workflow.add_edge("conduct_interview", "collect_answers")
    workflow.add_edge("collect_answers", "evaluate_responses")
    workflow.add_edge("evaluate_responses", "generate_report")
    workflow.add_edge("generate_report", END)
    
    # Compile and return
    return workflow.compile(checkpointer=checkpointer)