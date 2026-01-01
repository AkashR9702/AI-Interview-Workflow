"""
Pydantic schemas for structured data validation
"""
from typing import List, Annotated, TypedDict, Dict, Optional 
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# Main Interview State
class InterviewState(TypedDict, total=False):
    resume_path: str
    jd_path: str
    raw_resume: str 
    raw_job_description: str
    
    resume_profile: Optional[Dict[str, any]]
    jd_profile: Optional[Dict[str, any]]
    
    questions: List[str]
    answers: Dict[str, str]
    
    evaluation: List[Dict]
    
    final_report: Optional[str]

# Schema for Resume
class ResumeSchema(BaseModel):
    skills: List[str] = Field(default_factory=list)
    experience_years: int = Field(default=0)
    education: str = Field(default="Not specified")
    key_projects: List[str] = Field(default_factory=list)

# Schema for Job Description
class JDSchema(BaseModel):
    job_role: str 
    required_skills: List[str]
    experience_required: int
    responsibilities: List[str]

# Question Generation Schema
# Note: Change min_length and max_length to increase/decrease number of questions
class QuestionsSchema(BaseModel):
    questions: Annotated[List[str], Field(min_length=5, max_length=5)]

# Evaluation Schema
class EvaluationSchema(BaseModel):
    score: int = Field(ge=1, le=10)
    feedback: str = Field(description="Feedback for the User's Question")

# Create parsers
resume_parser = PydanticOutputParser(pydantic_object=ResumeSchema)
jd_parser = PydanticOutputParser(pydantic_object=JDSchema)
questions_parser = PydanticOutputParser(pydantic_object=QuestionsSchema)
evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationSchema)