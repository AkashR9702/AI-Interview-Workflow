
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama 
from typing import Annotated
from pydantic import BaseModel, Field 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import PyPDFLoader , TextLoader
# from langgraph.types import Interrupt , Command
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model= "llama3.2:1b")

#Schema for Resume
class ResumeSchema(BaseModel):

    skills: List[str] = Field(default_factory=list)
    experience_years: int = Field(default=0)
    education: str = Field(default="Not specified")
    key_projects: List[str] = Field(default_factory=list)


resume_parser = PydanticOutputParser(pydantic_object=ResumeSchema)

#Schema for Job-description
class JDSchema(BaseModel):

    job_role: str 
    required_skills: List[str]
    experience_required : int
    responsibilities: List[str]

jd_parser = PydanticOutputParser(pydantic_object=JDSchema)

# Question Generation Schema
class QuestionsSchema(BaseModel):
    questions: Annotated[List[str],Field(min_length=5, max_length=5)]

questions_parser = PydanticOutputParser(pydantic_object=QuestionsSchema)

# Evaluation Schema
class EvaluationSchema(BaseModel):

    score : int = Field(ge=1 ,le=10)
    feedback : str = Field(description="Feeback for the Users Question")

parser = PydanticOutputParser(pydantic_object=EvaluationSchema)


# Main Interview State
class InterviewState(TypedDict, total=False):
    resume_path : str
    jd_path : str
    raw_resume: str 
    raw_job_description: str
    
    resume_profile: Optional[Dict[str,any]]
    jd_profile: Optional[Dict[str,any]]
    
    questions: List[str]
    answers: Dict[str, str]
    
    evaluation: List[Dict]
    
    final_report: Optional[str]


# Nodes
def document_ingest(state):

    """Initial state setup - passthrough node"""

    return state

def resume_parse(state : InterviewState):

    """Load PDF resume and extracts raw text"""

    pdf_path = state["resume_path"]

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    resume_text = "\n\n".join(d.page_content for d in docs)

    state["raw_resume"] = resume_text

    return state 

def jd_parse(state : InterviewState):

    """Load Job description Text file"""

    jd_path = state["jd_path"]

    loader = TextLoader(jd_path)
    docs = loader.load()

    response = "\n\n".join(doc.page_content for doc in docs)

    state["raw_job_description"] = response
    return state
 
def normalize_resume(state):
    
    """Extract structured data from resume"""

    resume_text = state['raw_resume']

    examples = """
                    Example 1:
                    Resume: "John Doe, 5 years Python developer, worked at Google on ML projects. Skills: Python, TensorFlow, AWS. BS in Computer Science."

                    JSON Output:
                    {
                        "skills": ["Python", "TensorFlow", "AWS", "Machine Learning"],
                        "experience_years": 5,
                        "education": "BS in Computer Science",
                        "key_projects": ["ML projects at Google"]
                    }

                    Example 2:
                    Resume: "Fresh graduate with internship experience. Skills: Java, SQL. Built a todo app and weather app. BTech CSE 2024."

                    JSON Output:
                    {
                        "skills": ["Java", "SQL"],
                        "experience_years": 0,
                        "education": "BTech CSE",
                        "key_projects": ["Todo app", "Weather app"]
                    }
                    """

    prompt = f"""
                You MUST return ONLY valid JSON matching this EXACT structure. No extra fields allowed.

                REQUIRED FORMAT:
                {{
                    "skills": ["skill1", "skill2"],
                    "experience_years": 0,
                    "education": "degree as a SINGLE STRING",
                    "key_projects": ["project1", "project2"]
                }}

                CRITICAL RULES:
                - education must be a STRING, NOT an object/dict
                - Include ONLY these 4 fields, nothing else
                - No "name", "career_objective" or other extra fields

                Examples:
                {examples}

                Now extract from this resume:
                {resume_text}

                Output ONLY the JSON object. No explanations. No markdown code blocks.
                {resume_parser.get_format_instructions()}
"""

    response = model.invoke(prompt)
    
    try:
        parsed = resume_parser.parse(response.content)
        state['resume_profile'] = parsed.model_dump()
    except OutputParserException as e:
        print(f"PARSE ERROR: {e}")
        state['resume_profile'] = {}
    
    return state


def normalize_jd(state):

    """Extract structured job requirements from JD using LLM"""

    jd_text = state['raw_job_description']

    examples = """

                Example 1:
                JD: "We are hiring a Python Developer with 3 years experience.
                Skills: Python, Django, REST APIs.
                Responsibilities: Build APIs, maintain backend services."

                JSON Output:
                {
                    "job_role": "Python Developer",
                    "required_skills": ["Python", "Django", "REST APIs"],
                    "experience_required": 3,
                    "responsibilities": ["Build APIs", "Maintain backend services"]
                }

                Example 2:
                JD: "Entry level ML Engineer.
                Skills: Python, Machine Learning, Pandas.
                Responsibilities: Data preprocessing, model training."

                JSON Output:
                {
                    "job_role": "ML Engineer",
                    "required_skills": ["Python", "Machine Learning", "Pandas"],
                    "experience_required": 0,
                    "responsibilities": ["Data preprocessing", "Model training"]
                }
            """
    
    prompt = f"""
                Extract the structured information from job description:
                
                Example : {examples},
                Now Extract from  JD : {jd_text},
                {jd_parser.get_format_instructions()} 
            """
    
    response = model.invoke(prompt)

    try :

        parsed = jd_parser.parse(response.content)

        state['jd_profile'] = parsed.model_dump()

    except OutputParserException:

        state['jd_profile'] = {}

    return state

def generate_questions(state : InterviewState):

    """Generate contextual questions based on JD and resume"""

    resume = state["resume_profile"]
    jd = state["jd_profile"]

    prompt = f"""
                You are an expert interview question generator.

                Candidate: skills={resume.get('skills',[])}, experience={resume.get('experience_years', 0)} years
                Job: {jd.get('job_role', '')}, requires {jd.get('required_skills', [])}

                Generate EXACTLY 5 interview questions as a simple list of strings.

                Return ONLY this format:
                {{
                "questions": [
                    "Question 1 text here?",
                    "Question 2 text here?",
                    "Question 3 text here?",
                    "Question 4 text here?",
                    "Question 5 text here?"
                ]
                }}

                Example:
                {{
                "questions": [
                    "Explain your Python experience",
                    "Describe a challenging project",
                    "How do you debug code?",
                    "Why this role?",
                    "Where do you see yourself in 2 years?"
                ]
                }}

                {questions_parser.get_format_instructions()}
            """
    
    
    response = model.invoke(prompt)

    try:

        parsed = questions_parser.parse(response.content.strip())

        state["questions"] = parsed.questions

    except OutputParserException:

        skill = jd.get('required_skills', ['Python'])[0]
        state["questions"] = [
            f"Explain your experience with {skill}",
            f"Describe your work on: {resume.get('key_projects', ['a recent project'])[0]}",
            "What's the most challenging technical problem you've solved?",
            f"How would you approach {jd.get('responsibilities', ['this role'])[0]}?",
            "Why are you interested in this position?"
        ]
    
    return state


def conduct_interview(state):

    """Passthrough node for potential future HITL interrupt implementation"""

    return state

def collect_answers(state: InterviewState):

    """Collects answers from user via terminal input"""

    answers = {}
    for q in state.get('questions', []):
        answers[q] = input(f"\n{q}\nAnswer: ")
    
    state['answers'] = answers
    return state
   

def evaluate_responses(state: InterviewState):

    """Evaluates each answer using LLM with scoring (1-10) and feedback"""

    evaluations = []

    # Few-shot examples
    examples = [
        {
            "question": "Explain your experience with Python",
            "answer": "I have 2 years experience building web apps in Python",
            "score": 8,
            "feedback": "Good explanation with concrete projects mentioned."
        },
        {
            "question": "Describe a challenging project you worked on",
            "answer": "I worked on a bug-heavy legacy system and improved performance by 30%",
            "score": 9,
            "feedback": "Excellent example showing measurable impact and problem-solving."
        }
    ]

    # few-shot prompt
    example_text = ""
    for ex in examples:
        example_text += f"""
                            Example:
                            Question: {ex['question']}
                            Answer: {ex['answer']}

                            JSON Output:
                            {{
                            "score": {ex['score']},
                            "feedback": "{ex['feedback']}"
                            }}
                            """

    for question, answer in state["answers"].items():
        prompt = f"""
                    You are a strict interview evaluator.

                    Evaluate the candidate answer strictly.

                    Question : {question}

                    Answer : {answer}

                    Respond ONLY with JSON matching this format:
                    {{
                        "score": <integer 1-10>,
                        "feedback": "<string: professional evaluation feedback>"
                    }}

                    Here are examples for reference:{example_text}
                """
        response = model.invoke(prompt)

        try:
            parsed = parser.parse(response.content.strip())
            evaluation = {
                "question": question,
                "answer": answer,
                "score": parsed.score,
                "feedback": parsed.feedback
            }

        except OutputParserException:
            # Fallback safety
            evaluation = {
                "question": question,
                "answer": answer,
                "score": 5,
                "feedback": "Unable to parse evaluation reliably."
            }

        evaluations.append(evaluation)

    state["evaluation"] = evaluations
    return state


def generate_report(state : InterviewState):
    
    """Generates final professional interview report with strengths and weaknesses"""

    evaluation = state["evaluation"]

    prompt = f"""You a Interview Evaluator 
            Based on the Questions : {state['questions']} and,
            Candidates Answers : {state['answers']} and the Evaluation : {evaluation},
            
            Generate a Profressional Report Along with Details on Weaknesses , Strengths , Space for Improvement
            """
    response = model.invoke(prompt)

    state["final_report"] = response.content

    return state 


# Workflow Connection
workflow = StateGraph(InterviewState)

checkpointer = InMemorySaver()

workflow.add_node("document_ingest", document_ingest)
workflow.add_node("resume_parse", resume_parse)
workflow.add_node("jd_parse", jd_parse)
workflow.add_node("normalize_resume", normalize_resume)
workflow.add_node("normalize_jd", normalize_jd)
workflow.add_node("generate_questions",generate_questions)
workflow.add_node("conduct_interview",conduct_interview)
workflow.add_node("collect_answers", collect_answers)
workflow.add_node("evaluate_responses", evaluate_responses)
workflow.add_node("generate_report", generate_report)

workflow.add_edge(START, "document_ingest")
workflow.add_edge("document_ingest", "resume_parse")
workflow.add_edge("resume_parse", "jd_parse")
workflow.add_edge("jd_parse","normalize_resume")
workflow.add_edge("normalize_resume", "normalize_jd")  
workflow.add_edge("normalize_jd", "generate_questions")
workflow.add_edge("generate_questions", "conduct_interview")
workflow.add_edge("conduct_interview","collect_answers")
workflow.add_edge("collect_answers","evaluate_responses")
workflow.add_edge("evaluate_responses", "generate_report")
workflow.add_edge("generate_report", END)


graph = workflow.compile(checkpointer=checkpointer)


initial_state = {
    "resume_path": "C:/Users/Akash/Desktop/Langgraph/Pre_Int_Agent/sample_cv.pdf",
    "jd_path" : "C:/Users/Akash/Desktop/Langgraph/Pre_Int_Agent/sample_jd.txt",
    "raw_job_description": None, 
    "resume_profile": None,
    "jd_profile": None,
    "questions": [],
    "answers": {},
    "evaluation": [], 
    "final_report": None
}

config = {"configurable": {"thread_id": "1"}}


result = graph.invoke(initial_state, config)

print("\n=== EVALUATION SCORES ===")
for eval in result.get('evaluation', []):
    print(f"Score: {eval['score']}/10")

print("\n=== FINAL REPORT ===")
print(result['final_report'])


