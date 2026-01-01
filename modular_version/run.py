"""
Entry point for running the interview agent
"""
from .graph import create_interview_graph
from .config import RESUME_PATH, JD_PATH, THREAD_ID


def main():
    """Main execution function"""
    
    # Create the graph
    graph = create_interview_graph()
    
    # Initial state
    initial_state = {
        "resume_path": RESUME_PATH,
        "jd_path": JD_PATH,
        "raw_job_description": None,
        "resume_profile": None,
        "jd_profile": None,
        "questions": [],
        "answers": {},
        "evaluation": [],
        "final_report": None
    }
    
    # Configuration
    config = {"configurable": {"thread_id": THREAD_ID}}
    
    # Run the interview workflow
    print("Starting AI Interview Agent...")
    print("="*50)
    
    result = graph.invoke(initial_state, config)
    
    # Display results
    print("\n" + "="*50)
    print("EVALUATION SCORES")
    print("="*50)
    for eval in result.get('evaluation', []):
        print(f"Score: {eval['score']}/10")
    
    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(result['final_report'])


if __name__ == "__main__":
    main()