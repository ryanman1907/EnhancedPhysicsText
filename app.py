import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# Load OpenAI API key
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Add CORS Middleware to allow requests from your HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend's URL in production
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Request data model
class QARequest(BaseModel):
    question: str
    context: str

@app.get("/")
def root():
    return {"message": "Welcome to the Interactive Physics Textbook API!"}

@app.post("/qa")
async def answer_question(request: QARequest):
    """
    Endpoint to send a question and context to GPT-4 and return the answer.
    """
    try:
        # Construct the combined prompt
        prompt = f"""
        Context: {request.context}

        Question: {request.question}

        Answer the question based on the provided context.
        """

        # Call OpenAI API
        chat_completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful physics assistant that explains introductory physics concepts and renders all math equations in valid latex syntax using either double dollar signs or \( \)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )

        # Extract and return the answer
        answer = chat_completion.choices[0].message.content.strip()
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
