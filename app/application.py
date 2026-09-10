from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
from markupsafe import Markup
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

def nl2br(value):
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters['nl2br'] = nl2br

# --- IMPORTANT FIX: Load the QA chain ONCE globally when the server boots ---
print("Initializing QA Chain... Please wait.")
qa_chain = create_qa_chain()

if qa_chain is None:
    print("WARNING: QA chain initialization returned None. Check vectorstore/LLM configs.")
else:
    print("QA Chain successfully loaded globally!")

@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session['messages']
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                if qa_chain is None:
                    raise Exception("QA Chain is not initialized. Check your Groq API key or vectorstore.")
                
                # Fast inference using the pre-loaded QA chain
                response = qa_chain.invoke({"query": user_input})
                result = response.get("result", "No response generated.")

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template(
                    "index.html",
                    messages=session["messages"],
                    error=error_msg
                )

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))

@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False
    )