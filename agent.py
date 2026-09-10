import json
from pathlib import Path
from datetime import date, timedelta

import ollama

from rag import KnowledgeBase


MEMORY_FILE = Path("memory.json")


class LearningAgent:

    def __init__(
        self,
        knowledge_base,
        model="llama3.2:3b"
    ):

        self.kb = knowledge_base
        self.model = model
        self.memory = self.load_memory()


    # -------------------------
    # MEMORY
    # -------------------------

    def load_memory(self):

        if MEMORY_FILE.exists():

            try:

                return json.loads(
                    MEMORY_FILE.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:
                pass

        return {
            "student_name": "",
            "goal": "",
            "history": []
        }


    def save_memory(self):

        MEMORY_FILE.write_text(
            json.dumps(
                self.memory,
                indent=4
            ),
            encoding="utf-8"
        )


    def clear_memory(self):

        self.memory = {
            "student_name": "",
            "goal": "",
            "history": []
        }

        self.save_memory()


    # -------------------------
    # RAG TOOL
    # -------------------------

    def search_materials(self, question):

        return self.kb.search(
            question,
            k=4
        )


    # -------------------------
    # STUDY PLAN TOOL
    # -------------------------

    def create_study_plan(self, days=7):

        goal = self.memory.get("goal")

        if not goal:
            goal = "Complete my course preparation"

        start_date = date.today()

        result = []

        result.append(
            f"## 📅 {days}-Day Study Plan"
        )

        result.append(
            f"**Goal:** {goal}\n"
        )

        for i in range(days):

            current_date = (
                start_date +
                timedelta(days=i)
            )

            result.append(
                f"### Day {i + 1} "
                f"({current_date.strftime('%d-%m-%Y')})"
            )

            result.append(
                "- Study one important topic"
            )

            result.append(
                "- Review your notes"
            )

            result.append(
                "- Practice 5 questions"
            )

            result.append(
                "- Revise previous topics\n"
            )

        return "\n".join(result)


    # -------------------------
    # QUIZ TOOL
    # -------------------------

    def generate_quiz(self, topic):

        context = self.search_materials(
            topic
        )

        context_text = "\n\n".join(
            context
        )

        prompt = f"""

Create 5 multiple-choice questions
about the topic:

{topic}

Use the following course material:

{context_text}

For each question:

1. Give four options A, B, C and D.
2. Give the correct answer.
3. Give a short explanation.

"""

        return self.chat(prompt)


    # -------------------------
    # OLLAMA
    # -------------------------

    def chat(self, prompt):

        response = ollama.chat(

            model=self.model,

            messages=[

                {
                    "role": "system",

                    "content": """
You are an AI Learning and Study Assistant.

Help students understand their
course subjects clearly.

Use the provided course material
when available.

Give simple and accurate answers.
"""
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ]
        )

        return response["message"]["content"]


    # -------------------------
    # AGENT
    # -------------------------

    def run(self, user_input):

        text = user_input.lower()


        # Save student's name

        if text.startswith("my name is"):

            name = user_input[
                len("my name is"):
            ].strip()

            self.memory["student_name"] = name

            self.save_memory()

            return (
                f"Nice to meet you, {name}! 😊\n\n"
                "I will remember your name."
            )


        # Save goal

        if "my goal is" in text:

            goal = user_input.split(
                "my goal is",
                1
            )[1].strip()

            self.memory["goal"] = goal

            self.save_memory()

            return (
                f"Great! 🎯\n\n"
                f"Your goal is: **{goal}**"
            )


        # Study plan

        if (
            "study plan" in text
            or
            "study schedule" in text
        ):

            return self.create_study_plan()


        # Quiz

        if (
            "quiz" in text
            or
            "mcq" in text
        ):

            topic = user_input

            for word in [
                "generate",
                "create",
                "quiz",
                "mcq",
                "about"
            ]:

                topic = topic.replace(
                    word,
                    ""
                )

            topic = topic.strip()

            if not topic:
                topic = "the course material"

            return self.generate_quiz(
                topic
            )


        # RAG

        documents = self.search_materials(
            user_input
        )

        context = "\n\n---\n\n".join(
            documents
        )


        # Conversation memory

        history = self.memory[
            "history"
        ][-6:]


        history_text = "\n".join(

            f"{item['role']}: "
            f"{item['content']}"

            for item in history

        )


        prompt = f"""

Student Question:

{user_input}


Relevant Course Material:

{context if context else
"No relevant course material found."}


Previous Conversation:

{history_text}


Answer the student's question
clearly and simply.

If the course material does not
contain enough information,
say that clearly.

"""


        answer = self.chat(prompt)


        # Save conversation

        self.memory["history"].append({

            "role": "user",

            "content": user_input

        })


        self.memory["history"].append({

            "role": "assistant",

            "content": answer

        })


        self.save_memory()


        return answer

        