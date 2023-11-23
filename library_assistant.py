import streamlit as st
import json
import time
import openai
import pandas as pd

# Function to show JSON
def show_json(obj):
    st.json(json.loads(obj.model_dump_json()))

# Streamlit interface for API key and question input
st.title("OpenAI Query Interface")
api_key = st.text_input("Enter your API key:", type="password")
question = st.text_input("Enter your question:")

if st.button("Submit"):
    if api_key and question:
        # Initialize OpenAI client with API key
        client = OpenAI(api_key=api_key)

        # Create a thread
        thread = client.beta.threads.create()

        # Create an assistant
        assistant = client.beta.assistants.create(
            name="Text analyzer",
            instructions="You are library information specialist from the University of Pretoria in South Africa.",
            model="gpt-4-1106-preview",
        )

        # Context
        context = '''use the resources from all the site pages found in library.up.ac.za to answer questions. 
        If you can't find the answer, ask the user to provide their Faculty and then provide a list of contacts for the 
        information specialist from the user's faculty.'''

        # Create a message to request a summary of 'feedback_text'
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Given the context '{context}', answer the question {question}",
        )

        # Create a run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Function to wait on run
        def wait_on_run(run, thread):
            while run.status == "queued" or run.status == "in_progress":
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                time.sleep(0.5)
            return run

        run = wait_on_run(run, thread)

        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Assuming you have set up your client and have the thread_id and message_id
        messages = client.beta.threads.messages.list(
            thread_id=thread.id, 
            order="asc", 
            after=message.id
        )

        # Display messages
        if messages.data:
            for message in messages.data:
                content = message.content
                if content:
                    for item in content:
                        if item.type == 'text':
                            value = item.text.value
                            st.write(f"Answer: {value}")
        else:
            st.write("No messages found")
    else:
        st.warning("Please provide both an API key and a question.")
