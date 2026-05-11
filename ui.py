import requests

import streamlit as st

st.set_page_config(page_title="AWS AI Architect", page_icon="☁️")

st.title(" AWS Well-Architected GPT")
st.markdown("Ask questions about AWS security and best practices.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("How do I secure my root user?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call my FastAPI server
    with st.chat_message("assistant"):
        with st.spinner("Consulting the Framework..."):
            response = requests.post(
                "http://localhost:8000/ask", json={"question": prompt}
            )
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = ", ".join([str(s["page_number"]) for s in data["sources"]])

                full_response = f"{answer}\n\n**Sources: Pages {sources}**"
                st.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                st.error("Architect is busy. Try again later.")
