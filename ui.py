import requests

import streamlit as st

st.set_page_config(page_title="AWS AI Architect", page_icon="☁️")


with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg",
        width=100,
    )
    st.title("System Status")
    st.success("API: Connected")
    st.success("Vector DB: 3,160 Chunks")
    st.divider()
    st.markdown("""
    **Project:** AWS Well-Architected RAG  
    **Engineer:** Itoro Sampson  
    **Stack:** FastAPI, ChromaDB, Llama 3.2
    """)

    if st.button("Clear Conversation Cache"):
        st.session_state.messages = []
        st.rerun()

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
                sources = data.get("sources", [])

                # Display the Main Answer
                st.markdown("### 🤖 Architect's Guidance")
                st.write(answer)

                # Display Sources as Badges
                if sources:
                    st.markdown("---")
                    st.markdown("#### 📚 Verified References")

                    # Create a horizontal row of "chips" or badges
                    cols = st.columns(len(sources) if len(sources) > 0 else 1)
                    for i, source in enumerate(sources):
                        with cols[i]:
                            pillar_name = (
                                source["pillar"]
                                if source["pillar"]
                                else "AWS Framework"
                            )
                            st.info(
                                f"**{pillar_name}**\n\nPage {source['page_number']}"
                            )

                # If it was a cached response, show a small lightning bolt
                if data.get("cached"):
                    st.caption("⚡ *Response served instantly from local cache*")
