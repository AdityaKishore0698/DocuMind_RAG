import streamlit as st
import requests

API_URL = "http://api:8000"

st.set_page_config(page_title="DocuMind", page_icon="🤖")

# Initialize states
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def login(username, password):
    try:
        res = requests.post(f"{API_URL}/auth/token", data={"username": username, "password": password})
        if res.status_code == 200:
            st.session_state.token = res.json()["access_token"]
            st.rerun()
        else:
            st.error("Invalid credentials")
    except Exception as e:
        st.error(f"Login failed: {e}")

def register(username, email, password):
    try:
        res = requests.post(f"{API_URL}/auth/register", json={"username": username, "email": email, "password": password})
        if res.status_code == 200:
            st.success("Registered! Please login.")
        else:
            st.error(res.json().get("detail", "Registration failed"))
    except Exception as e:
        st.error(f"Registration failed: {e}")

if not st.session_state.token:
    st.title("Login to DocuMind")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        u = st.text_input("Username", key="l_u")
        p = st.text_input("Password", type="password", key="l_p")
        if st.button("Login"):
            login(u, p)
    with tab2:
        ru = st.text_input("Username", key="r_u")
        re = st.text_input("Email", key="r_e")
        rp = st.text_input("Password", type="password", key="r_p")
        if st.button("Register"):
            register(ru, re, rp)
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

def load_session(session_id):
    st.session_state.session_id = session_id
    try:
        res = requests.get(f"{API_URL}/chat/sessions/{session_id}/messages", headers=headers)
        if res.status_code == 200:
            st.session_state.messages = res.json()
        else:
            st.session_state.messages = []
    except:
        st.session_state.messages = []
    st.rerun()

st.title("DocuMind")

# Sidebar
with st.sidebar:
    st.header("📚 Knowledge Base")
    with st.expander("Upload Documents", expanded=False):
        uploaded_files = st.file_uploader("Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True)
        if st.button("Upload Files") and uploaded_files:
            with st.spinner("Uploading..."):
                files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
                res = requests.post(f"{API_URL}/document/upload", files=files, headers=headers)
                if res.status_code == 200:
                    st.success("Files uploaded and processing!")
                else:
                    st.error("Upload failed")
                    
    with st.expander("Manage Documents", expanded=False):
        try:
            # Added trailing slash to prevent 307 redirect stripping Auth header
            docs_res = requests.get(f"{API_URL}/document/", headers=headers)
            if docs_res.status_code == 200:
                docs = docs_res.json()
                if not docs:
                    st.write("No documents uploaded.")
                for d in docs:
                    cols = st.columns([4, 1])
                    cols[0].write(f"📄 {d['filename']}")
                    if cols[1].button("❌", key=f"del_doc_{d['id']}"):
                        requests.delete(f"{API_URL}/document/{d['id']}", headers=headers)
                        st.rerun()
            else:
                st.error(f"Failed to load documents: {docs_res.status_code}")
        except Exception as e:
            st.error(f"Failed to load documents: {e}")
    
    st.divider()
    st.header("💬 Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
        
    try:
        sessions_res = requests.get(f"{API_URL}/chat/sessions", headers=headers)
        if sessions_res.status_code == 200:
            for s in sessions_res.json():
                if st.button(f"💬 {s['title']}", key=f"session_{s['id']}", use_container_width=True):
                    load_session(s['id'])
    except:
        st.write("Could not load sessions.")
    
    st.divider()
    st.header("⚙️ Settings")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.token = None
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False

    if st.button("🚨 Delete Account", use_container_width=True):
        st.session_state.confirm_delete = True
        
    if st.session_state.confirm_delete:
        st.warning("Are you sure? This will permanently delete all your data.")
        col1, col2 = st.columns(2)
        if col1.button("Yes, Delete"):
            res = requests.delete(f"{API_URL}/auth/profile", headers=headers)
            if res.status_code == 200:
                st.session_state.token = None
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.confirm_delete = False
                st.rerun()
        if col2.button("Cancel"):
            st.session_state.confirm_delete = False
            st.rerun()

# Main Chat Header (Management for Active Chat)
if st.session_state.session_id:
    head_cols = st.columns([6, 2, 2])
    with head_cols[1].popover("✏️ Rename"):
        new_title = st.text_input("New Title", label_visibility="collapsed", key="rename_top")
        if st.button("Save", key="save_top"):
            requests.put(f"{API_URL}/chat/sessions/{st.session_state.session_id}", json={"title": new_title}, headers=headers)
            st.rerun()
    if head_cols[2].button("🗑️ Delete", type="primary"):
        requests.delete(f"{API_URL}/chat/sessions/{st.session_state.session_id}", headers=headers)
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()
    st.divider()

# Main Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            req_data = {"query": prompt}
            if st.session_state.session_id:
                req_data["session_id"] = st.session_state.session_id
                
            response = requests.post(f"{API_URL}/chat/chat", json=req_data, headers=headers, stream=True)
            response.raise_for_status()
            
            new_session_id = response.headers.get("X-Session-ID")
            if new_session_id:
                st.session_state.session_id = int(new_session_id)
            
            st.info("💡 Tip: To stop the AI from generating, press the 'Stop' button in the top-right corner of the page.")
            def generate_response():
                try:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk.decode("utf-8")
                except Exception:
                    pass # Handle premature stream termination gracefully
                        
            full_response = st.write_stream(generate_response())
            
        except requests.exceptions.RequestException as e:
            full_response = f"Error communicating with API: {e}"
            st.error(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
