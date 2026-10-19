import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Tenant RAG Portal", layout="wide")

API_BASE_URL = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000/api/v1")

# TEAM SELECTOR IN SIDEBAR
st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Authentication Context")
active_team = st.sidebar.selectbox("Active Team Context", options=["Team_Alpha", "Team_Beta", "Team_Gamma"])
st.sidebar.info(f"Viewing portal as **{active_team}**")

headers = {"X-Team-ID": active_team}

tab_file, tab_search, tab_manage = st.tabs([
    "📁 Ingest File", 
    "🔎 Search & RRF Inspector", 
    "🗂️ Manage Team Documents"
])

# -----------------------------------------------------------------------------
# TAB 1: INGEST FILE (Tagged with active_team)
# -----------------------------------------------------------------------------
with tab_file:
    st.header(f"Ingest Document for `{active_team}`")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "pptx", "txt", "rtf"])

    if st.button("Process Document", type="primary", disabled=not uploaded_file):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        try:
            with st.spinner("Ingesting and embedding document..."):
                res = requests.post(
                    f"{API_BASE_URL}/ingest/file", 
                    files=files, 
                    headers=headers,  # Sends active team header
                    timeout=60
                )
            
            if res.status_code in (200, 201):
                st.success(f"Document uploaded successfully under **{active_team}**!")
                st.json(res.json())
            else:
                st.error(f"Error ({res.status_code}): {res.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend server: {e}")

# -----------------------------------------------------------------------------
# TAB 2: SEARCH & RRF INSPECTOR (With Customizable Top K)
# -----------------------------------------------------------------------------
with tab_search:
    st.header(f"RAG Search & Fusion Inspector for `{active_team}`")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query_str = st.text_input("Query", value="Vector database security")
    with col2:
        search_mode = st.selectbox("Search Mode", options=["hybrid", "dense", "sparse"])
    with col3:
        top_k = st.number_input("Top K Chunks", min_value=1, max_value=50, value=4, step=1)

    if st.button("Run Search", type="primary"):
        payload = {
            "query": query_str, 
            "top_k": int(top_k), 
            "mode": search_mode, 
            "team_id": active_team
        }
        
        try:
            with st.spinner(f"Searching for top {top_k} chunks..."):
                res = requests.post(f"{API_BASE_URL}/search", json=payload, headers=headers, timeout=15)
            
            if res.status_code == 200:
                response_data = res.json()
                
                # Robust response extraction
                if isinstance(response_data, list):
                    results = response_data
                elif isinstance(response_data, dict):
                    results = response_data.get("results", [])
                else:
                    results = []

                st.subheader(f"Retrieved {len(results)} Chunks for `{active_team}` (Top K = {top_k})")
                
                if not results:
                    st.info(f"No matching documents found for team `{active_team}`.")
                
                for item in results:
                    rb = item.get("rank_breakdown", {})
                    rrf_rank = rb.get("rrf_rank") or rb.get("dense_rank") or rb.get("sparse_rank") or "N/A"
                    rrf_score = rb.get("rrf_score", "N/A")
                    
                    st.markdown(f"#### 🏆 Final Rank #{rrf_rank} | RRF Score: `{rrf_score}`")
                    st.info(item.get("content", ""))
                    
                    # Collapsible inspection section for Dense vs Sparse metrics
                    with st.expander("🔍 Inspect Dense vs Sparse Match Breakdown"):
                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            d_rank = f"#{rb.get('dense_rank')}" if rb.get('dense_rank') is not None else "Not Retrieved"
                            st.metric("Dense (Vector) Rank", d_rank)
                            st.caption(f"Distance Score: `{rb.get('dense_score', 'N/A')}`")
                            
                        with c2:
                            s_rank = f"#{rb.get('sparse_rank')}" if rb.get('sparse_rank') is not None else "Not Retrieved"
                            st.metric("Sparse (BM25) Rank", s_rank)
                            st.caption(f"BM25 Score: `{rb.get('sparse_score', 'N/A')}`")
                            
                        with c3:
                            via = str(rb.get("retrieved_via", "N/A")).upper()
                            st.metric("Retrieved Via", via)
                            metadata = item.get("metadata", {})
                            st.caption(f"Chunk ID: `{item.get('chunk_id')}` | Source: `{metadata.get('source', 'N/A')}`")
                    
                    st.divider()
            else:
                st.error(f"Error ({res.status_code}): {res.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend server: {e}")

# -----------------------------------------------------------------------------
# TAB 3: MANAGE TEAM DOCUMENTS
# -----------------------------------------------------------------------------
with tab_manage:
    st.header(f"Document Index for `{active_team}`")

    try:
        res = requests.get(f"{API_BASE_URL}/documents", headers=headers, timeout=10)
        if res.status_code == 200:
            docs = res.json()
            
            if isinstance(docs, dict):
                docs = docs.get("documents", [])

            if not docs:
                st.warning(f"No documents found for `{active_team}`.")
            else:
                st.dataframe(pd.DataFrame(docs), use_container_width=True)
                
                doc_options = [
                    d.get("doc_id") or d.get("parent_doc_id") 
                    for d in docs 
                    if "doc_id" in d or "parent_doc_id" in d
                ]
                
                if doc_options:
                    doc_to_delete = st.selectbox("Select Document ID to Delete:", doc_options)
                    if st.button("Delete Document", type="primary"):
                        del_res = requests.delete(f"{API_BASE_URL}/documents/{doc_to_delete}", headers=headers)
                        if del_res.status_code in (200, 204):
                            st.success("Deleted successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error ({del_res.status_code}): {del_res.text}")
        else:
            st.error(f"Error ({res.status_code}): {res.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend server: {e}")