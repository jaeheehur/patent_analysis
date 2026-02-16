import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.parse
from database import SessionLocal, Keyword, Patent, init_db
from scheduler import start_scheduler
import datetime
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import Counter

# Initialize DB and Scheduler
init_db()
if 'scheduler' not in st.session_state:
    st.session_state.scheduler = start_scheduler()

# Page Config
st.set_page_config(
    page_title="Patent Analysis Pro", 
    page_icon="🛡️",
    layout="wide",
)

# Load external CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Helper to load data from DB
def load_patents():
    db = SessionLocal()
    patents = db.query(Patent).all()
    db.close()
    if not patents:
        return pd.DataFrame()
    
    data = [{
        "App Number": p.application_number,
        "Reg Number": p.patent_number,
        "Title": p.title,
        "Abstract": p.abstract if p.abstract and p.abstract != 'nan' else "",
        "Applicant": p.applicant,
        "Filing Date": str(p.filing_date) if p.filing_date else "",
        "Source": p.source_file,
        "Ingested At": p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else ""
    } for p in patents]
    return pd.DataFrame(data)


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
with st.sidebar:
    st.title("🛡️ Formula Engine")
    st.markdown("---")
    
    user_keyword = st.text_input("Enter Search Keywords:", placeholder="e.g., graphene display")
    
    if st.button("Generate WIPS Formula"):
        if user_keyword:
            st.session_state.generated_formula = f"TTL:({user_keyword}) OR ABST:({user_keyword}) OR CLM:({user_keyword})"
        else:
            st.warning("Please enter a keyword.")
            
    if 'generated_formula' in st.session_state:
        st.markdown("---")
        st.subheader("Generated Formula")
        st.code(st.session_state.generated_formula, language="text", wrap_lines=True)
        
        st.markdown(f'<a href="https://www.wipson.com/service/scd/scdView.wips" target="_blank" class="wipson-link">Go to Wipson Search Page</a>', unsafe_allow_html=True)
        st.caption("💡 Copy the formula above and paste it into the WIPS On search bar.")

    # ── Upload Wipson Export (moved from main content) ──
    st.markdown("---")
    with st.expander("📤 Upload Wipson Export", expanded=False):
        st.write("Upload `.xlsx` or `.csv` files exported from WIPS On.")
        uploaded_file = st.file_uploader("Choose a file to ingest", type=['csv', 'xlsx'], label_visibility="collapsed")
        
        if uploaded_file:
            save_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = None
                    for encoding in ['utf-8', 'cp949', 'euc-kr']:
                        try:
                            df = pd.read_csv(save_path, encoding=encoding)
                            # st.success(f"✅ Loaded ({encoding})") # Optional: suppress loading success message if desired, or keep it.
                            break
                        except UnicodeDecodeError:
                            continue
                    if df is None:
                        raise Exception("Could not decode CSV with utf-8, cp949, or euc-kr.")
                else:
                    df = pd.read_excel(save_path)
                
                # Auto-save to Database
                db = SessionLocal()
                new_count = 0
                update_count = 0
                
                # Dictionary to track processed apps in this batch to handle in-file duplicates
                processed_apps = {}

                for _, row in df.iterrows():
                    app_num = str(row.get('Application Number', row.get('출원번호', ''))).strip()
                    if not app_num or app_num.lower() == 'nan':
                        continue
                    
                    # Extract Filing Date
                    # Common column names: 'Filing Date', '출원일', 'App Date'
                    filing_date_val = row.get('Filing Date', row.get('출원일', row.get('App Date', None)))
                    filing_date_obj = None
                    if filing_date_val and str(filing_date_val).lower() != 'nan':
                        try:
                            # Attempt to parse date. Formats can vary (YYYY.MM.DD, YYYY-MM-DD, YYYYMMDD, etc.)
                            # Using pd.to_datetime handles most standard formats
                            dt = pd.to_datetime(filing_date_val, errors='coerce')
                            if not pd.isna(dt):
                                filing_date_obj = dt.date()
                        except Exception:
                            pass # Keep None if parsing fails

                    # Prepare data dict
                    data = {
                        'application_number': app_num,
                        'patent_number': str(row.get('Patent Number', row.get('등록번호', ''))),
                        'title': str(row.get('Title', row.get('발명의 명칭', ''))),
                        'abstract': str(row.get('Abstract', row.get('요약', ''))),
                        'applicant': str(row.get('Applicant', row.get('출원인', ''))),
                        'filing_date': filing_date_obj,
                        'source_file': uploaded_file.name
                    }

                    # Check if we've seen this app_num in this batch
                    if app_num in processed_apps:
                        existing_obj = processed_apps[app_num]
                        for key, value in data.items():
                            setattr(existing_obj, key, value)
                        continue

                    existing = db.query(Patent).filter(Patent.application_number == app_num).first()
                    
                    if existing:
                        for key, value in data.items():
                            setattr(existing, key, value)
                        update_count += 1
                        processed_apps[app_num] = existing
                    else:
                        p = Patent(**data)
                        db.add(p)
                        new_count += 1
                        processed_apps[app_num] = p
                
                db.commit()
                db.close()
                st.success(f"✅ Successfully processed {uploaded_file.name}: Added {new_count}, Updated {update_count} records.")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    st.markdown("---")
    st.write("© 2026 Patent Analysis Pro")


# ═══════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════
st.title("💡 Patent Intelligence Dashboard")
st.markdown("---")

tab1, tab2 = st.tabs(["📂 Patent Database", "📊 Database Explorer"])

# ─── Tab 1: Patent Database (auto-load from DB) ───
with tab1:
    st.header("Patent Database")
    
    df_db = load_patents()
    
    if not df_db.empty:
        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Total Patents", len(df_db))
        with mc2:
            unique_applicants = df_db['Applicant'].nunique()
            st.metric("Unique Applicants", unique_applicants)
        with mc3:
            sources = df_db['Source'].nunique()
            st.metric("Source Files", sources)
        
        st.markdown("---")
        
        # Multi-column Search UI
        st.write("### 🔍 Filter Results")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            q_app = st.text_input("App Number:", placeholder="10-2024...", key="tab1_app")
        with col_s2:
            q_title = st.text_input("Title:", placeholder="마스크...", key="tab1_title")
        with col_s3:
            q_applicant = st.text_input("Applicant:", placeholder="테스...", key="tab1_applicant")
        with col_s4:
            q_source = st.text_input("Source File:", placeholder="TextDown...", key="tab1_source")
            
        # Apply Filters
        filtered_df = df_db.copy()
        if q_app:
            filtered_df = filtered_df[filtered_df['App Number'].str.contains(q_app, case=False, na=False)]
        if q_title:
            filtered_df = filtered_df[filtered_df['Title'].str.contains(q_title, case=False, na=False)]
        if q_applicant:
            filtered_df = filtered_df[filtered_df['Applicant'].str.contains(q_applicant, case=False, na=False)]
        if q_source:
            filtered_df = filtered_df[filtered_df['Source'].str.contains(q_source, case=False, na=False)]
            
        st.write(f"Displaying **{len(filtered_df)}** of **{len(df_db)}** patents.")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.info("📭 The database is currently empty. Upload Wipson export files using the sidebar uploader.")


# ─── Tab 2: Database Explorer (Visualizations) ───
with tab2:
    st.header("Database Explorer")
    
    df_viz = load_patents()
    
    if not df_viz.empty:
        # Filter for last 5 years
        current_year = datetime.datetime.now().year
        start_year = current_year - 5
        
        # Ensure we have a date column to filter
        df_viz['_date'] = pd.to_datetime(df_viz['Filing Date'], errors='coerce')
        df_recent = df_viz[df_viz['_date'].dt.year >= start_year].copy()
        
        st.info(f"📊 Displaying analysis for data from **{start_year}** to **{current_year}** (Last 5 Years).")
        
        if df_recent.empty:
             st.warning(f"No data found from {start_year} onwards.")
        else:
            # Swap df_viz with df_recent for the rest of the visualizations in this tab
            df_viz_all = df_viz # Keep original for reference if needed? No, user wants analysis on this subset.
            df_viz = df_recent 

            # ── Summary Metrics Row ──
            st.markdown("### 📈 Recent Overview (Last 5 Years)")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("📄 Recent Patents", len(df_viz))
            with m2:
                unique_app = df_viz['Applicant'].dropna().replace('nan', pd.NA).dropna().nunique()
                st.metric("🏢 Active Applicants", unique_app)
            with m3:
                dates = pd.to_datetime(df_viz['Filing Date'], errors='coerce').dropna()
                if not dates.empty:
                    st.metric("📅 Date Range", f"{dates.min().year} - {dates.max().year}")
                else:
                    st.metric("📅 Date Range", "N/A")
            with m4:
                has_abstract = df_viz['Abstract'].replace('', pd.NA).replace('nan', pd.NA).dropna().shape[0]
                st.metric("📝 With Abstract", has_abstract)
        
        st.markdown("---")
        
        # ── Top Applicants Bar Chart ──
        st.markdown("### 🏢 Top Applicants")
        
        applicants = df_viz['Applicant'].dropna().replace('nan', pd.NA).dropna()
        if not applicants.empty:
            # Some patents may have multiple applicants separated by ; or |
            all_applicants = []
            for a in applicants:
                parts = [x.strip() for x in str(a).replace('|', ';').split(';')]
                all_applicants.extend([p for p in parts if p and p.lower() != 'nan'])
            
            app_counts = Counter(all_applicants)
            top_n = 15
            top_applicants = app_counts.most_common(top_n)
            
            if top_applicants:
                df_top = pd.DataFrame(top_applicants, columns=['Applicant', 'Count'])
                df_top = df_top.sort_values('Count', ascending=True)
                
                fig_bar = px.bar(
                    df_top, 
                    x='Count', 
                    y='Applicant', 
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Blues',
                    title=f'Top {min(top_n, len(top_applicants))} Patent Applicants'
                )
                fig_bar.update_layout(
                    height=max(400, len(top_applicants) * 35),
                    showlegend=False,
                    coloraxis_showscale=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#000000'), # Darker text
                    yaxis=dict(title=''),
                    xaxis=dict(title='Number of Patents', gridcolor='#eeeeee'),
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("---")
        
        # ── Filing Timeline ──
        st.markdown("### 📅 Filing Timeline")
        
        df_viz['_date'] = pd.to_datetime(df_viz['Filing Date'], errors='coerce')
        dated = df_viz.dropna(subset=['_date']).copy()
        
        if not dated.empty:
            dated['Year'] = dated['_date'].dt.year
            
            # Filing trend per year
            yearly = dated.groupby('Year').size().reset_index(name='Patents')
            
            fig_timeline = px.area(
                yearly, 
                x='Year', 
                y='Patents',
                title='Patent Filing Trend by Year',
                markers=True,
                color_discrete_sequence=['#0066cc']
            )
            fig_timeline.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#000000'), # Darker text
                xaxis=dict(gridcolor='#eeeeee', dtick=1),
                yaxis=dict(gridcolor='#eeeeee', title='Number of Patents'),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Scatter: individual patents on timeline
            dated_display = dated.copy()
            dated_display['Month'] = dated_display['_date'].dt.month
            dated_display['Label'] = dated_display['Title'].str[:40]
            
            fig_scatter = px.scatter(
                dated_display,
                x='_date',
                y='Applicant',
                size_max=10,
                hover_data={'Title': True, 'App Number': True, '_date': False},
                color='Applicant',
                title='Individual Patent Filing Timeline',
                labels={'_date': 'Filing Date'}
            )
            fig_scatter.update_layout(
                height=max(400, dated_display['Applicant'].nunique() * 30),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#000000'), # Darker text
                xaxis=dict(gridcolor='#eeeeee'),
                yaxis=dict(gridcolor='#eeeeee', title=''),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("📅 No filing date available for timeline visualization.")
        
        st.markdown("---")
        
        # ── Keyword Network Graph ──
        st.markdown("### 🕸️ Technology Keyword Network")
        st.caption("Visualizes relationships between key technologies based on patent titles. Keywords appearing together in the same title are connected.")
        
        # Helper to extract keywords
        import re
        def extract_keywords(text):
            if not isinstance(text, str):
                return []
            # Basic stopwords (English + Korean common patent terms)
            stopwords = set([
                'method', 'device', 'apparatus', 'system', 'composition', 'manufacturing', 'producing', 
                'comprising', 'including', 'related', 'using', 'process', 'preparation', 'structure',
                'material', 'layer', 'substrate', 'having', 'provided', 'thereof', 'according', 'first', 'second',
                '방법', '장치', '시스템', '조성물', '제조', '생산', '포함하는', '이용하는', '관련', '구조', '물질', '재료', 
                '기판', '층', '상기', '및', '의', '에', '를', '을', '은', '는', '이', '가', '으로', '로', '에서', '에게', '따른'
            ])
            
            # Normalize and tokenize
            text = text.lower()
            # Split by non-alphanumeric (keep Hangul)
            tokens = re.findall(r'[a-z0-9가-힣]+', text)
            
            keywords = [t for t in tokens if len(t) > 1 and t not in stopwords and not t.isdigit()]
            return keywords

        # Build network
        titles = df_viz['Title'].dropna()
        if not titles.empty:
            G = nx.Graph()
            
            # Document frequency counter for node sizing
            keyword_counts = Counter()
            
            # Edges
            edge_counts = Counter()
            
            for title in titles:
                kws = extract_keywords(title)
                kws = list(set(kws)) # Unique per title
                
                for k in kws:
                    keyword_counts[k] += 1
                
                # Co-occurrence
                for i in range(len(kws)):
                    for j in range(i + 1, len(kws)):
                        pair = tuple(sorted([kws[i], kws[j]]))
                        edge_counts[pair] += 1
            
            # Filter top keywords to avoid clutter
            # Top N likely keywords
            top_n_keywords = 50
            most_common_kws = dict(keyword_counts.most_common(top_n_keywords))
            
            if len(most_common_kws) > 1:
                # Add nodes
                for k, count in most_common_kws.items():
                    G.add_node(k, count=count)
                
                # Add edges if both nodes are in top N
                for (u, v), w in edge_counts.items():
                    if u in most_common_kws and v in most_common_kws:
                        G.add_edge(u, v, weight=w)
                
                # Filter isolated nodes if any
                # G = G.subgraph([n for n in G.nodes if G.degree(n) > 0]).copy()

                if len(G.nodes) > 0:
                    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
                    
                    # Edges trace
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#cccccc'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Nodes trace
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    node_color = []
                    
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        count = G.nodes[node].get('count', 1)
                        node_text.append(f"{node}<br>Freq: {count}")
                        node_size.append(max(15, min(60, count * 3)))
                        node_color.append(count)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=[n for n in G.nodes()],
                        textposition="top center",
                        textfont=dict(size=10, color='#000000'), # Dark text
                        hovertext=node_text,
                        marker=dict(
                            showscale=True,
                            colorscale='Viridis',
                            size=node_size,
                            color=node_color,
                            colorbar=dict(
                                thickness=15,
                                title='Frequency',
                                xanchor='left',
                            ),
                            line=dict(width=1, color='#ffffff')
                        )
                    )
                    
                    fig_network = go.Figure(data=[edge_trace, node_trace])
                    fig_network.update_layout(
                        title=dict(
                            text='Technology Keyword Co-occurrence',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        height=600,
                        plot_bgcolor='rgba(248,248,248,0.8)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#000000'), # Dark text
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    st.plotly_chart(fig_network, use_container_width=True)
                else:
                    st.info("Not enough data to build a keyword network.")
            else:
                st.info("Not enough frequent keywords found for network analysis.")
        else:
            st.info("No titles available for keyword analysis.")
        
    else:
        st.info("📭 The database is currently empty. Upload Wipson export files using the sidebar uploader to see visualizations.")
