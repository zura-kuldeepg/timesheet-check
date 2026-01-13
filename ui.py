import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import base64
import os
import io
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Format Batch Dashboard", layout="wide")

# --- CONFIGURATION ---
DEFAULT_JSON_FILE = "s3_job_result_20260108_232054_dd1fdb79.json"

# ------------------------------------------------------------------
# --- AWS CREDENTIALS CONFIGURATION --------------------------------
# ------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID","")  
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY","") 
AWS_REGION = os.getenv("AWS_REGION","")
BUCKET_NAME = os.getenv("BUCKET_NAME","")



@st.cache_data
def process_json(uploaded_file):
    data = json.load(uploaded_file)
    results = data.get("results", [])
    
    flattened_data = []
    for r in results:
        if not r: continue

        # 1. PARSE S3 KEY & BUCKET
        raw_s3_key = r.get("s3_key") or ""
        
        # Logic to handle "s3://" format vs relative path
        if raw_s3_key.startswith("s3://"):
            # Format: s3://bucket-name/folder/file.ext
            try:
                stripped = raw_s3_key[5:] # Remove 's3://'
                split_parts = stripped.split("/", 1) # Split into [bucket, key]
                actual_bucket = split_parts[0]
                actual_key = split_parts[1]
            except IndexError:
                # Fallback if format is weird
                actual_bucket = BUCKET_NAME
                actual_key = raw_s3_key
        else:
            # Format: folder/file.ext (Use default bucket)
            actual_bucket = BUCKET_NAME
            actual_key = raw_s3_key

        # 2. IDENTIFY FILE TYPE
        filename = r.get("input_filename") or ""
        flow = r.get("flow") or []
        process_type = "Image/Scanned"
        for step in flow:
            if isinstance(step, dict) and "pdf_scanned_check" in step:
                if "vector/text PDF" in step["pdf_scanned_check"]:
                    process_type = "Native PDF (Vector)"

        # 3. DATE EXTRACTION
        file_timestamp = pd.NaT
        try:
            name_clean = filename.rsplit('.', 1)[0]
            date_part = name_clean.split('_')[-1]
            if len(date_part) == 14 and date_part.isdigit():
                file_timestamp = datetime.strptime(date_part, "%m%d%Y%H%M%S")
        except Exception:
            pass

        # 4. QUALITY SCORE
        cqs = r.get("custom_quality_score") or {}
        q_status = cqs.get("status", "N/A").lower() 
        q_reason = cqs.get("reason", "N/A")

        # 5. GENERAL & METADATA
        analysis = r.get("analysis") or {}
        ocr = r.get("ocr") or {}
        timings = r.get("timings") or {}
        
        # Extract Project/Month from the CLEAN key
        # Example Key: ZuraTM/PRIME_STAFFING/Ingested/November-2025/file.pdf
        key_parts = actual_key.split('/')
        project = key_parts[1] if len(key_parts) > 1 else "Unknown"
        file_ext = actual_key.split('.')[-1].upper() if '.' in actual_key else "IMG"

        flattened_data.append({
            "Filename": filename,
            "Timestamp": file_timestamp,
            "Decision": r.get("decision") or "N/A",
            "Type": process_type,
            "Quality_Status": q_status,
            "Quality_Reason": q_reason,
            "Confidence": analysis.get("overall_confidence") or 100 if process_type == "Native PDF (Vector)" else analysis.get("overall_confidence", 0),
            "Project": project,
            "File_Extension": file_ext,
            "Total_Duration": timings.get("total_duration", 0),
            "Text_Length": len(ocr.get("text") or ""),
            "S3_Bucket": actual_bucket, # Store the specific bucket
            "S3_Key": actual_key        # Store the clean key
        })
    
    df = pd.DataFrame(flattened_data)
    if not df.empty and 'Timestamp' in df.columns:
        df = df.sort_values(by='Timestamp', ascending=False)
        
    return data, df

@st.cache_data(show_spinner=False)
def get_s3_file_bytes(bucket, key):
    try:
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
        elif "aws" in st.secrets:
            s3 = boto3.client('s3', aws_access_key_id=st.secrets["aws"]["access_key_id"], aws_secret_access_key=st.secrets["aws"]["secret_access_key"], region_name=st.secrets["aws"].get("region", "us-east-1"))
        else:
            s3 = boto3.client('s3')
        
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read(), None
    except Exception as e:
        return None, str(e)

def display_file(file_bytes, file_ext, error_msg, download_filename="file"):
    """
    Smart display function that handles PDF, Images, and falls back to Download for DOCX/Others.
    """
    if error_msg:
        st.error(f"Could not load file: {error_msg}")
        return

    if not file_bytes:
        st.info("No file data found.")
        return

    # Normalize extension
    ext = file_ext.upper().replace(".", "")
    
    # 1. Handle PDF
    if ext == "PDF":
        base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    # 2. Handle Known Images
    elif ext in ["JPG", "JPEG", "PNG", "BMP", "TIFF", "WEBP"]:
        try:
            st.image(file_bytes, use_container_width=True)
        except Exception:
            st.error(f"Error rendering image ({ext}). The file might be corrupted.")
            st.download_button(f"Download {ext}", data=file_bytes, file_name=download_filename)

    # 3. Handle DOCX, TXT, JSON, etc. (Fallback)
    else:
        st.warning(f"Preview not available for **{ext}** files.")
        st.download_button(
            label=f"📥 Download {download_filename}",
            data=file_bytes,
            file_name=download_filename,
            mime="application/octet-stream"
        )

def filter_dataframe(df, quality_val, type_val):
    temp_df = df.copy()
    if quality_val == "pass":
        temp_df = temp_df[temp_df['Quality_Status'] == 'pass']
    elif quality_val == "fail":
        temp_df = temp_df[temp_df['Quality_Status'] == 'fail']
    elif quality_val == "na":
        temp_df = temp_df[~temp_df['Quality_Status'].isin(['pass', 'fail'])]
    
    if type_val == "pdf":
        temp_df = temp_df[temp_df['Type'] == 'Native PDF (Vector)']
    elif type_val == "image":
        temp_df = temp_df[temp_df['Type'] == 'Image/Scanned']
        
    return temp_df

# --- NAVIGATION CALLBACKS ---
def next_file():
    st.session_state.file_index += 1

def prev_file():
    st.session_state.file_index -= 1

def main():
    st.title("📂 Document Processing Dashboard")
    
    # Initialize Session State
    if 'filter_quality' not in st.session_state: st.session_state.filter_quality = 'All'
    if 'filter_type' not in st.session_state: st.session_state.filter_type = 'All'
    if 'file_index' not in st.session_state: st.session_state.file_index = 0

    # ==========================================
    # SIDEBAR: INPUT & FILTERS
    # ==========================================
    st.sidebar.header("1. Data Input")
    uploaded_file  = st.sidebar.file_uploader("Upload Response JSON", type="json")
    
    file = None
    
    if uploaded_file:
        file = uploaded_file
    elif os.path.exists(DEFAULT_JSON_FILE):
        # Load local file into BytesIO to mimic an uploaded file
        with open(DEFAULT_JSON_FILE, "rb") as f:
            file_content = f.read()
            file = io.BytesIO(file_content)
            # Optional: Set a name attribute if your logic ever needs it
            file.name = DEFAULT_JSON_FILE 
        st.sidebar.info(f"Using default file: {DEFAULT_JSON_FILE}")
    else:
        st.sidebar.warning(f"No upload and '{DEFAULT_JSON_FILE}' not found.")

    if file:
        with st.spinner("Processing records..."):
            raw_data, df = process_json(file)

        # --- DATE FILTER ---
        st.sidebar.divider()
        st.sidebar.header("2. Date Range")
        valid_dates = df['Timestamp'].dropna()
        
        df_date_filtered = df.copy()
        
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
                df_date_filtered = df[mask]

        if df_date_filtered.empty:
            st.warning("No records found for the selected date range.")
            return

        # ==========================================
        # TOP LEVEL KPI SECTION
        # ==========================================
        
        total_files = len(df_date_filtered)
        total_duration_sec = df_date_filtered['Total_Duration'].sum()
        avg_duration_sec = df_date_filtered['Total_Duration'].mean()
        pass_count = len(df_date_filtered[df_date_filtered['Quality_Status'] == 'pass'])
        pass_rate = (pass_count / total_files * 100) if total_files > 0 else 0

        if total_duration_sec >= 3600:
            total_time_str = f"{total_duration_sec / 3600:.2f} hr"
        elif total_duration_sec >= 60:
            total_time_str = f"{total_duration_sec / 60:.1f} min"
        else:
            total_time_str = f"{total_duration_sec:.2f} s"


        m1, m2, m3 = st.columns(3)
        m1.metric("Total Files", total_files)
        m2.metric("Total Processing Time", total_time_str)
        m3.metric("Avg Time / File", f"{avg_duration_sec:.2f} s")
        
        st.divider()

        # ==========================================
        # CROSS-FILTERING LOGIC
        # ==========================================
        
        # 1. Quality Counts
        df_for_quality_counts = filter_dataframe(df_date_filtered, "All", st.session_state.filter_type)
        q_counts = {
            "All": len(df_for_quality_counts),
            "pass": len(df_for_quality_counts[df_for_quality_counts['Quality_Status'] == 'pass']),
            "fail": len(df_for_quality_counts[df_for_quality_counts['Quality_Status'] == 'fail']),
            "na": len(df_for_quality_counts[~df_for_quality_counts['Quality_Status'].isin(['pass', 'fail'])])
        }

        def format_quality_label(option):
            label_map = {
                "All": f"All ({q_counts['All']})",
                "pass": f"✅ Passed ({q_counts['pass']})",
                "fail": f"❌ Failed ({q_counts['fail']})",
                "na": f"Not Graded(vector pdfs & docx) ({q_counts['na']})"
            }
            return label_map.get(option, option)

        st.sidebar.subheader("3. Filter by Quality")
        st.sidebar.radio(
            "Quality Status",
            options=["All", "pass", "fail", "na"],
            format_func=format_quality_label,
            key="filter_quality"
        )

        # 2. Type Counts
        df_for_type_counts = filter_dataframe(df_date_filtered, st.session_state.filter_quality, "All")
        t_counts = {
            "All": len(df_for_type_counts),
            "pdf": len(df_for_type_counts[df_for_type_counts['Type'] == 'Native PDF (Vector)']),
            "image": len(df_for_type_counts[df_for_type_counts['Type'] == 'Image/Scanned'])
        }

        def format_type_label(option):
            label_map = {
                "All": f"All ({t_counts['All']})",
                "pdf": f"📄 Native PDFs ({t_counts['pdf']})",
                "image": f"🖼️ Images ({t_counts['image']})"
            }
            return label_map.get(option, option)

        st.sidebar.subheader("4. Filter by Type")
        st.sidebar.radio(
            "File Type",
            options=["All", "pdf", "image"],
            format_func=format_type_label,
            key="filter_type"
        )

        # ==========================================
        # APPLY FINAL FILTERS
        # ==========================================
        df_final = filter_dataframe(df_date_filtered, st.session_state.filter_quality, st.session_state.filter_type)

        # ==========================================
        # MAIN DASHBOARD AREA
        # ==========================================
        
        st.subheader(f"Viewing: {len(df_final)} Files")
        
        if not df_final.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**Quality Distribution (Selection)**")
                status_counts = df_final['Quality_Status'].value_counts().reset_index()
                color_map = {"pass": "#2ecc71", "fail": "#e74c3c", "n/a": "#95a5a6"}
                st.plotly_chart(px.pie(status_counts, values='count', names='Quality_Status', 
                                     color='Quality_Status', color_discrete_map=color_map, hole=0.4), use_container_width=True)
            with col_b:
                st.write(f"**File Type Distribution (Selection)**")
                type_counts = df_final['Type'].value_counts().reset_index()
                st.plotly_chart(px.bar(type_counts, x='Type', y='count', color='Type'), use_container_width=True)
        else:
            st.info("No files match this combination of filters.")

        # --- INDIVIDUAL INSPECTOR ---
        st.divider()
        st.header("🔍 Individual File Inspector")
        
        if df_final.empty:
            st.write("No files to display.")
        else:
            file_options = df_final['Filename'].tolist()
            total_files = len(file_options)

            # --- NAVIGATION LOGIC ---
            if st.session_state.file_index >= total_files:
                st.session_state.file_index = 0
            
            col_prev, col_sel, col_next = st.columns([1, 6, 1])

            with col_prev:
                st.button("⬅️ Prev", on_click=prev_file, disabled=(st.session_state.file_index == 0))

            with col_next:
                st.button("Next ➡️", on_click=next_file, disabled=(st.session_state.file_index == total_files - 1))

            with col_sel:
                selected_filename = st.selectbox(
                    "Select File", 
                    options=file_options, 
                    index=st.session_state.file_index,
                    label_visibility="collapsed"
                )
                
                current_selection_index = file_options.index(selected_filename)
                if current_selection_index != st.session_state.file_index:
                    st.session_state.file_index = current_selection_index
                    st.rerun()

            # --- DISPLAY RECORD ---
            record = next((r for r in raw_data["results"] if r["input_filename"] == selected_filename), None)
            
            if record:
                analysis = record.get('analysis') or {}
                ocr = record.get('ocr') or {}
                flow = record.get('flow') or []
                
                file_row = df_final[df_final['Filename'] == selected_filename].iloc[0]
                
                # Status Badge
                status_color = "green" if file_row['Quality_Status'] == 'pass' else "red"
                if file_row['Quality_Status'] == 'n/a': status_color = "grey"

                st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:rgba(200,200,200,0.1); border-left: 5px solid {status_color};">
                        <strong>File {st.session_state.file_index + 1} of {total_files}</strong><br>
                        <strong>Date:</strong> {file_row['Timestamp']} &nbsp;|&nbsp; 
                        <strong>Quality Status:</strong> <span style="color:{status_color}">{file_row['Quality_Status'].upper()}</span> &nbsp;|&nbsp; 
                        <strong>Type:</strong> {file_row['Type']}
                    </div>
                    <br>
                """, unsafe_allow_html=True)

                if file_row['Quality_Status'] == 'fail':
                    st.error(f"**Failure Reason:** {file_row['Quality_Reason']}")

                tab1, tab2, tab3 = st.tabs(["📋 Process Flow & Content", "📊 Quality Metrics", "🛠 Technical Info"])
                
                with tab1:
                    col_flow, col_vis, col_text = st.columns([1, 2, 2])
                    with col_flow:
                        st.markdown("### ⚙️ Flow")
                        for step in flow:
                            if isinstance(step, dict):
                                for k, v in step.items():
                                    st.success(f"**{k.replace('_', ' ')}**\n{v}")
                            else:
                                st.info(step)
                    
                    with col_vis:
                        st.markdown("### 🖼️ Document Viewer")
                        s3_bucket = file_row['S3_Bucket']
                        s3_key = file_row['S3_Key']
                        
                        if s3_key:
                            with st.spinner(f"Fetching from {s3_bucket}..."):
                                file_bytes, error_msg = get_s3_file_bytes(s3_bucket, s3_key)
                                display_file(file_bytes, file_row['File_Extension'], error_msg, file_row['Filename'])
                        else:
                            st.warning("No S3 Key found.")

                    with col_text:
                        st.markdown("### 📝 Extracted Text")
                        st.text_area("Raw OCR Output", ocr.get("text", ""), height=600)

                with tab2:
                    if analysis:
                        # 1. BASIC METRICS
                        st.subheader("Basic Metrics")
                        basic_data = analysis.get('basic_metrics', {})
                        if basic_data:
                            # FIX: Convert all values to string to avoid ArrowInvalid error
                            df_basic = pd.DataFrame(
                                [{"Metric": k.replace("_", " ").title(), "Value": str(v)} for k, v in basic_data.items()]
                            )
                            st.dataframe(df_basic, hide_index=True, use_container_width=True)
                        else:
                            st.info("No basic metrics available.")

                        st.divider()

                        # 2. SEPARATE QUALITY TABLES
                        st.subheader("Quality Assessment Details")
                        
                        quality_keys = [
                            'sharpness', 'contrast', 'resolution', 'noise', 'skew', 
                            'lighting', 'binarization', 'text_density', 
                            'background_uniformity', 'artifacts'
                        ]
                        
                        cols = st.columns(2)
                        col_idx = 0

                        found_any = False
                        for key in quality_keys:
                            if key in analysis:
                                found_any = True
                                section_data = analysis[key]
                                # FIX: Convert all values to string to avoid ArrowInvalid error
                                rows = [{"Parameter": k.replace("_", " ").title(), "Value": str(v)} for k, v in section_data.items()]
                                df_cat = pd.DataFrame(rows)
                                
                                with cols[col_idx % 2]:
                                    st.markdown(f"**{key.title()}**")
                                    st.dataframe(df_cat, hide_index=True, use_container_width=True)
                                    st.write("") 
                                
                                col_idx += 1
                        
                        if not found_any:
                            st.info("No quality assessment metrics found (likely a Native PDF).")

                    else:
                        st.warning("Analysis data not available.")

                with tab3:
                    st.write("**Processing Timings:**")
                    st.json(record.get('timings', {}))
                    with st.expander("Full JSON Record"):
                        st.json(record)

    else:
        st.info("Upload JSON to begin.")

if __name__ == "__main__":
    main()