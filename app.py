import streamlit as st
import pandas as pd
import numpy as np
import re
import cv2

st.set_page_config(page_title="CosmeticOCR", layout="wide", initial_sidebar_state="collapsed")
from streamlit_sortables import sort_items
from OCRD1 import (
    process_single_product, save_checkpoint, load_history_data,
    update_sales_in_final_file, get_platform_column, match_fda 
)
from collections import OrderedDict

filename = 'OCRcos13.xlsx'
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:      #ffffff;
    --surface: #f8f7f5;
    --border:  #e8e5e0;
    --accent:  #f97316;
    --text:    #1a1a1a;
    --muted:   #9ca3af;
}

html, body, [class*="css"] {
    font-family: 'Kanit', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

div.stButton > button {
    height: 90px !important; font-size: 15px !important; font-weight: 500 !important;
    font-family: 'Kanit', sans-serif !important;
    border-radius: 14px !important; border: 1px solid var(--border) !important;
    background: var(--surface) !important; color: var(--text) !important;
    transition: all .2s !important;
}
div.stButton > button:hover {
    border-color: var(--accent) !important; background: #fff4ed !important;
}
button[kind="primary"] {
    background: var(--accent) !important; border: none !important;
    border-radius: 10px !important; font-family: 'Kanit', sans-serif !important;
    font-size: 15px !important; font-weight: 600 !important;
    height: 52px !important; color: #fff !important;
}
.sec {
    font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 500;
    letter-spacing: .2em; text-transform: uppercase; color: var(--accent);
    margin: 0 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
.thumb-wrap { overflow: hidden; border-radius: 10px; border: 1px solid var(--border); }
.thumb-wrap img { max-height: 90px !important; width: 100% !important; object-fit: cover !important; }
.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }
.stExpander { border: 1px solid var(--border) !important; border-radius: 12px !important; background: var(--bg) !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden; border: 1px solid var(--border) !important; }
[data-testid="stMetricValue"] {
    font-family: 'Kanit', sans-serif !important; font-size: 30px !important; font-weight: 700 !important; color: var(--text) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important;
    color: var(--muted) !important; letter-spacing: .08em;
}
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

section.main > div > div:first-child > div > div:nth-child(2) div.stButton > button {
    height: 36px !important; font-size: 10px !important;
    border-color: #fca5a5 !important; color: #dc2626 !important;
    background: #fff !important;
}
</style>
""", unsafe_allow_html=True)

SESSION_DEFAULTS = {
    "menu_select":        None,
    "saved_photos":       [],
    "uploaded_file_data": [],
    "group_layout":       [],
    "imgs_per_product":   2,
    "last_upload_hash":   None,
    "ocr_raw_text":       "",
    "ocr_img_bytes_list": [],
    "extract_result":     None,
}

def reset_all():
    for k, v in SESSION_DEFAULTS.items():
        st.session_state[k] = v.copy() if isinstance(v, list) else v

for k, v in SESSION_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v.copy() if isinstance(v, list) else v

def auto_layout(total, n):
    return [list(range(g*n, min(g*n+n, total))) for g in range((total+n-1)//n)]

# ── HEADER ──
col_title, col_reset = st.columns([7, 1])
with col_title:
    st.markdown('<p class="sec">CosmeticOCR</p>', unsafe_allow_html=True)
with col_reset:
    if st.button("🔄 Reset", key="btn_global_reset", use_container_width=True):
        reset_all()
        st.rerun()

# ── MAIN MENU ────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🖼️ Upload Picture", use_container_width=True):
        st.session_state.menu_select       = "upload_pic"
        st.session_state.uploaded_file_data = []
        st.session_state.group_layout      = []
        st.session_state.last_upload_hash  = None  # BUG FIX #1
with c2:
    if st.button("📸 Take Picture", use_container_width=True):
        st.session_state.menu_select = "take_pic"
with c3:
    if st.button("📂 Upload File", use_container_width=True):
        st.session_state.menu_select = "upload_file"


# ══════════════════════════════════════════════════════════
#  UPLOAD PIC
# ══════════════════════════════════════════════════════════
if st.session_state.menu_select == "upload_pic":
    st.markdown("---")

    st.markdown('<p class="sec">1. เลือกจำนวนรูปต่อสินค้า (เลือกก่อนอัปโหลด)</p>', unsafe_allow_html=True)
    n = st.radio(
        "", [2, 3, 4, 5, 6],
        index=[2, 3, 4, 5, 6].index(st.session_state.imgs_per_product),
        horizontal=True, key="n_radio",
        format_func=lambda x: f"  {x} รูป  ",
        label_visibility="collapsed",
    )
    st.session_state.imgs_per_product = n

    st.write("")
    st.markdown('<p class="sec">2. อัปโหลดรูปภาพ</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "ลากรูปมาวางที่นี่", type=["jpg", "png", "jpeg"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    # BUG FIX #1: hash จาก (name, size) แม่นกว่า name อย่างเดียว
    current_hash = hash(tuple((f.name, f.size) for f in uploaded_files)) if uploaded_files else 0

    if current_hash != st.session_state.last_upload_hash:
        st.session_state.last_upload_hash = current_hash
        if uploaded_files:
            st.session_state.uploaded_file_data = [(f.name, f.getvalue()) for f in uploaded_files]
            st.session_state.group_layout = auto_layout(len(uploaded_files), st.session_state.imgs_per_product)
        else:
            st.session_state.uploaded_file_data = []
            st.session_state.group_layout = []

    file_data    = st.session_state.uploaded_file_data
    group_layout = st.session_state.group_layout

    if not file_data:
        st.info("📂 ลากรูปภาพมาวางด้านบนเพื่อเริ่มต้น")
        st.stop()

    total = len(file_data)
    st.markdown("---")

    # ── label helpers ──────────────────────────────────────
    def make_label(idx):
        return f"#{idx+1}  {file_data[idx][0]}"

    def parse_label(label):
        return int(label.split()[0].replace("#", "")) - 1

    sortable_input = [
        {"header": f"สินค้าที่ {gi+1}", "items": [make_label(idx) for idx in grp]}
        for gi, grp in enumerate(group_layout) if grp
    ]

    col_drag, col_prev = st.columns([1, 1], gap="medium")

    with col_drag:
        st.markdown('<p class="sec">จัดกลุ่ม — ลากสลับได้</p>', unsafe_allow_html=True)
        st.caption("ลากชื่อไฟล์ข้ามกลุ่มเพื่อสลับ / ย้ายตำแหน่ง")

        sorted_result = sort_items(
            sortable_input,
            multi_containers=True,
            direction="vertical",
            key="sortable_groups",
        )

        # BUG FIX #2: เปรียบเทียบก่อน overwrite — ป้องกัน layout reset ทุก rerun
        new_layout = []
        for grp_data in sorted_result:
            indices = [parse_label(lbl) for lbl in grp_data["items"]]
            if indices:
                new_layout.append(indices)

        if new_layout and new_layout != st.session_state.group_layout:
            st.session_state.group_layout = new_layout
            group_layout = new_layout

    with col_prev:
        st.markdown('<p class="sec">Preview & ลบรูป</p>', unsafe_allow_html=True)

        for gi, grp in enumerate(group_layout):
            if not grp:
                continue
            st.markdown(
                f'<p style="font-family:JetBrains Mono,monospace;font-size:10px;'
                f'color:#f97316;letter-spacing:.08em;margin:8px 0 4px;">'
                f'สินค้าที่ {gi+1} '
                f'<span style="background:rgba(249,115,22,.15);color:#fb923c;'
                f'border-radius:99px;padding:1px 7px;font-size:9px;">{len(grp)} รูป</span></p>',
                unsafe_allow_html=True,
            )

            # รูปเล็กลง: แสดง max 8 คอลัมน์ + CSS จำกัดความสูง
            thumb_cols = st.columns(min(max(len(grp), 1), 8))
            for ci, idx in enumerate(grp):
                _, fbytes = file_data[idx]
                with thumb_cols[ci]:
                    st.markdown('<div class="thumb-wrap">', unsafe_allow_html=True)
                    st.image(fbytes, use_container_width=True, caption=f"#{idx+1}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # BUG FIX #3: key ใช้ (gi, ci) ไม่ใช่ idx เพื่อไม่ duplicate key หลัง drag
                    if st.button("🗑️", key=f"del_{gi}_{ci}", use_container_width=True):
                        st.session_state.uploaded_file_data.pop(idx)

                        new_gl = []
                        for g in st.session_state.group_layout:
                            ng = [
                                (i - 1 if i > idx else i)
                                for i in g if i != idx
                            ]
                            if ng:
                                new_gl.append(ng)
                        st.session_state.group_layout = new_gl
                        st.session_state.last_upload_hash = None  # force re-accept ไฟล์ใหม่
                        st.rerun()

    st.markdown("---")

    ready = [g for g in group_layout if g]
    if st.button(f"🚀 เริ่ม OCR · {len(ready)} สินค้า", type="primary", use_container_width=True):
        all_results = []
        existing_fdas = load_history_data(filename)
        bar  = st.progress(0)
        stat = st.empty()

        for gi, grp in enumerate(ready):
            stat.markdown(
                f'<p style="color:#94a3b8;font-size:13px;">⏳ สินค้าที่ {gi+1}/{len(ready)}…</p>',
                unsafe_allow_html=True,
            )
            group_bytes = [file_data[i][1] for i in grp]

            result = process_single_product(
                links=[], existing_fdas=existing_fdas,
                output_filename=filename, raw_contents=group_bytes,
            )
            if result:
                if result.get("is_duplicate"):
                    st.warning(f"สินค้าที่ {gi+1}: FDA **{result.get('FDA_Number')}** ซ้ำในระบบ")
                else:
                    all_results.append(result)
                    save_checkpoint([result], filename)
                    st.success(f"✅ สินค้าที่ {gi+1} บันทึกแล้ว")
            bar.progress((gi + 1) / len(ready))

        stat.markdown('<p style="color:#22c55e;font-weight:600;">🎉 เสร็จสิ้น!</p>', unsafe_allow_html=True)
        if all_results:
            st.markdown('<p class="sec">ผลลัพธ์</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(all_results), use_container_width=True)


# ══════════════════════════════════════════════════════════
#  TAKE PIC
# ══════════════════════════════════════════════════════════
if st.session_state.menu_select == "take_pic":
    st.markdown("---")

    st.markdown(
        '<p class="sec">1. ถ่ายรูปภาพสินค้า '
        '<span style="background:rgba(249,115,22,.15);color:#fb923c;'
        'border-radius:99px;padding:1px 8px;font-size:9px;">'
        'ถ่ายได้หลายรูปต่อ 1 สินค้า</span></p>',
        unsafe_allow_html=True,
    )

    picture = st.camera_input("ถ่ายรูปสินค้า", label_visibility="collapsed")

    col_save, col_clear = st.columns([1, 1])
    with col_save:
        if picture:
            if st.button("➕ บันทึกรูปนี้", type="primary", use_container_width=True):
                st.session_state.saved_photos.append(picture.getvalue())
                st.rerun()
    with col_clear:
        if st.session_state.saved_photos:
            if st.button("🗑️ Clear รูปทั้งหมด", use_container_width=True):
                st.session_state.saved_photos        = []
                st.session_state.ocr_raw_text        = ""
                st.session_state.ocr_img_bytes_list  = []
                st.session_state.extract_result      = None
                st.rerun()

    if st.session_state.saved_photos:
        st.markdown("---")
        st.markdown(
            f'<p class="sec">รูปที่บันทึก — {len(st.session_state.saved_photos)} รูป</p>',
            unsafe_allow_html=True,
        )
        thumb_cols = st.columns(min(len(st.session_state.saved_photos), 6))
        for i, img_bytes in enumerate(st.session_state.saved_photos):
            with thumb_cols[i % 6]:
                st.markdown('<div class="thumb-wrap">', unsafe_allow_html=True)
                st.image(img_bytes, use_container_width=True, caption=f"รูปที่ {i+1}")
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_photo_{i}", use_container_width=True):
                    st.session_state.saved_photos.pop(i)
                    st.rerun()

        st.markdown("---")

        if st.button(
            f"🔍 ประมวลผล OCR · {len(st.session_state.saved_photos)} รูป",
            type="primary", use_container_width=True,
        ):
            st.session_state.ocr_img_bytes_list = list(st.session_state.saved_photos)

            with st.spinner("กำลัง Extract & จัด Classification…"):
                existing_fdas = load_history_data(filename)
                result = process_single_product(
                    links=[], existing_fdas=existing_fdas,
                    output_filename=filename,
                    raw_contents=st.session_state.ocr_img_bytes_list,
                )

            st.markdown("---")

            if not result:
                st.error("❌ ไม่พบ Pattern เลขอย. — ข้ามรายการนี้")
            elif result.get("is_duplicate"):
                st.warning(
                    f"⚠️ พบ Pattern เลขอย. **{result.get('FDA_Number')}** "
                    f"แต่ซ้ำในระบบแล้ว — ไม่บันทึก"
                )
                st.markdown('<p class="sec">ข้อมูลที่พบ (ไม่บันทึกซ้ำ)</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([result]), use_container_width=True)
            else:
                save_checkpoint([result], filename)
                st.success("✅ จัด Classification และบันทึกลง Excel เรียบร้อย!")
                st.markdown('<p class="sec">ผลลัพธ์ — Classification</p>', unsafe_allow_html=True)
                st.dataframe(pd.DataFrame([result]), use_container_width=True)
# ══════════════════════════════════════════════════════════
#  UPLOAD FILE
# ══════════════════════════════════════════════════════════
if st.session_state.menu_select == "upload_file":
    st.markdown("---")

    st.markdown('<p class="sec">1. อัปโหลดไฟล์ CSV / Excel</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "ลากไฟล์มาวาง", type=["xlsx", "csv"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    if uploaded_files:
        all_dfs = []
        for file in uploaded_files:
            df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
            st.markdown(f'<p class="sec">{file.name}</p>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            all_dfs.append(df)

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)

            st.markdown("---")
            st.markdown('<p class="sec">2. เลือก Column ที่มี URL รูปภาพ</p>', unsafe_allow_html=True)
            url_column = st.selectbox(
                "เลือก column", options=combined_df.columns.tolist(),
                label_visibility="collapsed",
            )

            
            brand_col   = None
            product_col = None
            for col in combined_df.columns:
                c = str(col).strip().lower()
                if 'brand' in c:
                    brand_col = col
                if any(k in c for k in ['product', 'สินค้า', 'ชื่อสินค้า']):
                    product_col = col

            st.markdown("---")
            if st.button("🚀 ประมวลผล", type="primary", use_container_width=True):
                existing_fdas = load_history_data(filename)
                urls          = combined_df[url_column].dropna().tolist()
                bar           = st.progress(0)
                stat          = st.empty()
                summary_ph    = st.empty()
                summary_rows  = []

                for i, raw_val in enumerate(urls):
                    stat.markdown(
                        f'<p style="color:#94a3b8;font-size:13px;">'
                        f'⏳ กำลังประมวลผลแถวที่ {i+1} / {len(urls)}…</p>',
                        unsafe_allow_html=True,
                    )

                    clean_val = re.sub(
                        r",\s*(http)", r" \1",
                        str(raw_val).replace("|", " ").replace("\n", " "),
                    )
                    unique_links = list(OrderedDict.fromkeys(
                        [l.strip() for l in clean_val.split() if l.strip().startswith("http")]
                    ))

                    if not unique_links:
                        summary_rows.append({
                            "แถว": i + 1, "สถานะ": "⏭️ ข้าม",
                            "เหตุผล": "ไม่พบ URL",
                            "FDA Number": "-", "Classification": "-",
                        })
                        bar.progress((i + 1) / len(urls))
                        summary_ph.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
                        continue

                    excel_brand   = str(combined_df.loc[i, brand_col]).strip()   if brand_col   else "-"
                    excel_product = str(combined_df.loc[i, product_col]).strip() if product_col else "-"

                    result = process_single_product(
                        links=unique_links, existing_fdas=existing_fdas,
                        output_filename=filename,
                        excel_brand=excel_brand,
                        excel_product=excel_product,
                    )

                    if not result:
                        summary_rows.append({
                            "แถว": i + 1, "สถานะ": "⏭️ ข้าม",
                            "เหตุผล": "ไม่พบ Pattern เลขอย.",
                            "FDA Number": "-", "Classification": "-",
                        })
                    elif result.get("is_duplicate"):
                        summary_rows.append({
                            "แถว": i + 1, "สถานะ": "🔁 ซ้ำ",
                            "เหตุผล": "FDA ซ้ำในระบบ",
                            "FDA Number": result.get("FDA_Number", "-"),
                            "Classification": "-",
                        })
                    else:
                        result["Source_Links"] = raw_val
                        save_checkpoint([result], filename)
                        summary_rows.append({
                            "แถว":          i + 1,
                            "สถานะ":        "✅ บันทึก",
                            "FDA Number":   result.get("FDA_Number", "-"),
                            "Brand":        result.get("Brand", "-"),
                            "Product_Name": result.get("Product_Name", "-"),
                            "TypeFDA":      result.get("TypeFDA", "-"),
                            "Claims_Dct":   result.get("Claims_Dct", "-"),
                            "Active_Claim": result.get("Active_Claim", "-"),
                            "Made_in":      result.get("Made_in", "-"),
                            "Net_Weight":   result.get("Net_Weight", "-"),
                        })

                    bar.progress((i + 1) / len(urls))
                    summary_ph.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                stat.empty()
                skipped = sum(1 for r in summary_rows if r["สถานะ"].startswith("⏭️"))
                dupes   = sum(1 for r in summary_rows if r["สถานะ"].startswith("🔁"))
                saved   = sum(1 for r in summary_rows if r["สถานะ"].startswith("✅"))

                st.markdown("---")
                st.markdown('<p class="sec">สรุปผลการประมวลผล</p>', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("✅ บันทึกแล้ว", saved)
                mc2.metric("🔁 ซ้ำ",        dupes)
                mc3.metric("⏭️ ข้าม",       skipped)