import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import calendar
import warnings

warnings.filterwarnings('ignore')

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Dashboard BHBĐ (v7.0)", layout="wide", initial_sidebar_state="expanded")

# --- MÀU SẮC THƯƠNG HIỆU ---
COLOR_REV = '#EFB000'      # Vàng Honey (VNPOST)
COLOR_TOP_SALES = '#001f3f' # Xanh Navy
COLOR_WASTE = '#fd7e14'    # Cam
COLOR_POS = '#28a745'      # Xanh lá
COLOR_NEG = '#dc3545'      # Đỏ
COLOR_DEAD = '#6c757d'     # Xám

# CSS TÙY CHỈNH
st.markdown(f"""
<style>
    /* Sidebar Navy */
    [data-testid="stSidebar"] {{ background-color: #001f3f; color: white; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {{ color: white !important; }}
    
    /* Metrics */
    div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 700; color: #001f3f; }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{ height: 45px; background-color: #f8f9fa; border-radius: 5px; color: #495057; font-weight: 600; }}
    .stTabs [aria-selected="true"] {{ background-color: #001f3f; color: white; }}
</style>
""", unsafe_allow_html=True)

st.title("🚀 DASHBOARD BHBĐ - EXECUTIVE COCKPIT (V7.0)")

# --- 2. HÀM XỬ LÝ ---
@st.cache_data(show_spinner=False)
def load_data(file):
    if not file: return None
    try:
        if file.name.lower().endswith('.xlsx'): df = pd.read_excel(file, dtype=str)
        else: df = pd.read_csv(file, dtype=str, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Lỗi đọc file {file.name}: {e}"); return None

def safe_float(series):
    return pd.to_numeric(series.str.replace(',', '').str.replace(r'[()]', '', regex=True), errors='coerce').fillna(0)

def safe_date(series):
    return pd.to_datetime(series, dayfirst=True, errors='coerce')

# --- 3. UPLOAD FILES ---
with st.sidebar.expander("📂 UPLOAD DỮ LIỆU (6 FILES)", expanded=True):
    f_prod = st.file_uploader("1. DS Sản Phẩm (Master)", type=['xlsx', 'csv'])
    f_price = st.file_uploader("2. Bảng Giá", type=['xlsx', 'csv'])
    f_kpi = st.file_uploader("3. KPI", type=['xlsx', 'csv'])
    f_sales = st.file_uploader("4. BC Bán Hàng (Lợi Nhuận)", type=['xlsx', 'csv'])
    f_inv = st.file_uploader("5. BC Xuất Nhập Tồn", type=['xlsx', 'csv'])
    f_waste = st.file_uploader("6. BC Xuất Hủy", type=['xlsx', 'csv'])

if st.sidebar.button("🔄 Reset Dữ Liệu"):
    st.cache_data.clear()
    st.rerun()

# --- 4. MAPPING & PROCESSING ---
if f_prod and f_price and f_kpi and f_sales and f_inv and f_waste:
    df_sales_raw = load_data(f_sales)
    df_inv_raw = load_data(f_inv)
    df_waste_raw = load_data(f_waste)
    df_prod = load_data(f_prod)
    df_kpi = load_data(f_kpi)

    st.markdown("---")
    st.subheader("⚙️ CẤU HÌNH DỮ LIỆU")

    col1, col2, col3 = st.columns(3)
    
    def get_idx(cols, keys):
        for i, c in enumerate(cols):
            if any(k in c.lower() for k in keys): return i
        return 0

    with col1:
        st.info("Mapping Bán Hàng")
        cols_s = df_sales_raw.columns.tolist()
        s_ma = st.selectbox("Mã Hàng", cols_s, index=get_idx(cols_s, ['mã hàng', 'mã sp']))
        s_cn = st.selectbox("Chi Nhánh", cols_s, index=get_idx(cols_s, ['chi nhánh']))
        s_time = st.selectbox("Ngày GD", cols_s, index=get_idx(cols_s, ['thời gian (theo giao dịch)', 'ngày']))
        s_sl = st.selectbox("Số Lượng Bán", cols_s, index=get_idx(cols_s, ['sl', 'số lượng']))
        s_dg_ban = st.selectbox("Đơn Giá Bán", cols_s, index=get_idx(cols_s, ['giá bán/sp', 'đơn giá bán']))
        s_dg_von = st.selectbox("Đơn Giá Vốn", cols_s, index=get_idx(cols_s, ['giá vốn/sp', 'đơn giá vốn']))

    with col2:
        st.info("Mapping Kho (Quan Trọng)")
        cols_i = df_inv_raw.columns.tolist()
        i_ma = st.selectbox("Mã Hàng (Kho)", cols_i, index=get_idx(cols_i, ['mã hàng']))
        i_cn = st.selectbox("Chi Nhánh (Kho)", cols_i, index=get_idx(cols_i, ['chi nhánh']))
        i_ton_sl = st.selectbox("Tồn Cuối (SL)", cols_i, index=get_idx(cols_i, ['tồn cuối', 'sl tồn']))
        i_ton_val = st.selectbox("Giá Trị Tồn", cols_i, index=get_idx(cols_i, ['giá trị cuối', 'thành tiền tồn']))
        
        # Chọn cột nhập để so sánh
        import_candidates = [c for c in cols_i if 'nhập' in c.lower() and 'giá trị' not in c.lower()]
        i_nhap_cols = st.multiselect("Cột Nhập Hàng (Cộng dồn)", cols_i, default=import_candidates)

    with col3:
        st.info("Mapping Hủy & Khác")
        cols_w = df_waste_raw.columns.tolist()
        w_ma = st.selectbox("Mã Hàng (Hủy)", cols_w, index=get_idx(cols_w, ['mã hàng']))
        w_cn = st.selectbox("Chi Nhánh (Hủy)", cols_w, index=get_idx(cols_w, ['chi nhánh']))
        w_time = st.selectbox("Ngày Hủy", cols_w, index=get_idx(cols_w, ['ngày', 'thời gian']))
        w_val = st.selectbox("Giá Trị Hủy", cols_w, index=get_idx(cols_w, ['giá trị hủy', 'thành tiền']))

        cat_ma = st.selectbox("DS SP: Mã", df_prod.columns, index=get_idx(df_prod.columns, ['mã hàng']))
        cat_nhom = st.selectbox("DS SP: Ngành", df_prod.columns, index=get_idx(df_prod.columns, ['nhóm hàng', 'ngành']))
        
        kpi_cn = st.selectbox("KPI: Chi Nhánh", df_kpi.columns, index=get_idx(df_kpi.columns, ['chi nhánh']))
        kpi_val = st.selectbox("KPI: Target Ngày", df_kpi.columns, index=get_idx(df_kpi.columns, ['chỉ tiêu', 'target']))
        kpi_kv = st.selectbox("KPI: Khu vực", df_kpi.columns, index=get_idx(df_kpi.columns, ['khu vực', 'region']))

    if st.button("🚀 XÁC NHẬN & CHẠY"):
        try:
            with st.spinner("Đang tính toán..."):
                # 1. Master Data
                df_prod[cat_ma] = df_prod[cat_ma].astype(str).str.strip().str.upper()
                d_cat = dict(zip(df_prod[cat_ma], df_prod[cat_nhom].astype(str).str.split('>').str[0].str.strip()))
                d_name = dict(zip(df_prod[cat_ma], df_prod[df_prod.columns[get_idx(df_prod.columns, ['tên'])]]))

                df_kpi[kpi_cn] = df_kpi[kpi_cn].astype(str).str.strip()
                d_reg = dict(zip(df_kpi[kpi_cn], df_kpi[kpi_kv]))
                d_target = dict(zip(df_kpi[kpi_cn], safe_float(df_kpi[kpi_val])))

                # 2. Sales Processing
                df_m = pd.DataFrame()
                df_m['PROD_ID'] = df_sales_raw[s_ma].astype(str).str.strip().str.upper()
                df_m['BRANCH_ID'] = df_sales_raw[s_cn].astype(str).str.strip()
                df_m['DATE'] = safe_date(df_sales_raw[s_time])
                df_m = df_m.dropna(subset=['DATE'])
                df_m['MONTH'] = df_m['DATE'].dt.strftime('%Y-%m')

                df_m['QTY'] = safe_float(df_sales_raw[s_sl])
                df_m['REV'] = df_m['QTY'] * safe_float(df_sales_raw[s_dg_ban])
                df_m['COST'] = df_m['QTY'] * safe_float(df_sales_raw[s_dg_von])
                df_m['GP'] = df_m['REV'] - df_m['COST']

                df_m['REGION'] = df_m['BRANCH_ID'].map(d_reg).fillna('Unknown')
                df_m['CATEGORY'] = df_m['PROD_ID'].map(d_cat).fillna('Khác')
                df_m['NAME'] = df_m['PROD_ID'].map(d_name).fillna(df_m['PROD_ID'])

                # 3. Waste Processing
                df_w = pd.DataFrame()
                df_w['PROD_ID'] = df_waste_raw[w_ma].astype(str).str.strip().str.upper()
                df_w['BRANCH_ID'] = df_waste_raw[w_cn].astype(str).str.strip()
                df_w['DATE'] = safe_date(df_waste_raw[w_time])
                df_w = df_w.dropna(subset=['DATE'])
                df_w['MONTH'] = df_w['DATE'].dt.strftime('%Y-%m')
                df_w['VAL'] = safe_float(df_waste_raw[w_val])
                df_w['REGION'] = df_w['BRANCH_ID'].map(d_reg).fillna('Unknown')
                df_w['CATEGORY'] = df_w['PROD_ID'].map(d_cat).fillna('Khác')

                # 4. Inventory Processing
                df_i = pd.DataFrame()
                df_i['PROD_ID'] = df_inv_raw[i_ma].astype(str).str.strip().str.upper()
                df_i['BRANCH_ID'] = df_inv_raw[i_cn].astype(str).str.strip()
                df_i['STOCK_QTY'] = safe_float(df_inv_raw[i_ton_sl])
                df_i['STOCK_VAL'] = safe_float(df_inv_raw[i_ton_val])
                
                # Import calculation
                df_i['IMPORT_QTY'] = 0
                for c in i_nhap_cols:
                    df_i['IMPORT_QTY'] += safe_float(df_inv_raw[c])

                df_i['REGION'] = df_i['BRANCH_ID'].map(d_reg).fillna('Unknown')
                df_i['CATEGORY'] = df_i['PROD_ID'].map(d_cat).fillna('Khác')
                df_i['NAME'] = df_i['PROD_ID'].map(d_name).fillna(df_i['PROD_ID'])

                st.session_state.data = {
                    'sales': df_m, 'waste': df_w, 'inv': df_i, 
                    'targets': d_target, 'processed': True
                }
                st.rerun()

        except Exception as e:
            st.error(f"Lỗi: {e}"); st.stop()

# --- 5. DASHBOARD VIEW ---
if 'data' in st.session_state and st.session_state.data.get('processed'):
    data = st.session_state.data
    df_m = data['sales']
    df_w = data['waste']
    df_i = data['inv']
    d_target = data['targets']

    # FILTER
    st.sidebar.markdown("### 🔍 BỘ LỌC")
    months = sorted(list(set(df_m['MONTH'].unique()) | set(df_w['MONTH'].unique())))
    sel_months = st.sidebar.multiselect("1. Tháng", months, default=months)
    regions = sorted(df_m['REGION'].unique())
    sel_regions = st.sidebar.multiselect("2. Khu Vực", regions, default=regions)
    cats = sorted(df_m['CATEGORY'].unique())
    sel_cats = st.sidebar.multiselect("3. Ngành Hàng", cats, default=cats)
    valid_b = df_m[df_m['REGION'].isin(sel_regions)]['BRANCH_ID'].unique()
    sel_branches = st.sidebar.multiselect("4. Chi Nhánh", sorted(valid_b), default=sorted(valid_b))

    if not (sel_months and sel_regions and sel_cats and sel_branches):
        st.warning("Vui lòng chọn bộ lọc."); st.stop()

    # FILTERED DATA
    dm = df_m[df_m['MONTH'].isin(sel_months) & df_m['BRANCH_ID'].isin(sel_branches) & df_m['CATEGORY'].isin(sel_cats)]
    dw = df_w[df_w['MONTH'].isin(sel_months) & df_w['BRANCH_ID'].isin(sel_branches) & df_w['CATEGORY'].isin(sel_cats)]
    di = df_i[df_i['BRANCH_ID'].isin(sel_branches) & df_i['CATEGORY'].isin(sel_cats)]

    # METRICS
    total_rev = dm['REV'].sum()
    total_gp = dm['GP'].sum()
    total_waste = dw['VAL'].sum()
    total_net = total_gp - total_waste
    total_stock = di['STOCK_VAL'].sum()

    # KPI
    days_count = 0
    for m in sel_months:
        y, mm = map(int, m.split('-'))
        days_count += calendar.monthrange(y, mm)[1]
    
    total_daily_target = sum([d_target.get(b, 0) for b in sel_branches])
    total_target = total_daily_target * days_count
    kpi_pct = (total_rev / total_target * 100) if total_target > 0 else 0
    net_margin = (total_net / total_rev * 100) if total_rev > 0 else 0

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 EXECUTIVE VIEW", "⚡ ACTION CENTER", "🔎 DATA EXPLORER"])

    # --- TAB 1 ---
    with tab1:
        st.markdown("#### 🌟 SỨC KHỎE DOANH NGHIỆP")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Doanh Thu", f"{total_rev:,.0f}", f"{kpi_pct:.1f}% KPI")
        c2.metric("Lợi Nhuận Gộp", f"{total_gp:,.0f}", f"{(total_gp/total_rev*100 if total_rev else 0):.1f}% Rev")
        c3.metric("Xuất Hủy", f"{total_waste:,.0f}", f"-{(total_waste/total_rev*100 if total_rev else 0):.1f}% Rev", delta_color="inverse")
        c4.metric("LN Ròng (Net)", f"{total_net:,.0f}", f"{net_margin:.1f}% Margin")
        c5.metric("Tồn Kho (Vốn)", f"{total_stock:,.0f}", "Hiện tại")
        
        st.markdown("---")
        
        # CHART 1: MONTHLY COMBO
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("##### 📉 Xu hướng Doanh thu & LN Ròng")
            m_rev = dm.groupby('MONTH')['REV'].sum()
            m_gp = dm.groupby('MONTH')['GP'].sum()
            m_waste = dw.groupby('MONTH')['VAL'].sum()
            df_chart = pd.DataFrame({'REV': m_rev}).fillna(0)
            df_chart['NET'] = m_gp.sub(m_waste, fill_value=0).fillna(0)
            df_chart['PCT'] = (df_chart['NET'] / df_chart['REV'] * 100).fillna(0)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['REV'], name='Doanh Thu', marker_color=COLOR_REV), secondary_y=False)
            
            # Logic màu Lợi nhuận (Xanh/Đỏ)
            line_color = [COLOR_POS if x>=0 else COLOR_NEG for x in df_chart['PCT']]
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['PCT'], name='% LN Ròng', mode='lines+markers', 
                                     line=dict(color='gray', width=1), marker=dict(size=8, color=line_color)), secondary_y=True)
            
            fig.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("##### 🏆 Top 10 Chi Nhánh (Hiệu quả)")
            b_rev = dm.groupby('BRANCH_ID')['REV'].sum()
            b_gp = dm.groupby('BRANCH_ID')['GP'].sum()
            b_waste = dw.groupby('BRANCH_ID')['VAL'].sum()
            df_b = pd.DataFrame({'REV': b_rev}).fillna(0)
            df_b['NET'] = b_gp.sub(b_waste, fill_value=0).fillna(0)
            df_b = df_b.sort_values('REV', ascending=False).head(10)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(y=df_b.index, x=df_b['REV'], name='Doanh Thu', orientation='h', marker_color=COLOR_REV))
            fig2.add_trace(go.Bar(y=df_b.index, x=df_b['NET'], name='LN Ròng', orientation='h', 
                                  marker_color=df_b['NET'].apply(lambda x: COLOR_POS if x>=0 else COLOR_NEG)))
            fig2.update_layout(barmode='group', height=350)
            st.plotly_chart(fig2, use_container_width=True)

        # ROW 3: DRIVERS
        c_dr1, c_dr2 = st.columns(2)
        with c_dr1:
            st.markdown("##### 💎 Top 10 SP Doanh Thu Cao")
            top_prod = dm.groupby('NAME')['REV'].sum().nlargest(10).sort_values()
            fig_p = go.Figure(go.Bar(x=top_prod.values, y=top_prod.index, orientation='h', marker_color=COLOR_TOP_SALES))
            st.plotly_chart(fig_p, use_container_width=True)
        
        with c_dr2:
            st.markdown("##### 💀 Top 10 SKU No Sales (Tồn Cao)")
            # Logic: Sales=0, Tồn > 0, Sort by Tồn Value
            agg_s = dm.groupby('PROD_ID')['REV'].sum().reset_index()
            agg_stk = di.groupby('PROD_ID')['STOCK_VAL'].sum().reset_index()
            df_dead = pd.merge(agg_stk, agg_s, on='PROD_ID', how='left').fillna(0)
            
            # Map name
            n_map = di[['PROD_ID', 'NAME']].drop_duplicates('PROD_ID').set_index('PROD_ID')['NAME'].to_dict()
            df_dead['NAME'] = df_dead['PROD_ID'].map(n_map).fillna(df_dead['PROD_ID'])
            
            # Filter
            no_sales = df_dead[df_dead['REV'] == 0].sort_values('STOCK_VAL', ascending=False).head(10).sort_values('STOCK_VAL')
            
            fig_d = go.Figure(go.Bar(x=no_sales['STOCK_VAL'], y=no_sales['NAME'], orientation='h', marker_color=COLOR_DEAD))
            st.plotly_chart(fig_d, use_container_width=True)

    # --- TAB 2 ---
    with tab2:
        st.markdown("#### ⚡ TRUNG TÂM HÀNH ĐỘNG")
        
        # Prep
        agg_qty = dm.groupby(['PROD_ID', 'NAME'])['QTY'].sum().reset_index().rename(columns={'QTY':'SALES_QTY'})
        agg_stock = di.groupby('PROD_ID').agg({'STOCK_QTY':'sum', 'STOCK_VAL':'sum', 'IMPORT_QTY':'sum'}).reset_index()
        df_act = pd.merge(agg_qty, agg_stock, on='PROD_ID', how='outer').fillna(0)
        
        # Name fallback
        all_names = pd.concat([dm[['PROD_ID','NAME']], di[['PROD_ID','NAME']]]).drop_duplicates('PROD_ID').set_index('PROD_ID')['NAME'].to_dict()
        df_act['NAME'] = df_act.apply(lambda x: x['NAME'] if x['NAME']!=0 else all_names.get(x['PROD_ID'], x['PROD_ID']), axis=1)

        # Logic OOS
        avg_daily = df_act['SALES_QTY'] / (days_count if days_count > 0 else 1)
        df_act['DAYS_SUPPLY'] = np.where(avg_daily > 0, df_act['STOCK_QTY'] / avg_daily, 999)
        
        # OOS Condition: Sales > 0 AND Days < 3
        oos = df_act[(df_act['SALES_QTY'] > 0) & (df_act['DAYS_SUPPLY'] < 3)].sort_values('SALES_QTY', ascending=False).head(50)
        
        # Dead Stock Condition
        dead = df_act[(df_act['SALES_QTY'] == 0) & (df_act['STOCK_VAL'] > 1000000)].sort_values('STOCK_VAL', ascending=False).head(50)

        c_a1, c_a2 = st.columns(2)
        with c_a1:
            st.error(f"🚨 BÁO ĐỘNG ĐỨT HÀNG ({len(oos)} SP)")
            st.caption("Tiêu chí: Đang bán tốt nhưng tồn < 3 ngày bán. Cột 'Nhập Trong Kỳ' giúp đánh giá lượng hàng về.")
            st.dataframe(oos[['NAME', 'SALES_QTY', 'STOCK_QTY', 'IMPORT_QTY', 'DAYS_SUPPLY']]
                         .rename(columns={'SALES_QTY':'Đã Bán', 'STOCK_QTY':'Tồn', 'IMPORT_QTY':'Nhập Trong Kỳ', 'DAYS_SUPPLY':'Số Ngày Còn'})
                         .style.format("{:.1f}", subset=['Số Ngày Còn'])
                         .background_gradient(subset=['Đã Bán'], cmap='Oranges'), use_container_width=True)
        
        with c_a2:
            st.warning(f"🐢 BÁO ĐỘNG TỒN CHẾT ({len(dead)} SP)")
            st.caption("Tiêu chí: Không bán được (Sales=0) & Giá trị tồn > 1 Triệu")
            st.dataframe(dead[['NAME', 'STOCK_QTY', 'STOCK_VAL']]
                         .rename(columns={'STOCK_QTY':'SL Tồn', 'STOCK_VAL':'Giá Trị Tồn'})
                         .style.format("{:,.0f}", subset=['Giá Trị Tồn'])
                         .background_gradient(subset=['Giá Trị Tồn'], cmap='Greys'), use_container_width=True)

    # --- TAB 3 ---
    with tab3:
        st.dataframe(df_act, use_container_width=True)