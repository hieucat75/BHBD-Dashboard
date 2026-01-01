import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import calendar

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Dashboard Quản Trị (Final)", layout="wide")
st.title("🚀 Dashboard Quản Trị Kinh Doanh")

if 'master_data' not in st.session_state:
    st.session_state.master_data = None
if 'waste_data' not in st.session_state:
    st.session_state.waste_data = None
if 'kpi_data' not in st.session_state:
    st.session_state.kpi_data = None

# --- 2. HÀM HỖ TRỢ ---
def to_float(x):
    try:
        if pd.isna(x): return 0.0
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(',', '').replace('\xa0', '')
        if not s: return 0.0
        if '(' in s and ')' in s: s = '-' + s.replace('(', '').replace(')', '')
        return float(s)
    except: return 0.0

def normalize_key(s):
    return str(s).strip().upper()

def normalize_branch(s):
    return str(s).strip().lower()

def fmt_lbl(val, is_pct=False):
    color = "red" if val < 0 else "black"
    if is_pct: return f"<b style='color:{color}'>{val:.1f}%</b>"
    return f"<b style='color:{color}'>{val:,.1f}</b>"

@st.cache_data
def load_file(file):
    if not file: return None
    try:
        if file.name.lower().endswith('.xlsx'): df = pd.read_excel(file)
        else: df = pd.read_csv(file, on_bad_lines='skip', sep=None, engine='python')
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except: return None

# --- 3. UPLOAD ---
with st.expander("📂 UPLOAD DỮ LIỆU", expanded=st.session_state.master_data is None):
    c1, c2, c3, c4 = st.columns(4)
    f_p = c1.file_uploader("1. BC Lợi Nhuận")
    f_w = c2.file_uploader("2. BC Xuất Hủy")
    f_k = c3.file_uploader("3. KPI")
    f_pr = c4.file_uploader("4. Bảng Giá")

if not (f_p and f_w and f_k and f_pr):
    st.info("Vui lòng upload đủ 4 file."); st.stop()

df_p_raw = load_file(f_p)
df_w_raw = load_file(f_w)
df_k_raw = load_file(f_k)
df_pr_raw = load_file(f_pr)

if len(df_p_raw) < len(df_pr_raw):
    st.error("⛔ Lỗi: File Lợi Nhuận nhỏ hơn Bảng Giá. Vui lòng kiểm tra lại vị trí upload.")
    st.stop()

# --- 4. CẤU HÌNH CỘT ---
if st.session_state.master_data is None:
    st.markdown("---")
    st.header("⚙️ Cấu hình cột")
    
    def idx(cols, keys):
        for i, c in enumerate(cols):
            if any(k in str(c).lower() for k in keys): return i
        return 0

    # Profit
    st.subheader("1. File Lợi Nhuận")
    cp = df_p_raw.columns.tolist()
    c1, c2 = st.columns(2)
    p_ma = c1.selectbox("Mã Hàng:", cp, index=idx(cp, ['mã hàng', 'mã sp']))
    p_ten = c1.selectbox("Tên Hàng:", cp, index=idx(cp, ['tên hàng']))
    p_cn = c2.selectbox("Chi Nhánh:", cp, index=idx(cp, ['chi nhánh']))
    p_ngay = c2.selectbox("Ngày:", cp, index=idx(cp, ['ngày', 'thời gian']))
    
    c3, c4, c5 = st.columns(3)
    p_sl = c3.selectbox("Số Lượng:", cp, index=idx(cp, ['số lượng', 'sl']))
    p_gb = c4.selectbox("Giá Bán:", cp, index=idx(cp, ['giá bán', 'doanh thu']))
    p_gv = c5.selectbox("Giá Vốn:", cp, index=idx(cp, ['giá vốn']))

    # Price
    st.subheader("2. Bảng Giá")
    cpr = df_pr_raw.columns.tolist()
    pr_ma = st.selectbox("Mã Hàng (Giá):", cpr, index=idx(cpr, ['mã hàng']))
    c_pr = st.columns(5)
    pr_chung = c_pr[0].selectbox("Chung:", cpr, index=idx(cpr, ['chung']))
    pr_mb = c_pr[1].selectbox("Miền Bắc:", cpr, index=idx(cpr, ['mienbac']))
    pr_mn = c_pr[2].selectbox("Miền Nam:", cpr, index=idx(cpr, ['miennam']))
    pr_tl = c_pr[3].selectbox("Thăng Long:", cpr, index=idx(cpr, ['thanglong']))
    pr_dq = c_pr[4].selectbox("Đặc Quyền:", cpr, index=idx(cpr, ['dacquyen']))

    # Waste
    st.subheader("3. Xuất Hủy")
    cw = df_w_raw.columns.tolist()
    c_w1, c_w2 = st.columns(2)
    w_ma = c_w1.selectbox("Mã Hủy:", cw, index=idx(cw, ['mã hàng']))
    w_val = c_w1.selectbox("Giá Trị:", cw, index=idx(cw, ['giá trị', 'thành tiền']))
    w_cn = c_w2.selectbox("CN Hủy:", cw, index=idx(cw, ['chi nhánh']))
    w_ngay = c_w2.selectbox("Ngày Hủy:", cw, index=idx(cw, ['ngày', 'thời gian']))

    # KPI
    st.subheader("4. KPI")
    ck = df_k_raw.columns.tolist()
    c_k1, c_k2, c_k3 = st.columns(3)
    k_cn = c_k1.selectbox("CN KPI:", ck, index=idx(ck, ['chi nhánh']))
    k_val = c_k2.selectbox("Target:", ck, index=idx(ck, ['chỉ tiêu']))
    k_kv = c_k3.selectbox("Khu Vực:", ck, index=idx(ck, ['khu vực', 'region']) or 4)

    if st.button("🚀 TÍNH TOÁN"):
        with st.spinner("Đang xử lý..."):
            try:
                # A. DICTIONARIES
                # Price Dicts
                df_pr = df_pr_raw.copy().drop_duplicates(pr_ma)
                df_pr['K'] = df_pr[pr_ma].apply(normalize_key)
                d_chung = df_pr.set_index('K')[pr_chung].apply(to_float).to_dict()
                d_mb = df_pr.set_index('K')[pr_mb].apply(to_float).to_dict()
                d_mn = df_pr.set_index('K')[pr_mn].apply(to_float).to_dict()
                d_tl = df_pr.set_index('K')[pr_tl].apply(to_float).to_dict()
                d_dq = df_pr.set_index('K')[pr_dq].apply(to_float).to_dict()

                # Region Dict
                df_k = df_k_raw.copy().drop_duplicates(k_cn)
                df_k['B'] = df_k[k_cn].apply(normalize_branch)
                d_reg = df_k.set_index('B')[k_kv].to_dict()

                # Name Dict
                df_n = df_p_raw[[p_ma, p_ten]].copy().drop_duplicates(p_ma)
                df_n['K'] = df_n[p_ma].apply(normalize_key)
                d_name = df_n.set_index('K')[p_ten].to_dict()

                # B. PROCESS SALES (MASTER)
                df_m = df_p_raw.copy().reset_index(drop=True)
                df_m['KEY'] = df_m[p_ma].apply(normalize_key)
                df_m['BRANCH'] = df_m[p_cn].apply(normalize_branch)
                df_m['NAME'] = df_m['KEY'].map(d_name).fillna(df_m['KEY'])
                df_m['REGION'] = df_m['BRANCH'].map(d_reg).fillna("Unknown")
                
                df_m['DATE'] = pd.to_datetime(df_m[p_ngay], dayfirst=True, errors='coerce')
                df_m['MONTH'] = df_m['DATE'].dt.to_period('M').astype(str)
                df_m['QTY'] = df_m[p_sl].apply(to_float)
                df_m['SELL'] = df_m[p_gb].apply(to_float)
                df_m['COST'] = df_m[p_gv].apply(to_float)
                
                df_m['REV'] = df_m['SELL'] * df_m['QTY']
                df_m['GP'] = (df_m['SELL'] - df_m['COST']) * df_m['QTY']

                # Discount Logic
                def calc_disc(row):
                    k, b, s, q = row['KEY'], row['BRANCH'], row['SELL'], row['QTY']
                    p = 0.0
                    if 'hà đông' in b or 'cầu diễn' in b:
                        p = d_dq.get(k, 0)
                        if p==0: p = d_mb.get(k, 0)
                    elif 'thăng long' in b: p = d_tl.get(k, 0)
                    elif 'hcm' in b: p = d_mn.get(k, 0)
                    elif any(x in b for x in ['hà nội', 'hải phòng', 'hp', 'hni']): p = d_mb.get(k, 0)
                    
                    if p==0: p = d_chung.get(k, 0)
                    
                    if p > s: return (p - s) * q
                    return 0.0

                df_m['DISC'] = df_m.apply(calc_disc, axis=1)

                # C. PROCESS WASTE
                df_w = df_w_raw.copy().reset_index(drop=True)
                df_w['KEY'] = df_w[w_ma].apply(normalize_key)
                df_w['BRANCH'] = df_w[w_cn].apply(normalize_branch)
                df_w['NAME'] = df_w['KEY'].map(d_name).fillna(df_w['KEY'])
                df_w['REGION'] = df_w['BRANCH'].map(d_reg).fillna("Unknown")
                df_w['VAL'] = df_w[w_val].apply(to_float)
                df_w['DATE'] = pd.to_datetime(df_w[w_ngay], dayfirst=True, errors='coerce')
                df_w['MONTH'] = df_w['DATE'].dt.to_period('M').astype(str)

                # Save Session
                st.session_state.master_data = df_m
                st.session_state.waste_data = df_w
                
                # Save KPI raw for target calc
                df_k_final = df_k_raw.copy()
                df_k_final['B'] = df_k_final[k_cn].apply(normalize_branch)
                df_k_final['T'] = df_k_final[k_val].apply(to_float)
                st.session_state.kpi_data = df_k_final

                st.rerun()

            except Exception as e:
                st.error(f"Lỗi xử lý: {e}"); st.stop()

# --- 6. VIEW ---
if st.session_state.master_data is not None:
    df_m = st.session_state.master_data
    df_w = st.session_state.waste_data
    df_k = st.session_state.kpi_data

    # Filter
    st.markdown("### 🔍 Bộ Lọc")
    c1, c2, c3 = st.columns(3)
    months = sorted(list(set(df_m['MONTH'].unique()) | set(df_w['MONTH'].unique())))
    sel_m = c1.multiselect("Tháng:", months, default=months)
    
    regions = sorted(df_m['REGION'].unique())
    sel_r = c2.multiselect("Khu Vực:", regions, default=regions)
    
    valid_b = df_m[df_m['REGION'].isin(sel_r)]['BRANCH'].unique()
    sel_b = c3.multiselect("Chi Nhánh:", valid_b, default=valid_b)

    if not sel_m or not sel_b: st.warning("Chọn bộ lọc"); st.stop()

    # Apply Filter
    dm = df_m[df_m['MONTH'].isin(sel_m) & df_m['BRANCH'].isin(sel_b)]
    dw = df_w[df_w['MONTH'].isin(sel_m) & df_w['BRANCH'].isin(sel_b)]

    # Metrics
    rev = dm['REV'].sum()
    gp = dm['GP'].sum()
    disc = dm['DISC'].sum()
    waste = dw['VAL'].sum()
    net = gp - waste
    
    # Target
    dk = df_k[df_k['B'].isin(sel_b)]
    daily = dk['T'].sum()
    days = 0
    for m in sel_m:
        try: y, mm = map(int, m.split('-')); days += calendar.monthrange(y, mm)[1]
        except: continue
    target = daily * days

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Doanh Thu", f"{rev/1e6:,.1f} Tr", f"{(rev/target*100 if target else 0):.1f}% KPI")
    k2.metric("LN Gộp", f"{gp/1e6:,.1f} Tr", f"{(gp/rev*100 if rev else 0):.1f}% DT")
    k3.metric("Hủy", f"{waste/1e6:,.1f} Tr", f"{(waste/rev*100 if rev else 0):.1f}% DT", delta_color="inverse")
    k4.metric("LN Ròng", f"{net/1e6:,.1f} Tr", f"{(net/rev*100 if rev else 0):.1f}% Margin")
    k5.metric("Giảm Giá", f"{disc/1e6:,.1f} Tr", f"{(disc/rev*100 if rev else 0):.1f}% DT", delta_color="inverse")

    st.markdown("---")

    # Chart Helpers
    def style_fig(fig, zero=False):
        fig.update_layout(yaxis={'showgrid':False, 'showticklabels':True}, xaxis={'showgrid':False}, margin=dict(t=30, b=30, l=150, r=0), height=400, showlegend=False)
        if zero: fig.add_hline(y=0, line_dash="dash", line_color="red")
        return fig

    r1, r2 = st.columns(2)

    # 1. Monthly
    with r1:
        st.subheader("📊 Monthly Trend")
        gr = dm.groupby('MONTH')['REV'].sum()/1e6
        gn = (dm.groupby('MONTH')['GP'].sum() - dw.groupby('MONTH')['VAL'].sum())/1e6
        dfc = pd.DataFrame({'R': gr}).fillna(0)
        dfc['N'] = gn.reindex(dfc.index).fillna(0)
        dfc['M'] = (dfc['N']/dfc['R']*100).fillna(0)
        
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig1.add_trace(go.Bar(x=dfc.index, y=dfc['R'], text=[fmt_lbl(x) for x in dfc['R']], textposition='outside', marker_color='#F5B041'), secondary_y=False)
        fig1.add_trace(go.Scatter(x=dfc.index, y=dfc['M'], text=[fmt_lbl(x, True) for x in dfc['M']], mode='lines+markers+text', textposition='top center', line=dict(color='#27AE60')), secondary_y=True)
        fig1.update_yaxes(showticklabels=False, secondary_y=False)
        fig1.update_yaxes(showticklabels=False, secondary_y=True)
        st.plotly_chart(style_fig(fig1, True), use_container_width=True)

    # 2. Branch
    with r2:
        st.subheader("📊 Branch Performance")
        br = dm.groupby('BRANCH')['REV'].sum()/1e6
        bn = (dm.groupby('BRANCH')['GP'].sum() - dw.groupby('BRANCH')['VAL'].sum())/1e6
        dfb = pd.DataFrame({'R': br}).fillna(0)
        dfb['N'] = bn.reindex(dfb.index).fillna(0)
        dfb['M'] = (dfb['N']/dfb['R']*100).fillna(0)
        dfb = dfb.sort_values('R', ascending=False)
        
        # Map Name back (Optional, here using norm name)
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Bar(x=dfb.index, y=dfb['R'], text=[fmt_lbl(x) for x in dfb['R']], textposition='outside', marker_color='#F5B041'), secondary_y=False)
        fig2.add_trace(go.Scatter(x=dfb.index, y=dfb['M'], text=[fmt_lbl(x, True) for x in dfb['M']], mode='lines+markers+text', textposition='top center', line=dict(color='#2E86C1')), secondary_y=True)
        fig2.update_yaxes(showticklabels=True, secondary_y=False)
        fig2.update_yaxes(showticklabels=False, secondary_y=True)
        st.plotly_chart(style_fig(fig2, True), use_container_width=True)

    # Top 10
    top_s = dm.groupby(['NAME']).agg({'REV':'sum', 'GP':'sum', 'DISC':'sum'}).reset_index()
    top_w = dw.groupby('NAME')['VAL'].sum().to_dict()
    top_s['WASTE'] = top_s['NAME'].map(top_w).fillna(0)
    top_s['NET'] = top_s['GP'] - top_s['WASTE']

    c3, c4, c5 = st.columns(3)
    
    with c3:
        st.subheader("🏆 Top 10 Doanh Thu")
        t = top_s.nlargest(10, 'REV')
        lbls = [f"<b style='color:black'>{r['REV']/1e6:,.1f}</b> <span style='color:{'red' if r['NET']<0 else 'green'}'>({r['NET']/r['REV']*100:.1f}%)</span>" if r['REV'] else "0" for _, r in t.iterrows()]
        fig3 = px.bar(t, y='NAME', x='REV', orientation='h', text=lbls)
        fig3.update_traces(marker_color='#2980B9', textposition='outside')
        fig3.update_layout(yaxis={'categoryorder':'total ascending', 'title':None})
        st.plotly_chart(style_fig(fig3), use_container_width=True)

    with c4:
        st.subheader("⚠️ Top 10 Hủy")
        tw = dw.groupby('NAME')['VAL'].sum().nlargest(10).reset_index()
        rev_map = dm.groupby('NAME')['REV'].sum().to_dict()
        tw['REV'] = tw['NAME'].map(rev_map).fillna(0)
        lbls = [f"{fmt_lbl(r['VAL']/1e6)} ({r['VAL']/r['REV']*100:.1f}%)" if r['REV'] else fmt_lbl(r['VAL']/1e6) for _, r in tw.iterrows()]
        fig4 = px.bar(tw, y='NAME', x='VAL', orientation='h', text=lbls)
        fig4.update_traces(marker_color='#E74C3C', textposition='outside')
        fig4.update_layout(yaxis={'categoryorder':'total ascending', 'title':None})
        st.plotly_chart(style_fig(fig4), use_container_width=True)

    with c5:
        st.subheader("📉 Top 10 Giảm Giá")
        td = top_s.nlargest(10, 'DISC')
        lbls = [f"<b>{r['DISC']/1e6:,.1f}</b> ({r['DISC']/r['REV']*100:.1f}%)" if r['REV'] else "0" for _, r in td.iterrows()]
        fig5 = px.bar(td, y='NAME', x='DISC', orientation='h', text=lbls)
        fig5.update_traces(marker_color='#8E44AD', textposition='outside')
        fig5.update_layout(yaxis={'categoryorder':'total ascending', 'title':None})
        st.plotly_chart(style_fig(fig5), use_container_width=True)