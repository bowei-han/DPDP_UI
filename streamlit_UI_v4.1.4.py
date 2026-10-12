#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import os
import json

st.set_page_config(page_title="DPDP S-XAI Dispatcher", layout="wide")

# 1. Initialize Session States
if 'active_node' not in st.session_state:
    st.session_state.active_node = None
if 'active_v_cluster' not in st.session_state:
    st.session_state.active_v_cluster = None 
if 'active_o_cluster' not in st.session_state:
    st.session_state.active_o_cluster = None
if 'click_buffer' not in st.session_state:
    st.session_state.click_buffer = None
if 'trigger_xai_dialog' not in st.session_state:
    st.session_state.trigger_xai_dialog = False
    
# Static definitions on the frontend (No need to pass these via JSON anymore!)
FEATURE_NAMES = [
    "Fuel Rate (L/km)", "Urgency Rem. (min)", "Est. Finish (min)", "Loading Ratio (%)", "Fatigue (min)", "Shift Rem. (min)",
    "Order Weight (kg)", "Time Budget (min)", "Order Est. Finish",
    "Pickup Dist (km)", "Pickup Time (min)", 
    "Delivery Dist (km)", "Delivery Time (min)"
]
PREDICATE_TRANSLATIONS = {
    "causally_influences": "directly impacts",
    "is_restricted_by": "is limited by the",
    "triggers_explanation_reason": "triggers the rule:",
    "has_correlation_with": "is correlated with",
    "describes": "is a property of",
    "is_measured_by": "is measured by",
    "is_causally_influenced_by": "is heavily influenced by",
    "is_computed_from": "is computed directly from"
}

ENTITY_TRANSLATIONS = {
    # Core Physical & Dispatch Entities
    "CarryingOrderEstimateFinishTime": "Estimated Finish Time",
    "AccumulatedFatigue": "Driver Fatigue",
    "MandatoryBreakProcess": "Mandatory Rest Break",
    "CarryingOrderUrgency": "Order Urgency",
    "WorkshiftRemainingTime": "Remaining Shift Time",
    "PickUpTimeDuration": "Time to Reach Pickup",
    "PickUpDistance": "Distance to Pickup",
    "AccumulatedFuelConsumption": "Total Fuel Consumption",
    "PickingUpProcess": "the Pickup Process",
    "FuelEfficiency": "Vehicle Fuel Efficiency",
    "OrderUrgency": "Global Delivery Urgency",
    "LIFO_Constraint": "LIFO Loading Constraint",
    "CargoWeight": "Total Cargo Weight",
    "LoadingTimeDuration": "Required Loading Time",
    "UnloadingTimeDuration": "Required Unloading Time",
    "ProductAdaptivity": "Vehicle Equipment Adaptivity",
    "DeliveryDistance": "Distance to Delivery",
    "DeliveryingProcess": "the Delivery Process",
    "OrderTimeBudget": "Total Time Budget",
    "OrderEstimateFinishTime": "Global Estimated Finish",
    "LoadingProcess": "the Loading Process",
    "OrderCommittedFinishTime": "Committed Delivery Deadline",
    "LoadingRatio": "Vehicle Loading Ratio",
    "LoadingWeight": "Total Payload Weight",
    "WeightCapacity": "Max Vehicle Capacity",
    
    # Outcomes & Rules (Dropping the ugly prefixes)
    "explanationreason_FatigueMandatoryBreakReason": "Fatigue Limit Exceeded",
    "explanationreason_WorkshiftApproachingReason": "Approaching End of Shift",
    "rejectionreason_WorkshiftOvertimeReason": "Strict Overtime Violation",
    "explanationreason_FuelExpenseReason": "Excessive Fuel Cost",
    "explanationreason_OvertimeDeliveryReason": "Delivery Deadline Risk",
    "explanationreason_UrgencyReason": "High Priority Order",
    "explanationreason_ProductAdaptivityOptimisationReason": "Specialized Equipment Required",
    "rejectionreason_OverweightReason": "Strict Weight Capacity Violation"
}

# --- S-XAI LOGS (Dynamic Decoupled Loader) ---
@st.cache_data(ttl=5) # Checks for file updates every 5 seconds
def load_xai_logs():
    file_path = "xai_logs.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} # Return empty if JSON is currently being overwritten
    return {}

@st.cache_data(ttl=5) 
def load_swrl_logs():
    file_path = "swrl_audit.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def render_explanation_expanders(veh_data):
    """Reusable renderer that translates raw triples into human-readable sentences using Graph DFS"""
    shap_arr = np.array(veh_data["shap"])
    phys_arr = np.array(veh_data["physical"])
    causal_dict = veh_data["causal"]
    
    # Sort by absolute impact to get top 5
    top_indices = np.argsort(np.abs(shap_arr))[-5:][::-1]
    
    for idx in top_indices:
        impact = shap_arr[idx]
        val = phys_arr[idx]
        name = FEATURE_NAMES[idx]
        
        # Format the Header
        color_icon = "🟩 [Reward]" if impact > 0 else "🟥 [Penalty]"
        header_text = f"{color_icon} {name} : {val:.2f} (Impact: {impact:+.2f})"
        
        # Render the Expander
        with st.expander(header_text):
            str_idx = str(idx) 
            if str_idx in causal_dict:
                triples = causal_dict[str_idx]
                
                # 1. Build the Adjacency List (Reconstruct the Graph)
                adj = {}
                for t in triples:
                    parts = t.split(",")
                    if len(parts) == 3:
                        s, p, o = [x.strip() for x in parts]
                        if s not in adj: adj[s] = []
                        if (p, o) not in adj[s]: # Prevent duplicates
                            adj[s].append((p, o))

                if not adj:
                    continue
                    
                # 2. Identify the Root Node (The first subject in the logged triples)
                root_node = triples[0].split(",")[0].strip()
                
                # 3. Depth-First Search (DFS) to extract independent causal chains
                all_paths = []
                def dfs(current_node, current_path, visited):
                    # End the path if we hit a dead end, or if we hit a loop (visited)
                    if current_node not in adj or not adj[current_node] or current_node in visited:
                        if current_path:
                            all_paths.append(current_path)
                        return
                    
                    # Continue down all branches
                    for pred, obj in adj[current_node]:
                        # pass visited.union() to avoid modifying the set for other branches
                        dfs(obj, current_path + [(current_node, pred, obj)], visited.union({current_node}))

                dfs(root_node, [], set())
                
                # 4. Render the Independent Paths
                chain_html = '<div style="font-size: 13.5px; line-height: 1.6;">'
                
                for path in all_paths:
                    for i, (raw_subj, raw_pred, raw_obj) in enumerate(path):
                        # Dictionary Translation
                        clean_subj = ENTITY_TRANSLATIONS.get(raw_subj, raw_subj.replace("_", " "))
                        clean_pred = PREDICATE_TRANSLATIONS.get(raw_pred, raw_pred.replace("_", " "))
                        clean_obj = ENTITY_TRANSLATIONS.get(raw_obj, raw_obj.replace("_", " ").replace("explanationreason_", "").replace("rejectionreason_", ""))
                        
                        # Tiered Indentation (Cap at 72px so it doesn't run off screen)
                        indent_px = min(i * 18, 72) 
                        
                        # Styling
                        obj_color = "#E11D48" if any(word in clean_obj for word in ["Violation", "Exceeded", "Cost", "Risk"]) else "#0F172A"
                        
                        chain_html += f'<div style="margin-left: {indent_px}px; padding: 2px 0;">'
                        chain_html += f'<span style="color: #94A3B8;">↳</span> <span style="color:#1D4ED8; font-weight:500;">{clean_subj}</span> <i style="color:#64748B;">{clean_pred}</i> <span style="color:{obj_color}; font-weight:600;">{clean_obj}</span>'
                        chain_html += '</div>'
                        
                    # Inject the dashed separator after a complete chain finishes
                    chain_html += '<div style="border-bottom: 1px dashed #CBD5E1; margin: 8px 0 8px 10px;"></div>'

                chain_html += '</div>'
                st.markdown(chain_html, unsafe_allow_html=True)
            else:
                st.markdown("*No semantic rules triggered for this feature.*")

def render_swrl_audit_tab(audit_trail, veh_id):
    """Renders the SWRL Governance Audit tab using dynamically colored HTML cards"""
    # Merged the Vehicle ID directly into the header!
    st.markdown(f"#### 🛡️ Dynamic Rule Governance (Vehicle {veh_id})")
    
    if not audit_trail:
        st.info("No active SWRL constraints triggered for this VOPair.")
        return

    for rule in audit_trail:
        is_pass = rule['swrl_status'] == "PASS"
        icon = "✅" if is_pass else "❌"
        border_color = "#10B981" if is_pass else "#E11D48"
        bg_color = "rgba(16, 185, 129, 0.05)" if is_pass else "rgba(225, 29, 72, 0.05)"
        
        act_color = "#3B82F6" if rule['actionability'] == "Actionable" else "#64748B"
        badge = f"<span style='background-color:{act_color}; color:white; padding:3px 8px; border-radius:12px; font-size:11px; font-weight:600;'>{rule['actionability']}</span>"
        
        # Clean formatting for values
        val_str = f"{float(rule['actual_value']):.2f}"
        lim_str = f"{float(rule['dynamic_limit']):.2f}"
        # Override C4.5 tree operators with true physical safety bounds??? 
        benefit_features = ["Urgency Rem. (min)", "Shift Rem. (min)", "Time Budget (min)"]
        logical_op = "&ge;" if rule['feature'] in benefit_features else "&le;"
        
        card_html = f"""
        <div style="border-left: 4px solid {border_color}; background-color: {bg_color}; padding: 12px 16px; margin-bottom: 12px; border-radius: 4px; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong style="font-size: 15px; color: #0F172A;">{icon} {rule['feature']}</strong>
                {badge}
            </div>
            <div style="font-size: 13.5px; color: #475569; margin-bottom: 8px; background-color: rgba(255,255,255,0.6); padding: 6px; border-radius: 4px;">
                <span style="margin-right: 15px;"><b>Current Value:</b> {val_str}</span> 
                <span><b>Safety Limit:</b> {logical_op} {lim_str}</span>
            </div>
            <div style="font-size: 13.5px; font-weight: 600; color: {border_color};">
                ↳ {rule['swrl_message']}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)      
                
@st.dialog("🧠 AI Dispatch Proposal")
def xai_decision_dialog(order_id, alloc_veh, p_node, d_node, prod_type, weight, pallets, urgency):
    st.markdown(f"### 📦 Order {order_id} Pending Allocation")
    
    # 1. Rich Dispatcher Information
    st.markdown(
        f"**Route:** Factory {p_node} ➔ Factory {d_node} &nbsp;|&nbsp; "
        f"**Product:** {prod_type}<br>"
        f"**Payload:** {weight}kg ({pallets} Pallets) &nbsp;|&nbsp; "
        f"**Urgency:** {urgency} mins to deadline", 
        unsafe_allow_html=True
    )
    
    xai_data = load_xai_logs().get(str(order_id))
    
    if not xai_data:
        st.info("Awaiting S-XAI semantic parsing logs for this order...")
        return
        
    st.divider()
    
    # 2. Tabs for Proposal vs Override vs Governance
    tab1, tab2, tab3 = st.tabs(["✅ AI Proposal (Active)", "⚙️ Manual Override Options", "⚖️ SWRL Governance Audit"])
    
    with tab1:
        st.success(f"**Proposed Action:** Allocate to **Vehicle {alloc_veh}**")
        veh_1_data = xai_data["vehicles"].get(str(alloc_veh))
        if veh_1_data:
            st.markdown("#### Top Decision Factors (Click to expand causality)")
            render_explanation_expanders(veh_1_data)
        
        if st.button("Approve AI Allocation", use_container_width=True, type="primary"):
            st.session_state.active_o_cluster = None
            st.rerun()

    with tab2:
        # Extract valid vehicle IDs from the zero-indexed mask
        valid_indices = [i for i, val in enumerate(xai_data["mask"]) if val]
        valid_veh_ids = [i + 1 for i in valid_indices]
        alternative_vehs = [v for v in valid_veh_ids if v != alloc_veh]
        
        selected_override = st.selectbox("Select Alternative Vehicle:", alternative_vehs, index=alternative_vehs.index(10) if 10 in alternative_vehs else 0)
        
        override_data = xai_data["vehicles"].get(str(selected_override))
        if override_data:
            st.warning(f"**Diagnostic Analysis vs Vehicle {selected_override}**")
            render_explanation_expanders(override_data)
            st.button(f"Force Assign to Veh {selected_override}", use_container_width=True)
        else:
            st.info("Counterfactual SHAP data not generated for this vehicle.")

    with tab3:
        swrl_logs = load_swrl_logs()
        swrl_order_data = swrl_logs.get(str(order_id), {}).get("vehicles", {})
        
        if not swrl_order_data:
            st.info("SWRL Governance data not generated for this order.")
        else:
            available_swrl_vehs = sorted([int(v) for v in swrl_order_data.keys()])
            
            # 1. Hide the dropdown if there is only 1 vehicle
            if len(available_swrl_vehs) > 1:
                selected_swrl_veh = st.selectbox(
                    "Select Vehicle to Audit:", 
                    options=available_swrl_vehs,
                    index=available_swrl_vehs.index(alloc_veh) if alloc_veh in available_swrl_vehs else 0,
                    key="swrl_audit_veh_select"
                )
            else:
                selected_swrl_veh = available_swrl_vehs[0]
                st.caption(f"**Vehicle {selected_swrl_veh}** is the only valid allocation. Dynamic safety bounds are automatically relaxed.")
            
            # 2. Render the specific vehicle's trail
            swrl_audit_trail = swrl_order_data.get(str(selected_swrl_veh), [])
            
            if swrl_audit_trail:
                # We removed the st.info banner entirely and pass the veh_id to the render function
                render_swrl_audit_tab(swrl_audit_trail, selected_swrl_veh)
            else:
                st.info(f"No audit trail found for Vehicle {selected_swrl_veh}.")
            
# 2 & 3. Load Data & Build Network (Keep your existing functions here)
@st.cache_data
def load_routing_data():
    file_path = "create_real_dpdp_glossary_of_terms.xlsx"
    return pd.read_excel(file_path, sheet_name="route_info")

route_df = load_routing_data()

@st.cache_data
def load_vehicle_data():
    file_path = "create_real_dpdp_glossary_of_terms.xlsx"
    df = pd.read_excel(file_path, sheet_name="vehicle_info")
    # Clean up empty rows just in case
    df = df.dropna(subset=['vehicle_id'])
    return df

vehicle_df = load_vehicle_data()

@st.cache_data
def load_simulation_history():
    # Load the new v2 datasets
    df_v = pd.read_csv("simulation_history_vehicle_2.csv")
    df_o = pd.read_csv("simulation_history_order_2.csv")
    
    # --- DATA CLEANING FOR S-XAI ---
    # 1. Drop records from before the order was actually created
    df_o = df_o[df_o['time_minute'] >= df_o['hasCreationTimeMinutes']].copy()
    
    # 2. Rename the transitional 'ungenerated' state to 'waiting' (unallocated)
    df_o.loc[df_o['CurrentState'].str.lower() == 'ungenerated', 'CurrentState'] = 'waiting'
    
    return df_v, df_o
vehicle_history_df, order_history_df = load_simulation_history()

STATE_MAP = {
    0: "Waiting",
    1: "Picking up",
    2: "Loading",
    3: "Delivering",
    4: "Unloading",
    5: "Returning"
}


def build_abstract_network(route_df):
    G = nx.DiGraph() 
    nodes = {
        10: {"pos": (15, 50), "city": "Old Downtown Lyon"},
        6:  {"pos": (32, 45), "city": "Part-Dieu"},
        5:  {"pos": (40, 80), "city": "Villeurbanne"},
        2:  {"pos": (50, 80), "city": "Villeurbanne"},
        0:  {"pos": (55, 55), "city": "Bron (Depot)"},
        7:  {"pos": (40, 15), "city": "Vénissieux"},
        3:  {"pos": (50, 15), "city": "Vénissieux"},
        1:  {"pos": (70, 77), "city": "Meyzieu"},
        9:  {"pos": (75, 42), "city": "Chassieu"},
        4:  {"pos": (75, 12), "city": "Saint-Priest"},
        8:  {"pos": (92, 45), "city": "Genas"}
    }
    for node in sorted(nodes.keys()):
        G.add_node(node, pos=nodes[node]["pos"], city=nodes[node]["city"])
    for _, row in route_df.iterrows():
        u, v = row['hasStartNodeAt'], row['hasEndNodeAt']
        dist, time = row['hasTravelingDistance(km)'], row['hasTravelingTime(min)']
        restrict = row['hasTimeRestrict']
        is_bi = row['isBidirectionalRoute']
        G.add_edge(u, v, dist=dist, time=time, restrict=restrict, is_bi=is_bi)
        if is_bi == 1:
            G.add_edge(v, u, dist=dist, time=time, restrict=restrict, is_bi=is_bi)
    return G

G = build_abstract_network(route_df)
node_ids = list(G.nodes())

# 4. Streamlit Sidebar UI & Navigation
with st.sidebar:
    st.header("🎛️ DPDP Control Panel")
    
    # --- NAVIGATION TOGGLE ---
    app_mode = st.radio("Dashboard View:", [
        "Topology Overview", 
        "Dispatcher Mode (Global)", 
        "Customer View (Single Order)"
    ], index=1)
    st.divider()
            
    # --- MODE 1: TOPOLOGY SIDEBAR ---
    if app_mode == "Topology Overview":
        st.subheader("Route Inspector")
        active_n = st.session_state.active_node
        
        if active_n is not None:
            st.success(f"Currently Inspecting: **Factory {active_n}**")
            if st.button("Reset View"):
                st.session_state.active_node = None
                st.rerun()
        else:
            st.info("Click on any Factory on the map to highlight its routes.")
            
    # --- MODE 2 & 3: SIMULATION CONTROLS ---
    elif app_mode in ["Dispatcher Mode (Global)", "Customer View (Single Order)"]:
        st.subheader("Simulator Controls")
        
        if 'sim_time_minutes' not in st.session_state:
            st.session_state.sim_time_minutes = 0 
            
        hrs = st.session_state.sim_time_minutes // 60
        mins = st.session_state.sim_time_minutes % 60
        st.metric(label="Simulated Time", value=f"{hrs:02d}:{mins:02d}")
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ +10 mins"): 
                st.session_state.sim_time_minutes += 10
                st.rerun()
        with col2:
            if st.button("⏭️ +1 hour"): 
                st.session_state.sim_time_minutes += 60
                st.rerun()
                
        if st.button("⏹️ Reset to 00:00", use_container_width=True):
            st.session_state.sim_time_minutes = 0
            st.rerun()

        # Only show the specific order search box in Customer View
        if app_mode == "Customer View (Single Order)":
            st.divider()
            st.subheader("🔍 Locate Order")
            if 'active_order' not in st.session_state:
                st.session_state.active_order = "None"
            if not order_history_df.empty:
                order_list = ["None"] + sorted(order_history_df['order_id'].unique().tolist())
                selected_order = st.selectbox("Track Specific Order:", order_list, index=order_list.index(st.session_state.active_order))
                if selected_order != st.session_state.active_order:
                    st.session_state.active_order = selected_order
                    st.session_state.active_node = None
                    st.session_state.active_v_cluster = None
                    st.session_state.active_o_cluster = None
                    st.rerun()
# ==========================================
# 5. SHARED VISUAL ASSETS (Used by both modes)
# ==========================================

# Background shapes
subtle_line = dict(color="rgba(200, 200, 200, 0.4)", width=1)
bg_fill = "rgba(245, 245, 245, 0.3)"

shapes = [
    dict(type="rect", x0=5, y0=20, x1=25, y1=80, line=subtle_line, fillcolor=bg_fill, layer="below"),
    dict(type="rect", x0=25, y0=30, x1=40, y1=70, line=subtle_line, fillcolor=bg_fill, layer="below"),
    dict(type="rect", x0=35, y0=70, x1=55, y1=90, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=45, y0=45, x1=65, y1=65, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=35, y0=10, x1=55, y1=25, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=60, y0=70, x1=80, y1=85, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=65, y0=35, x1=85, y1=50, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=85, y0=35, x1=100, y1=55, line=subtle_line, fillcolor=bg_fill, layer="below"), 
    dict(type="rect", x0=65, y0=5, x1=85, y1=20, line=subtle_line, fillcolor=bg_fill, layer="below")
]

# Zone Labels (Updated with accurate Lyon Metropolitan topology)
label_font = dict(size=11, color="rgba(120, 120, 120, 0.9)", family="Arial Black")
zone_annotations = [
    dict(x=15, y=79, text="DOWNTOWN LYON", showarrow=False, font=label_font, yanchor="top"),
    dict(x=32.5, y=69, text="PART-DIEU", showarrow=False, font=label_font, yanchor="top"),
    dict(x=45, y=89, text="VILLEURBANNE", showarrow=False, font=label_font, yanchor="top"),
    dict(x=55, y=64, text="BRON", showarrow=False, font=label_font, yanchor="top"),
    dict(x=45, y=24, text="VÉNISSIEUX", showarrow=False, font=label_font, yanchor="top"),
    dict(x=70, y=84, text="MEYZIEU", showarrow=False, font=label_font, yanchor="top"),
    dict(x=75, y=49, text="CHASSIEU", showarrow=False, font=label_font, yanchor="top"),
    dict(x=92.5, y=54, text="GENAS", showarrow=False, font=label_font, yanchor="top"),
    dict(x=75, y=19, text="SAINT-PRIEST", showarrow=False, font=label_font, yanchor="top")
]
# Base Nodes
# node_x, node_y, node_text, hover_texts = [], [], [], []
# for node in G.nodes():
#     x, y = G.nodes[node]['pos']
#     node_x.append(x)
#     node_y.append(y)
#     node_text.append(f"<b>{node}</b>")
#     hover_texts.append(f"<b>Factory {node}</b><br>Zone: {G.nodes[node]['zone']}")
node_x, node_y, node_text, hover_texts = [], [], [], []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    node_text.append(f"<b>{node}</b>")
    hover_texts.append(f"<b>{G.nodes[node]['city']}</b><br>Factory {node}")

# ==========================================
# MAIN PAGE ROUTING
# ==========================================

if app_mode == "Topology Overview":
    st.title("Dynamic Routing Topology")
    
    fig = go.Figure()
    
    # Nodes
    node_colors = ['#FFD700' if n == st.session_state.active_node else 'white' for n in G.nodes()]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center",
        textfont=dict(size=18, color="black"),
        hoverinfo='text', hovertext=hover_texts,
        marker=dict(color=node_colors, size=35, line=dict(width=2, color='black')),
        unselected=dict(marker=dict(opacity=1), textfont=dict(color="black")),
        selected=dict(marker=dict(opacity=1), textfont=dict(color="black"))
    ))

    # Edges
    annotations = list(zone_annotations) # Copy the base zone labels
    active_n = st.session_state.active_node
    
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        
        if active_n is None:
            arrow_color, width, arrow_size = "rgba(150, 150, 150, 0.15)", 1.0, 0.8
        elif u == active_n or v == active_n:
            arrow_color, width, arrow_size = "rgba(59, 130, 246, 0.85)", 1.8, 1.2
        else:
            arrow_color, width, arrow_size = "rgba(200, 200, 200, 0.05)", 0.5, 0.5

        annotations.append(dict(
            ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
            showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=width, 
            arrowcolor=arrow_color, standoff=24, startstandoff=24
        ))

    # Floating Info Window
    if active_n is not None:
        outbound = [(v, data) for u, v, data in G.edges(data=True) if u == active_n]
        inbound = [(u, data) for u, v, data in G.edges(data=True) if v == active_n]
        
        panel_text = f"<span style='font-size: 16px;'><b>🛰️ Factory {active_n} Routes</b></span><br><br>"
        if outbound:
            panel_text += "<b>📤 OUTBOUND (Can drive to):</b><br>"
            for target, data in outbound:
                bi_text = "🔄 Two-Way" if data.get('is_bi') == 1 else "➡️ One-Way"
                panel_text += f"➔ To <b>Factory {target}</b> ({data.get('dist')}km, {data.get('time')}min)<br>     <span style='color:gray; font-size:11px;'>Type: {bi_text} | Restrict: {data.get('restrict', 'None')}</span><br>"
            panel_text += "<br>"
        if inbound:
            panel_text += "<b>📥 INBOUND (Can drive from):</b><br>"
            for origin, data in inbound:
                if data.get('is_bi') == 0: 
                    panel_text += f"➔ From <b>Factory {origin}</b> ({data.get('dist')}km, {data.get('time')}min)<br>     <span style='color:gray; font-size:11px;'>Type: ⬅️ One-Way | Restrict: {data.get('restrict', 'None')}</span><br>"
                    
        annotations.append(dict(
            text=panel_text, align="left", showarrow=False, xref="paper", yref="paper", 
            x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255, 255, 255, 0.95)", 
            bordercolor="black", borderwidth=2, borderpad=12, font=dict(size=13, color="black")
        ))

    fig.update_layout(
        showlegend=False, hovermode='closest', shapes=shapes, annotations=annotations,
        margin=dict(b=20,l=5,r=5,t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 105]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        height=750, plot_bgcolor="white", clickmode='event+select', uirevision="constant"
    )

    selected_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="map_selection")

    if selected_event and "selection" in selected_event:
        points = selected_event["selection"].get("points", [])
        if points:
            point_index = points[0].get("point_index")
            if point_index is not None and point_index < len(node_ids):
                clicked_node = node_ids[point_index]
                if st.session_state.active_node != clicked_node:
                    st.session_state.active_node = clicked_node
                    st.session_state.click_buffer = "node_selected" 
                    st.rerun()
                elif st.session_state.active_node == clicked_node:
                    st.session_state.active_node = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
        else:
            if st.session_state.active_node is not None:
                if st.session_state.click_buffer == "node_selected":
                    st.session_state.click_buffer = "ready_for_background"
                elif st.session_state.click_buffer == "ready_for_background":
                    st.session_state.active_node = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()

elif app_mode == "Live Simulation":
    st.title("Live DPDP Simulator")
    
    sim_fig = go.Figure()
    
    # 1. State Variables
    active_n = st.session_state.active_node
    active_cluster = st.session_state.active_v_cluster
    active_o = st.session_state.get('active_order', "None")
    sim_annotations = list(zone_annotations)
    
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        
        is_highlighted = False
        if active_n is not None and (u == active_n or v == active_n):
            is_highlighted = True
        elif active_cluster is not None and active_cluster[0] == "edge" and active_cluster[1] == (u, v):
            is_highlighted = True
            
        if is_highlighted:
            arrow_color, width, arrow_size = "rgba(59, 130, 246, 0.85)", 1.8, 1.2
        else:
            arrow_color, width, arrow_size = "rgba(200, 200, 200, 0.2)", 0.8, 0.8
            
        sim_annotations.append(dict(
            ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
            showarrow=True, arrowhead=2, arrowsize=arrow_size, arrowwidth=width, 
            arrowcolor=arrow_color, standoff=24, startstandoff=24
        ))
        
    sim_fig.update_layout(shapes=shapes)
    
    current_time = st.session_state.sim_time_minutes
    current_orders = order_history_df[order_history_df['time_minute'] == current_time] if not order_history_df.empty else pd.DataFrame()
    
    # 2. Extract Active Order Real-Time Data
    order_data = None
    pickup_node, delivery_node, allocated_veh = None, None, 0
    if active_o != "None" and not current_orders.empty:
        order_row = current_orders[current_orders['order_id'] == active_o]
        if not order_row.empty:
            order_data = order_row.iloc[0]
            pickup_node = int(order_data['hasPickupFactory'])
            delivery_node = int(order_data['hasDeliveryFactory'])
            allocated_veh = int(order_data['AllocatedVehicle'])

    # 3. Nodes (Trace 0) - Dynamic Order Coloring
    sim_node_colors = []
    for n in G.nodes():
        if n == active_n:
            sim_node_colors.append('#FFD700') # Selected Node: Yellow
        elif n == pickup_node:
            sim_node_colors.append('#F97316') # Order Pickup: Bright Orange
        elif n == delivery_node:
            sim_node_colors.append('#3B82F6') # Order Delivery: Bright Blue
        else:
            sim_node_colors.append('white')
            
    sim_fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition="middle center", textfont=dict(size=18, color="black"),
        hoverinfo='text', hovertext=hover_texts,
        marker=dict(color=sim_node_colors, size=35, line=dict(width=2, color='black')),
        unselected=dict(marker=dict(opacity=1), textfont=dict(color="black")),
        selected=dict(marker=dict(opacity=1), textfont=dict(color="black"))
    ))
    
    # ---------------------------------------------------------
    # 4. VEHICLE LAYER (Trace 1) - LIVE DATA BRIDGE
    # ---------------------------------------------------------
    current_time = st.session_state.sim_time_minutes
    current_history = vehicle_history_df[vehicle_history_df['time_minute'] == current_time]
    
    dynamic_states = []
    
    # Fallback: If time is before our first log, stack everyone at the depot (Node 0)
    if current_history.empty and current_time < vehicle_history_df['time_minute'].min():
        for vid in vehicle_history_df['vehicle_id'].unique():
            dynamic_states.append({"v_id": float(vid), "state": "Waiting", "loc_type": "node", "loc": 0})
            
    # Fallback: If we go past the end of the simulation, keep the last known state
    elif current_history.empty and current_time > vehicle_history_df['time_minute'].max():
        last_time = vehicle_history_df['time_minute'].max()
        current_history = vehicle_history_df[vehicle_history_df['time_minute'] == last_time]
        
    # Process the active timestep
    if not current_history.empty:
        for _, row in current_history.iterrows():
            vid = float(row['vehicle_id'])
            raw_state = int(row['CurrentState'])
            state_str = STATE_MAP.get(raw_state, "Unknown")
            
            pos_type = str(row['CurrentPosition']).lower()
            pos_num = int(row['CurrentPositionNumber'])
            prev_node = int(row['PreviousFactoryId'])
            
            if pos_type == 'factory':
                dynamic_states.append({
                    "v_id": vid, 
                    "state": state_str, 
                    "loc_type": "node", 
                    "loc": pos_num
                })
            elif pos_type == 'route':
                # Map the route numero to your route_df
                # (Assuming pos_num perfectly matches the dataframe's row index)
                route_row = route_df[route_df['route_id'] == pos_num].iloc[0]
                node_a = int(route_row['hasStartNodeAt'])
                node_b = int(route_row['hasEndNodeAt'])
                
                # Determine direction of travel based on where it came from
                if prev_node == node_a:
                    u, v = node_b, node_a
                else:
                    u, v = node_a, node_b
                    
                dynamic_states.append({
                    "v_id": vid, 
                    "state": state_str, 
                    "loc_type": "edge", 
                    "loc": (u, v)
                })

    # --- CLUSTERING LOGIC ---
    from collections import defaultdict
    clusters = defaultdict(list)
    for vs in dynamic_states:  # Use the new dynamic_states instead of dummy_states
        clusters[(vs['loc_type'], vs['loc'])].append(vs)       
    v_x, v_y, v_text, v_hover, v_colors = [], [], [], [], []
    cluster_keys = [] 
    
    for (loc_type, loc_val), v_list in clusters.items():
        cluster_keys.append((loc_type, loc_val))
        
        if loc_type == "node":
            nx, ny = G.nodes[loc_val]['pos']
            x, y = nx + 2, ny + 2.5  
            loc_str = f"Factory {loc_val}"
        else: 
            u, v = loc_val
            nx0, ny0 = G.nodes[u]['pos']
            nx1, ny1 = G.nodes[v]['pos']
            x, y = (nx0 + nx1) / 2, (ny0 + ny1) / 2
            loc_str = f"Route {u} ➔ {v}"
            
        v_x.append(x)
        v_y.append(y)
        
        # Fully spelled out labels for French supervisors
        v_count = len(v_list)
        if v_count == 1:
            v_text.append(f"<b>Vehicle {int(v_list[0]['v_id'])}</b>")
            v_hover.append(f"<b>Vehicle {int(v_list[0]['v_id'])}</b><br>State: {v_list[0]['state']}<br>Location: {loc_str}")
        else:
            v_text.append(f"<b>{v_count} Vehicles</b>")
            v_ids_str = ", ".join([str(int(v['v_id'])) for v in v_list])
            v_hover.append(f"<b>{v_count} Vehicles Here</b><br>IDs: {v_ids_str}<br>Location: {loc_str}")
            
        #color = '#10B981' if active_cluster is None or active_cluster == (loc_type, loc_val) else 'rgba(16, 185, 129, 0.2)'
        has_tracked_order = any(int(v['v_id']) == allocated_veh for v in v_list) if allocated_veh != 0 else False
        
        if has_tracked_order:
            color = '#F97316' # Orange to match the order tracking theme
        elif active_cluster is None or active_cluster == (loc_type, loc_val):
            color = '#10B981' # Default Green
        else:
            color = 'rgba(16, 185, 129, 0.2)' # Faded Green
        v_colors.append(color)

    sim_fig.add_trace(go.Scatter(
        x=v_x, y=v_y, mode='markers+text',
        text=v_text, textposition="top right", textfont=dict(size=12, color="#065F46"),
        hoverinfo='text', hovertext=v_hover,
        marker=dict(symbol='square', size=20, color=v_colors, line=dict(width=2, color='black')), 
        unselected=dict(marker=dict(opacity=1), textfont=dict(color="#065F46")),
        selected=dict(marker=dict(opacity=1), textfont=dict(color="#065F46"))
    ))
    # 5. Order Manifest Floating Window (Anchored Top-Left)
    if order_data is not None:
        state_str = str(order_data['CurrentState']).upper()
        
        panel_text = f"<span style='font-size: 15px;'><b>📦 Order {active_o} Manifest</b></span><br><br>"
        panel_text += f"<b>Status:</b> {state_str}<br>"
        if allocated_veh != 0:
            panel_text += f"<b>Allocated To:</b> Vehicle {allocated_veh}<br>"
        panel_text += f"<b>Route:</b> Factory {pickup_node} ➔ Factory {delivery_node}<br><br>"
        
        # Pulling from the exact CSV column names
        panel_text += f"<b>Product:</b> {order_data['hasOrderProductType']}<br>"
        panel_text += f"<b>Weight:</b> {order_data['hasScaleOfCargo(kg)']} kg<br>"
        panel_text += f"<b>Pallets:</b> {order_data['ScaleOfCargo hasNumberOfPallets']}<br>"
        panel_text += f"<b>Created:</b> {order_data['hasCreationTimeMinutes']} min<br>"
        panel_text += f"<b>Due By:</b> {order_data['hasCommittedDeliveryTimeMinutes']} min<br>"

        sim_annotations.append(dict(
            text=panel_text, align="left", showarrow=False, xref="paper", yref="paper", 
            x=0.01, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255, 247, 237, 0.95)", # Pale Orange
            bordercolor="#F97316", borderwidth=2, borderpad=10, font=dict(size=12, color="black")
        ))
        
        # 5. Vehicle Floating Info Window (No abbreviations)
    if active_cluster is not None:
        loc_type, loc_val = active_cluster
        loc_str = f"Factory {loc_val}" if loc_type == "node" else f"Route {loc_val[0]} ➔ {loc_val[1]}"
        cluster_states = [vs for vs in dynamic_states if vs['loc_type'] == loc_type and vs['loc'] == loc_val]
        
        panel_text = f"<span style='font-size: 15px;'><b>🚚 {len(cluster_states)} Vehicle(s) at {loc_str}</b></span><br><br>"
        
        for vs in cluster_states:
            v_data = vehicle_df[vehicle_df['vehicle_id'] == vs['v_id']].iloc[0]
            vid = int(vs['v_id'])
            state = vs['state'].upper()
            cap = v_data['hasLoadingCapacity(kg)']
            fuel = v_data['hasNumberOfFuelConsumption(L/km)']
            type_str = v_data['hasProductAdaptivity']
            
            # Explicit formatting
            panel_text += f"<b>Vehicle {vid} ({state}):</b><br>"
            panel_text += f"&nbsp;&nbsp;• Capacity: {cap} kg<br>"
            panel_text += f"&nbsp;&nbsp;• Fuel Consumption: {fuel} L/km<br>"
            panel_text += f"&nbsp;&nbsp;• Type: {type_str}<br><br>"

        sim_annotations.append(dict(
            text=panel_text, align="left", showarrow=False, xref="paper", yref="paper", 
            x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(240, 253, 244, 0.95)", 
            bordercolor="#10B981", borderwidth=2, borderpad=10, font=dict(size=12, color="black")
        ))
    
    sim_fig.update_layout(
        showlegend=False, hovermode='closest', annotations=sim_annotations,
        margin=dict(b=20,l=5,r=5,t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 105]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        height=750, plot_bgcolor="white", clickmode='event+select', uirevision="constant"
    )
    
    # 6. Render Chart & Catch Clicks
    selected_event = st.plotly_chart(sim_fig, use_container_width=True, on_select="rerun", selection_mode="points", key="sim_selection")
    
    # 7. Dual Click Processor
    if selected_event and "selection" in selected_event:
        points = selected_event["selection"].get("points", [])
        if points:
            pt = points[0]
            curve = pt.get("curve_number")
            idx = pt.get("point_index")
            
            if curve == 0:  
                clicked_n = node_ids[idx]
                if st.session_state.active_node != clicked_n:
                    st.session_state.active_node = clicked_n
                    st.session_state.active_v_cluster = None 
                    st.session_state.click_buffer = "node_selected" 
                    st.rerun()
                else:
                    st.session_state.active_node = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
                    
            elif curve == 1: 
                clicked_cluster = cluster_keys[idx]
                if st.session_state.active_v_cluster != clicked_cluster:
                    st.session_state.active_v_cluster = clicked_cluster
                    st.session_state.active_node = None 
                    st.session_state.click_buffer = "node_selected" 
                    st.rerun()
                else:
                    st.session_state.active_v_cluster = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
        else:
            if st.session_state.active_node is not None or st.session_state.active_v_cluster is not None:
                if st.session_state.click_buffer == "node_selected":
                    st.session_state.click_buffer = "ready_for_background"
                elif st.session_state.click_buffer == "ready_for_background":
                    st.session_state.active_node = None
                    st.session_state.active_v_cluster = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
                    
elif app_mode == "Dispatcher Mode (Global)":
    st.title("Live DPDP Dispatcher (Global Fleet)")
    
    sim_fig = go.Figure()
    active_n = st.session_state.active_node
    active_v_cluster = st.session_state.active_v_cluster
    active_o_cluster = st.session_state.active_o_cluster
    
    current_time = st.session_state.sim_time_minutes
    current_v_history = vehicle_history_df[vehicle_history_df['time_minute'] == current_time]
    
    # --- PRE-CALCULATE ACTIVE ROUTES FOR HIGHLIGHTING ---
    active_edges = set()
    if not current_v_history.empty:
        for _, row in current_v_history.iterrows():
            if str(row['CurrentPosition']).lower() == 'route':
                pos_num = int(row['CurrentPositionNumber'])
                prev_node = int(row['PreviousFactoryId'])
                route_row = route_df[route_df['route_id'] == pos_num].iloc[0]
                node_a, node_b = int(route_row['hasStartNodeAt']), int(route_row['hasEndNodeAt'])
                u, v = (node_b, node_a) if prev_node == node_a else (node_a, node_b)
                active_edges.add((u, v))
                active_edges.add((v, u))

    # 1. Background Map & Edges
    sim_annotations = []
    
    # Copy shapes so we don't modify the global shapes list on every rerun
    current_shapes = list(shapes)
    
    # Only show metropolitan names if a factory node is actively selected
    if active_n is not None:
        sim_annotations.extend(zone_annotations)
        
    for u, v, data in G.edges(data=True):
        x0, y0 = G.nodes[u]['pos']
        x1, y1 = G.nodes[v]['pos']
        
        is_selected_fwd = False
        is_selected_rev = False
        
        # Check if this edge is the selected one
        if active_v_cluster is not None and active_v_cluster[0] == "edge":
            target_u, target_v = active_v_cluster[1]
            if (u, v) == (target_u, target_v):
                is_selected_fwd = True
            elif (u, v) == (target_v, target_u):
                is_selected_rev = True
                
        is_active = (u, v) in active_edges
        
        if is_selected_fwd:
            # Selected Direction: Green WITH arrow (Annotation, drawn on top)
            sim_annotations.append(dict(
                ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.5, arrowcolor="rgba(16, 185, 129, 0.95)", standoff=24, startstandoff=24
            ))
        elif is_selected_rev:
            # Reverse Direction: Green NO arrow (Shape, drawn below markers)
            current_shapes.append(dict(
                type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="rgba(16, 185, 129, 0.95)", width=2.5), layer="below"
            ))
        elif is_active:
            # Active Route: Bold Blue NO arrow (Shape, drawn below markers!)
            current_shapes.append(dict(
                type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="rgba(59, 130, 246, 0.6)", width=1.5), layer="below"
            ))
        else:
            # Base Topology: Faint Gray WITH arrow (Annotation)
            sim_annotations.append(dict(
                ax=x0, ay=y0, axref='x', ayref='y', x=x1, y=y1, xref='x', yref='y',
                showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=0.8, arrowcolor="rgba(200, 200, 200, 0.2)", standoff=24, startstandoff=24
            ))
            
    sim_fig.update_layout(shapes=current_shapes)
    
    # 2. GLOBAL VEHICLE LOGIC (Trace 0 - Bottom Layer)
    
    # --- FIND VEHICLES RELEVANT TO SELECTED ORDER CLUSTER ---
    active_o_veh_ids = set()
    if active_o_cluster is not None and not order_history_df.empty:
        past_o = order_history_df[order_history_df['time_minute'] <= current_time]
        if not past_o.empty:
            l_orders = past_o.sort_values('time_minute').groupby('order_id').last()
            rel_orders = l_orders[
                ((l_orders['hasPickupFactory'] == active_o_cluster) & (l_orders['CurrentState'].str.lower().isin(['waiting', 'delayed', 'allocated']))) |
                ((l_orders['hasDeliveryFactory'] == active_o_cluster) & (l_orders['CurrentState'].str.lower() == 'finished'))
            ]
            active_o_veh_ids = set(rel_orders['AllocatedVehicle'].dropna().astype(int))

    dynamic_v_states = []
    if not current_v_history.empty:
        for _, row in current_v_history.iterrows():
            vid = float(row['vehicle_id'])
            state_str = STATE_MAP.get(int(row['CurrentState']), "Unknown")
            pos_type = str(row['CurrentPosition']).lower()
            pos_num = int(row['CurrentPositionNumber'])
            prev_node = int(row['PreviousFactoryId'])
            
            if pos_type == 'factory':
                dynamic_v_states.append({"v_id": vid, "state": state_str, "loc_type": "node", "loc": pos_num})
            elif pos_type == 'route':
                route_row = route_df[route_df['route_id'] == pos_num].iloc[0]
                node_a, node_b = int(route_row['hasStartNodeAt']), int(route_row['hasEndNodeAt'])
                u, v = (node_b, node_a) if prev_node == node_a else (node_a, node_b)
                dynamic_v_states.append({"v_id": vid, "state": state_str, "loc_type": "edge", "loc": (u, v)})

    from collections import defaultdict
    v_clusters = defaultdict(list)
    for vs in dynamic_v_states:
        v_clusters[(vs['loc_type'], vs['loc'])].append(vs)
        
    v_x, v_y, v_text, v_colors, v_line_colors, v_text_colors = [], [], [], [], [], []
    v_cluster_keys = [] 
    
    for (loc_type, loc_val), v_list in v_clusters.items():
        v_cluster_keys.append((loc_type, loc_val))
        if loc_type == "node":
            nx, ny = G.nodes[loc_val]['pos']
            x, y = nx + 2.5, ny + 2.5  
        else: 
            u, v = loc_val
            nx0, ny0 = G.nodes[u]['pos']
            nx1, ny1 = G.nodes[v]['pos']
            x, y = (nx0 + nx1) / 2, (ny0 + ny1) / 2
            
        v_x.append(x)
        v_y.append(y)
        v_text.append(f"<b>Vehicle {int(v_list[0]['v_id'])}</b>" if len(v_list) == 1 else f"<b>{len(v_list)} Vehicles</b>")
        
        # --- VEHICLE CROSS-FILTERING LOGIC ---
        is_relevant_to_order = any(int(v['v_id']) in active_o_veh_ids for v in v_list)
        
        if active_v_cluster is not None and active_v_cluster != (loc_type, loc_val):
            # Dim if another vehicle is selected
            v_colors.append('rgba(200, 200, 200, 0.2)')
            v_line_colors.append('rgba(0, 0, 0, 0.2)')
            v_text_colors.append('rgba(0, 0, 0, 0.2)')
        elif active_o_cluster is not None and not is_relevant_to_order:
            # Dim if an order is selected, but this vehicle is NOT carrying it
            v_colors.append('rgba(200, 200, 200, 0.2)')
            v_line_colors.append('rgba(0, 0, 0, 0.2)')
            v_text_colors.append('rgba(0, 0, 0, 0.2)')
        else:
            # Normal visibility
            v_colors.append('#10B981')
            v_line_colors.append('black')
            v_text_colors.append('#065F46')

    sim_fig.add_trace(go.Scatter(
        x=v_x, y=v_y, mode='markers+text', text=v_text, textposition="top right", 
        textfont=dict(size=12, color=v_text_colors),
        hoverinfo='none', marker=dict(symbol='square', size=20, color=v_colors, line=dict(width=2, color=v_line_colors))
    ))

    # 3. NODES LOGIC (Trace 1 - Middle Layer)
    sim_node_colors = ['#FFD700' if n == active_n else 'white' for n in G.nodes()]
    sim_fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="middle center", textfont=dict(size=18, color="black"),
        hoverinfo='text', hovertext=hover_texts, 
        marker=dict(color=sim_node_colors, size=35, line=dict(width=2, color='black')),
        unselected=dict(marker=dict(opacity=1), textfont=dict(color="black")),
        selected=dict(marker=dict(opacity=1), textfont=dict(color="black"))
    ))
    
# 4. GLOBAL ORDER LOGIC & DECAY (Trace 2)
    dynamic_o_states = []
    if not order_history_df.empty:
        past_o_history = order_history_df[order_history_df['time_minute'] <= current_time]
        if not past_o_history.empty:
            # Get the exact state of each order at this current minute (or its last known state)
            latest_orders = past_o_history.sort_values('time_minute').groupby('order_id').last().reset_index()
            
            for _, row in latest_orders.iterrows():
                state = str(row['CurrentState']).lower()
                oid = int(row['order_id'])
                veh = int(row['AllocatedVehicle']) if pd.notna(row['AllocatedVehicle']) else 0
                
                # Waiting -> Solid Orange at Pickup
                if state == 'waiting':
                    dynamic_o_states.append({"o_id": oid, "state": state, "loc": int(row['hasPickupFactory']), "opacity": 1.0, "veh": veh})
                # Delayed -> Solid Red at Pickup
                elif state == 'delayed':
                    dynamic_o_states.append({"o_id": oid, "state": state, "loc": int(row['hasPickupFactory']), "opacity": 1.0, "veh": veh})
                # Allocated -> Solid Yellow at Pickup
                elif state == 'allocated':
                    dynamic_o_states.append({"o_id": oid, "state": state, "loc": int(row['hasPickupFactory']), "opacity": 1.0, "veh": veh})
                # Loaded (Carrying) -> Solid Blue at Delivery (Shows as Incoming)
                elif state == 'loaded':
                    dynamic_o_states.append({"o_id": oid, "state": state, "loc": int(row['hasDeliveryFactory']), "opacity": 1.0, "veh": veh})
                # Finished -> Green at Delivery with 30-min fadeout
                elif state == 'finished':
                    finished_rows = past_o_history[(past_o_history['order_id'] == oid) & (past_o_history['CurrentState'].str.lower() == 'finished')]
                    if not finished_rows.empty:
                        finish_time = finished_rows['time_minute'].min()
                        decay_mins = current_time - finish_time
                        # Changed from <= 30 to < 30 to completely drop the object when opacity hits 0
                        if decay_mins < 30:
                            opacity = max(0.0, 1.0 - (decay_mins / 30.0))
                            dynamic_o_states.append({"o_id": oid, "state": state, "loc": int(row['hasDeliveryFactory']), "opacity": opacity, "veh": veh})

    o_clusters = defaultdict(list)
    for o_state in dynamic_o_states:
        o_clusters[o_state['loc']].append(o_state)
        
    o_x, o_y, o_text, o_colors, o_line_colors, o_text_colors = [], [], [], [], [], []
    o_cluster_keys = []
    
    # Pre-calculate selected vehicle IDs for cross-filtering
    active_v_ids = []
    if active_v_cluster is not None:
        active_v_ids = [v['v_id'] for v in v_clusters[active_v_cluster]]
    
    for loc_val, o_list in o_clusters.items():
        # Remove 'loaded' orders from the visual map rendering (they are inside the vehicle!)
        visible_o_list = [o for o in o_list if o['state'] != 'loaded']
        if not visible_o_list:
            continue # Skip drawing this cluster if all orders here are loaded inside vehicles
            
        o_cluster_keys.append(loc_val)
        nx, ny = G.nodes[loc_val]['pos']
        o_x.append(nx - 2.5) 
        o_y.append(ny - 2.5)
        
        # Calculate cluster opacity based on maximum visibility order
        max_opacity = max([o['opacity'] for o in visible_o_list])
        
        # Priority Coloring for Clusters
        states = [o['state'] for o in visible_o_list]
        if 'delayed' in states:
            color_str = f'rgba(239, 68, 68, {max_opacity})'  # Red (Critical - Highest Priority)
        elif 'waiting' in states:
            color_str = f'rgba(249, 115, 22, {max_opacity})' # Orange (Action Required - 2nd Priority)
        elif 'allocated' in states:
            color_str = f'rgba(59, 130, 246, {max_opacity})' # Blue (Safe/Stable - 3rd Priority)
        else:
            color_str = f'rgba(16, 185, 129, {max_opacity})' # Green (Finished - Lowest Priority)
            
        # --- CROSS-FILTERING LOGIC ---
        is_relevant_to_vehicle = any(o['veh'] in active_v_ids for o in visible_o_list)
        
        if active_v_cluster is not None and not is_relevant_to_vehicle:
            # Dim if a vehicle is selected, but this order cluster doesn't belong to it
            color_str = 'rgba(200, 200, 200, 0.1)'
            line_color = 'rgba(0, 0, 0, 0.1)'
            text_color = 'rgba(0, 0, 0, 0.1)'
        elif active_o_cluster is not None and active_o_cluster != loc_val:
            # Dim if another order cluster is explicitly selected
            color_str = 'rgba(200, 200, 200, 0.1)' 
            line_color = 'rgba(0, 0, 0, 0.1)'
            text_color = 'rgba(0, 0, 0, 0.1)'
        else:
            # Normal visibility
            line_color = f'rgba(0, 0, 0, {max_opacity})'
            text_color = f'rgba(0, 0, 0, {max_opacity})'
            
        # Dynamic Text with Standard Font Size for Vehicle IDs
        if len(visible_o_list) == 1:
            o = visible_o_list[0]
            if o['veh'] != 0:
                o_text.append(f"<b>Ord {o['o_id']}</b><br>(Veh {o['veh']})")
            else:
                o_text.append(f"<b>Ord {o['o_id']}</b>")
        else:
            txt = f"<b>{len(visible_o_list)} Orders</b>"
            vehs = set([o['veh'] for o in visible_o_list if o['veh'] != 0])
            if vehs:
                veh_str = ", ".join(map(str, list(vehs)[:3])) + ("..." if len(vehs) > 3 else "")
                txt += f"<br>(Vehs: {veh_str})"
            o_text.append(txt)
            
        o_colors.append(color_str)
        o_line_colors.append(line_color)
        o_text_colors.append(text_color)
        
    sim_fig.add_trace(go.Scatter(
        x=o_x, y=o_y, mode='markers+text', text=o_text, textposition="bottom left", 
        textfont=dict(size=12, color=o_text_colors), 
        hoverinfo='none', marker=dict(symbol='diamond', size=20, color=o_colors, line=dict(width=2, color=o_line_colors))
    ))

    # 5. DISPATCHER FLOATING WINDOWS
    if active_o_cluster is not None:
        o_list = o_clusters[active_o_cluster]
        city_name = G.nodes[active_o_cluster]['city']
        panel_text = f"<span style='font-size: 15px;'><b>📦 {len(o_list)} Orders at {city_name}</b></span><br><br>"
        
        for o in o_list[:5]: 
            o_data = latest_orders[latest_orders['order_id'] == o['o_id']].iloc[0]
            
            p_node = int(o_data['hasPickupFactory'])
            d_node = int(o_data['hasDeliveryFactory'])
            cargo_weight = o_data['hasScaleOfCargo(kg)']
            
            veh_str = f" <span style='color:#0284C7;'>➔ <b>Veh {o['veh']}</b></span>" if o['veh'] != 0 else ""
            panel_text += f"<b>Order {o['o_id']}</b> [F{p_node} ➔ F{d_node}] ({o['state'].upper()}){veh_str} | {cargo_weight}kg<br>"
            
        if len(o_list) > 5: 
            panel_text += f"<i>...and {len(o_list) - 5} more.</i><br>"
        
        sim_annotations.append(dict(
            text=panel_text, align="left", showarrow=False, xref="paper", yref="paper", 
            x=0.01, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255, 247, 237, 0.95)", 
            bordercolor="#F97316", borderwidth=2, borderpad=10, font=dict(size=12, color="black")
        ))
        
    if active_v_cluster is not None:
        loc_type, loc_val = active_v_cluster
        loc_str = G.nodes[loc_val]['city'] if loc_type == "node" else f"Route {loc_val[0]} ➔ {loc_val[1]}"
        v_list = v_clusters[active_v_cluster]
        
        panel_text = f"<span style='font-size: 15px;'><b>🚚 {len(v_list)} Vehicle(s) at {loc_str}</b></span><br><br>"
        for vs in v_list:
            vid = int(vs['v_id'])
            state = vs['state'].upper()
            panel_text += f"<b>Vehicle {vid} ({state})</b><br>"
            
            # Cross-reference loaded orders at this specific time minute
            current_o_history = order_history_df[order_history_df['time_minute'] == current_time] if not order_history_df.empty else pd.DataFrame()
            if not current_o_history.empty:
                loaded_orders = current_o_history[(current_o_history['CurrentState'].str.lower() == 'loaded') & (current_o_history['AllocatedVehicle'] == vid)]
                if not loaded_orders.empty:
                    panel_text += f"<span style='color:#EA580C;'>&nbsp;&nbsp;📦 Carrying: "
                    cargo_strs = []
                    for _, row in loaded_orders.iterrows():
                        p_node = int(row['hasPickupFactory'])
                        d_node = int(row['hasDeliveryFactory'])
                        cargo_strs.append(f"Ord {row['order_id']} [F{p_node}➔F{d_node}] ({row['hasScaleOfCargo(kg)']}kg)")
                    
                    panel_text += ", ".join(cargo_strs) + "</span><br>"
                else:
                    panel_text += "<span style='color:gray;'>&nbsp;&nbsp;<i>Empty Cargo</i></span><br>"
            panel_text += "<br>"

        sim_annotations.append(dict(
            text=panel_text, align="left", showarrow=False, xref="paper", yref="paper", 
            x=0.99, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(240, 253, 244, 0.95)", 
            bordercolor="#10B981", borderwidth=2, borderpad=10, font=dict(size=12, color="black")
        ))
        
    sim_fig.update_layout(
        showlegend=False, hovermode='closest', annotations=sim_annotations, margin=dict(b=20,l=5,r=5,t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 105]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        height=750, plot_bgcolor="white", clickmode='event+select', uirevision="constant"
    )
    
    # 5.5 S-XAI DIALOG TRIGGER
    if st.session_state.get('trigger_xai_dialog', False):
        st.session_state.trigger_xai_dialog = False # Reset immediately
        
        if active_o_cluster is not None:
            o_list = o_clusters[active_o_cluster]
            waiting_orders = [o for o in o_list if o['state'] == 'waiting']
            if waiting_orders:
                target_oid = waiting_orders[0]['o_id']
                future_o = order_history_df[(order_history_df['order_id'] == target_oid) & (order_history_df['CurrentState'] == 'allocated')]
                if not future_o.empty:
                    alloc_veh = int(future_o.iloc[0]['AllocatedVehicle'])
                    o_data = latest_orders[latest_orders['order_id'] == target_oid].iloc[0]
                    
                    # Extract rich dispatcher information
                    p_node = int(o_data['hasPickupFactory'])
                    d_node = int(o_data['hasDeliveryFactory'])
                    cargo_weight = o_data['hasScaleOfCargo(kg)']
                    prod_type = str(o_data['hasOrderProductType'])
                    pallets = int(o_data['ScaleOfCargo hasNumberOfPallets'])
                    urgency = int(o_data['hasCommittedDeliveryTimeMinutes']) - int(o_data['hasCreationTimeMinutes'])
                    
                    # Pop the dialog without the time parameter!
                    xai_decision_dialog(target_oid, alloc_veh, p_node, d_node, prod_type, cargo_weight, pallets, urgency)
                    
    # 6. Tri-Click Processor
    selected_event = st.plotly_chart(sim_fig, use_container_width=True, on_select="rerun", selection_mode="points", key="dispatcher_sim")
    if selected_event and "selection" in selected_event:
        points = selected_event["selection"].get("points", [])
        if points:
            pt = points[0]
            curve, idx = pt.get("curve_number"), pt.get("point_index")
            
            if curve == 1: # CLICKED NODE
                clicked_n = node_ids[idx]
                if st.session_state.active_node != clicked_n:
                    st.session_state.active_node = clicked_n
                    st.session_state.active_v_cluster = None 
                    st.session_state.active_o_cluster = None
                    st.session_state.click_buffer = "selected" 
                    st.rerun()
                else:
                    st.session_state.active_node = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
            elif curve == 0: # CLICKED VEHICLE
                clicked_v = v_cluster_keys[idx]
                if st.session_state.active_v_cluster != clicked_v:
                    st.session_state.active_v_cluster = clicked_v
                    st.session_state.active_node = None 
                    st.session_state.active_o_cluster = None
                    st.session_state.click_buffer = "selected" 
                    st.rerun()
                else:
                    st.session_state.active_v_cluster = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
            elif curve == 2: # CLICKED ORDER
                clicked_o = o_cluster_keys[idx]
                if st.session_state.active_o_cluster != clicked_o:
                    st.session_state.active_o_cluster = clicked_o
                    st.session_state.active_node = None 
                    st.session_state.active_v_cluster = None
                    st.session_state.click_buffer = "selected" 
                    
                    # TRIGGER THE DIALOG ON NEXT RERUN
                    st.session_state.trigger_xai_dialog = True 
                    st.rerun()
                else:
                    st.session_state.active_o_cluster = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()
        else:
            if any(st.session_state[k] is not None for k in ['active_node', 'active_v_cluster', 'active_o_cluster']):
                if st.session_state.click_buffer == "selected":
                    st.session_state.click_buffer = "ready_for_bg"
                elif st.session_state.click_buffer == "ready_for_bg":
                    st.session_state.active_node = None
                    st.session_state.active_v_cluster = None
                    st.session_state.active_o_cluster = None
                    st.session_state.click_buffer = "cleared"
                    st.rerun()


# In[ ]:




