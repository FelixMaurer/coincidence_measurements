import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import time

st.set_page_config(page_title="Coincidence Experiments", layout="wide")

# --- State Management for the Virtual MCB (Tab 1) ---
if 'mcb_data' not in st.session_state:
    st.session_state.mcb_data = []
if 'current_distance' not in st.session_state:
    st.session_state.current_distance = 4.0

st.title("Experimental Methods: Gamma and Positron Coincidence")
st.markdown("""
This application outlines the four distinct experimental setups required to measure positron lifetimes and gamma-gamma angular correlations. 
The experiments rely on coincidence measurements, where two related nuclear events are detected simultaneously (or with a measurable delay).
""")

# --- Helper Functions for Drawing (Tabs 2, 3, 4) ---
def draw_detector_setup(ax, source_label, det1_label, det2_label, det2_pos='shift', value=10, has_collimator=False):
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.axis('off')

    # Source
    source = patches.Circle((0, 0), 1, color='red', zorder=5)
    ax.add_patch(source)
    ax.text(0, 1.5, source_label, ha='center', fontsize=10, weight='bold')

    # Detector 1 (Left / Fixed)
    det1 = patches.Rectangle((-10, -2), 4, 4, color='blue', alpha=0.6)
    ax.add_patch(det1)
    ax.text(-8, -3.5, det1_label, ha='center')
    
    if has_collimator:
        collimator = patches.Rectangle((-6, -1), 0.5, 2, color='gray', alpha=0.9)
        ax.add_patch(collimator)
        ax.text(-5.5, 1.5, "Lead Collimator", ha='left', fontsize=8)

    # Detector 2 (Right / Moveable)
    if det2_pos == 'shift':
        distance_val = value
        det2 = patches.Rectangle((distance_val, -2), 4, 4, color='green', alpha=0.6)
        ax.add_patch(det2)
        ax.text(distance_val + 2, -3.5, det2_label, ha='center')
        ax.plot([1, distance_val], [0, 0], 'k--', alpha=0.5)
        ax.plot([-1, -6], [0, 0], 'k--', alpha=0.5)
    elif det2_pos == 'rotate':
        angle_deg = value
        radius = 8
        angle_rad = np.radians(angle_deg)
        x_center = radius * np.cos(angle_rad)
        y_center = radius * np.sin(angle_rad)
        
        det2 = patches.Rectangle((x_center - 2, y_center - 2), 4, 4, color='orange', alpha=0.6)
        t = transforms.Affine2D().rotate_deg_around(x_center, y_center, angle_deg) + ax.transData
        det2.set_transform(t)
        ax.add_patch(det2)
        
        ax.plot([0, x_center], [0, y_center], 'k--', alpha=0.5)
        ax.plot([0, -radius], [0, 0], 'k--', alpha=0.5)
        
        arc = patches.Arc((0, 0), 4, 4, theta1=angle_deg, theta2=180, color='black')
        ax.add_patch(arc)
        ax.text(0, 3, f"{angle_deg}°", ha='center', fontsize=12)

# --- Tabs for the 4 Setups ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Time Calibration", 
    "2. Positron Lifetime", 
    "3. Angular Efficiency", 
    "4. Angular Correlation"
])

# --- Setup 1: Time Calibration (Detailed Simulation) ---
with tab1:
    st.header("Setup 1: Time Calibration & Resolution")
    st.markdown("""
    This virtual experiment simulates the time calibration of the coincidence apparatus. 
    The setup uses a **22Na source**. When a positron annihilates, it creates two **511 keV** gamma quanta emitted back-to-back at **180°**. 
    We detect these using **BaF2 detectors**, which are excellent for time resolution. 
    
    By shifting Detector 2, we create a measurable time-of-flight difference to calibrate the Multichannel Buffer (MCB).
    """)
    
    # Tab-specific controls
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        distance_t1 = st.slider("Detector 2 Distance (cm)", min_value=4.0, max_value=40.0, value=st.session_state.current_distance, step=6.0, key="t1_dist")
        num_events = st.slider("Number of Sequential Events (N)", min_value=1, max_value=50, value=15, step=1)
        if distance_t1 != st.session_state.current_distance:
            st.session_state.mcb_data = []
            st.session_state.current_distance = distance_t1
    
    with col_ctrl2:
        st.write("Controls:")
        trigger_animation = st.button("Simulate Experiment (Animation)")
        trigger_bulk = st.button("Record 500 Events Rapidly")

    # Physics Constants
    c_cm_per_ps = 0.03 
    fixed_det_dist = 4.0 
    t1_true = fixed_det_dist / c_cm_per_ps
    t2_true = distance_t1 / c_cm_per_ps
    true_delta_t = t2_true - t1_true
    time_resolution_sigma = 150.0 # ps

    st.subheader("The Coincidence Principle & Time Uncertainty")
    st.markdown("""
    To avoid "false coincidences", the source activity must be low enough that the time between decays is much larger than the photon flight time. 
    Watch the timeline below: as each pair arrives, the electronic "Start" and "Stop" triggers appear. Because of the detector's **time resolution**, the exact trigger times fluctuate. As multiple events are recorded, you will see the Gaussian jitter distribution build up on the timeline.
    """)

    placeholder = st.empty()
    
    # Use matplotlib dark theme to ensure axes, ticks, and titles are bright white
    plt.style.use('dark_background')

    if trigger_animation:
        frames_per_event = 8  # Snappy animation per photon pair
        max_dist_frame = max(fixed_det_dist, distance_t1) + 2
        
        # Generate N independent jitters
        j1_arr = np.random.normal(0, time_resolution_sigma, num_events)
        j2_arr = np.random.normal(0, time_resolution_sigma, num_events)
        measured_dts = (t2_true + j2_arr) - (t1_true + j1_arr)
        
        st.session_state.mcb_data.extend(measured_dts)
        
        # Track past signals to build up the distribution visually
        past_j1 = []
        past_j2 = []
        
        for ev in range(num_events):
            y_offset = np.random.uniform(-0.3, 0.3)  # Slight visual vertical scatter
            
            for i in range(frames_per_event + 1):
                fig, (ax_phys, ax_time) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})
                
                # Make backgrounds transparent
                fig.patch.set_facecolor('none')
                ax_phys.set_facecolor('none')
                ax_time.set_facecolor('none')
                
                # --- Physical Space ---
                ax_phys.set_xlim(-10, 45)
                ax_phys.set_ylim(-2, 2)
                ax_phys.axis('off')
                ax_phys.set_title(f"Physical Space: Event {ev + 1} of {num_events}", color='white')
                
                ax_phys.add_patch(patches.Circle((0, 0), 0.5, color='tomato'))
                ax_phys.text(0, 0.8, "22Na", ha='center', color='tomato', weight='bold')
                ax_phys.add_patch(patches.Rectangle((-fixed_det_dist - 2, -1), 2, 2, color='cyan', alpha=0.4))
                ax_phys.text(-fixed_det_dist - 1, -1.5, "Det 1 (Fixed)", ha='center', color='cyan')
                ax_phys.add_patch(patches.Rectangle((distance_t1, -1), 2, 2, color='lime', alpha=0.4))
                ax_phys.text(distance_t1 + 1, -1.5, f"Det 2 ({distance_t1} cm)", ha='center', color='lime')
                
                progress = i / frames_per_event
                pos1 = -progress * max_dist_frame
                pos2 = progress * max_dist_frame
                
                # Draw the currently flying photons
                if pos1 > -fixed_det_dist:
                    ax_phys.scatter(pos1, y_offset, color='cyan', s=60, zorder=10)
                if pos2 < distance_t1:
                    ax_phys.scatter(pos2, y_offset, color='lime', s=60, zorder=10)
                    
                # --- Timeline ---
                ax_time.set_xlim(0, 1500)
                ax_time.set_ylim(0, 3)
                ax_time.set_yticks([1, 2])
                ax_time.set_yticklabels(["Detector 2\n(Stop)", "Detector 1\n(Start)"])
                ax_time.set_xlabel("Time (ps)")
                ax_time.set_title("Electronic Timelines (Accumulating Time Jitter)", color='white')
                
                current_time_ps = progress * (max_dist_frame / c_cm_per_ps)
                
                # Draw theoretical center lines
                ax_time.axvline(t1_true, ymin=0.55, ymax=0.95, color='white', linestyle='dotted', alpha=0.5)
                ax_time.axvline(t2_true, ymin=0.05, ymax=0.45, color='white', linestyle='dotted', alpha=0.5)
                
                # Draw past events faintly to show distribution
                for p_j1 in past_j1:
                    ax_time.axvline(t1_true + p_j1, ymin=0.55, ymax=0.95, color='cyan', alpha=0.2, lw=1)
                for p_j2 in past_j2:
                    ax_time.axvline(t2_true + p_j2, ymin=0.05, ymax=0.45, color='lime', alpha=0.2, lw=1)
                
                # Draw current triggers ONLY if the physical photons have reached the detector
                if current_time_ps >= t1_true:
                    ax_time.axvline(t1_true + j1_arr[ev], ymin=0.55, ymax=0.95, color='cyan', alpha=1.0, lw=2.5)
                if current_time_ps >= t2_true:
                    ax_time.axvline(t2_true + j2_arr[ev], ymin=0.05, ymax=0.45, color='lime', alpha=1.0, lw=2.5)

                plt.tight_layout()
                with placeholder.container():
                    st.pyplot(fig, transparent=True)
                plt.close(fig)
                
            # After an event finishes flying, save its jitter to the past list
            past_j1.append(j1_arr[ev])
            past_j2.append(j2_arr[ev])
            
    else:
        fig, (ax_phys, ax_time) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})
        fig.patch.set_facecolor('none')
        ax_phys.set_facecolor('none')
        ax_time.set_facecolor('none')
        
        ax_phys.set_xlim(-10, 45)
        ax_phys.set_ylim(-2, 2)
        ax_phys.axis('off')
        ax_phys.set_title("Physical Space: Setup Geometry", color='white')
        ax_phys.add_patch(patches.Circle((0, 0), 0.5, color='tomato'))
        ax_phys.text(0, 0.8, "22Na", ha='center', color='tomato', weight='bold')
        ax_phys.add_patch(patches.Rectangle((-fixed_det_dist - 2, -1), 2, 2, color='cyan', alpha=0.4))
        ax_phys.text(-fixed_det_dist - 1, -1.5, "Det 1 (Fixed)", ha='center', color='cyan')
        ax_phys.add_patch(patches.Rectangle((distance_t1, -1), 2, 2, color='lime', alpha=0.4))
        ax_phys.text(distance_t1 + 1, -1.5, f"Det 2 ({distance_t1} cm)", ha='center', color='lime')
        
        ax_time.set_xlim(0, 1500)
        ax_time.set_ylim(0, 3)
        ax_time.set_yticks([1, 2])
        ax_time.set_yticklabels(["Detector 2\n(Stop)", "Detector 1\n(Start)"])
        ax_time.set_xlabel("Time (ps)")
        ax_time.set_title("Electronic Timelines (Awaiting Experiment...)", color='white')
        
        plt.tight_layout()
        with placeholder.container():
            st.pyplot(fig, transparent=True)
            plt.close(fig)

    st.markdown("---")
    st.subheader("Recorded Data: Multichannel Buffer (MCB) Spectrum")

    if trigger_bulk:
        j1 = np.random.normal(0, time_resolution_sigma, 500)
        j2 = np.random.normal(0, time_resolution_sigma, 500)
        measured_dt_array = (t2_true + j2) - (t1_true + j1)
        st.session_state.mcb_data.extend(measured_dt_array)

    st.markdown(f"**Total recorded coincidence events:** {len(st.session_state.mcb_data)}")

    if len(st.session_state.mcb_data) > 0:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        fig_hist.patch.set_facecolor('none')
        ax_hist.set_facecolor('none')
        
        bins = np.linspace(-500, 2000, 100)
        ax_hist.hist(st.session_state.mcb_data, bins=bins, color='mediumorchid', alpha=0.8, edgecolor='white')
        ax_hist.axvline(true_delta_t, color='tomato', linestyle='dashed', linewidth=2, label=f"Theoretical $\Delta t$ ({true_delta_t:.0f} ps)")
        ax_hist.set_title("Time Spectrum (Mimicking the MCB)", color='white')
        ax_hist.set_xlabel("Measured Time Difference $\Delta t$ (ps)")
        ax_hist.set_ylabel("Counts")
        ax_hist.legend()
        st.pyplot(fig_hist, transparent=True)
        plt.close(fig_hist)
    else:
        st.info("Click 'Simulate Experiment' or 'Record 500 Events Rapidly' to build the spectrum.")
# --- Setup 2: Positron Lifetime Measurement (3-Component Simulation) ---
with tab2:
    st.header("Setup 2: Positron Lifetime Measurement")
    st.markdown("""
    This experiment measures how long positrons "live" inside a material before annihilating.
    We use a **22Na source**. The **Start** signal is the 1275 keV gamma emitted when 22Ne de-excites (almost simultaneous with positron creation). 
    The **Stop** signal is one of the 511 keV gamma quanta from the positron's annihilation.
    """)
    
    # State Management for Tab 2
    if 'lifetime_data' not in st.session_state:
        st.session_state.lifetime_data = []
        st.session_state.lifetime_components = []

    # Tab-specific controls
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        num_lt_events = st.slider("Number of Sequential Events (N)", min_value=1, max_value=30, value=10, step=1, key="t2_num")
        if st.button("Clear Data", key="t2_clear"):
            st.session_state.lifetime_data = []
            st.session_state.lifetime_components = []
            
    with col_ctrl2:
        st.write("Controls:")
        trigger_lt_anim = st.button("Simulate Experiment (Animation)", key="t2_anim")
        trigger_lt_bulk = st.button("Record 5000 Events Rapidly", key="t2_bulk")

    # Physics Constants for 3-Component Polymer (e.g., Plexiglas)
    time_res_sigma = 100.0  # Detector jitter (ps)
    
    # Component 1: Para-positronium (p-Ps) - Very fast
    tau1, alpha1 = 150.0, 0.20
    # Component 2: Free Positron Annihilation - Medium
    tau2, alpha2 = 400.0, 0.45
    # Component 3: Ortho-positronium (o-Ps) - Slow (Pick-off annihilation)
    tau3, alpha3 = 2000.0, 0.35
    
    components = [
        {"name": "Para-Ps", "tau": tau1, "prob": alpha1, "color": "cyan"},
        {"name": "Free", "tau": tau2, "prob": alpha2, "color": "lime"},
        {"name": "Ortho-Ps", "tau": tau3, "prob": alpha3, "color": "gold"}
    ]

    st.subheader("The Three-Component Lifetime")
    st.markdown("""
    Positrons in polymers like Plexiglas can annihilate in three different states, each with a distinct lifetime ($\tau_i$) and probability ($\\alpha_i$). 
    Watch the timeline: the "Start" trigger fires, the positron lives in the material, and eventually annihilates to fire the "Stop" trigger. The delay is colored by which state the positron entered!
    """)

    placeholder_t2 = st.empty()
    plt.style.use('dark_background')

    if trigger_lt_anim:
        # Pre-calculate N events
        states = np.random.choice([0, 1, 2], size=num_lt_events, p=[alpha1, alpha2, alpha3])
        lifetimes = [np.random.exponential(components[s]["tau"]) for s in states]
        
        j1_arr = np.random.normal(0, time_res_sigma, num_lt_events)
        j2_arr = np.random.normal(0, time_res_sigma, num_lt_events)
        
        # We assume physical flight time to detectors is constant and cancels out.
        # Delay = true lifetime + stop_jitter - start_jitter
        measured_dts = np.array(lifetimes) + j2_arr - j1_arr
        
        st.session_state.lifetime_data.extend(measured_dts)
        st.session_state.lifetime_components.extend(states)
        
        past_start = []
        past_stop = []
        past_colors = []
        
        for ev in range(num_lt_events):
            state_idx = states[ev]
            comp = components[state_idx]
            t_life = lifetimes[ev]
            
            # Animation frames: scale frames by lifetime to show waiting
            total_time_to_simulate = max(t_life + 300, 600)  # At least 600ps to see triggers
            frames = int(min(15, total_time_to_simulate / 50)) # Cap frames for speed
            
            for i in range(frames + 1):
                fig, (ax_phys, ax_time) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})
                fig.patch.set_facecolor('none')
                ax_phys.set_facecolor('none')
                ax_time.set_facecolor('none')
                
                # --- Physical Space ---
                ax_phys.set_xlim(-15, 15)
                ax_phys.set_ylim(-2, 2)
                ax_phys.axis('off')
                ax_phys.set_title(f"Event {ev + 1}/{num_lt_events}: {comp['name']} State ($\tau$={comp['tau']:.0f}ps)", color=comp['color'])
                
                # Detectors & Source
                ax_phys.add_patch(patches.Rectangle((-10, -1), 2, 2, color='mediumorchid', alpha=0.4))
                ax_phys.text(-9, -1.5, "Det 1 (Start)\n1275 keV", ha='center', color='mediumorchid')
                
                ax_phys.add_patch(patches.Rectangle((8, -1), 2, 2, color='white', alpha=0.4))
                ax_phys.text(9, -1.5, "Det 2 (Stop)\n511 keV", ha='center', color='white')
                
                # Sample block
                ax_phys.add_patch(patches.Rectangle((-1, -1), 2, 2, color='gray', alpha=0.5))
                ax_phys.text(0, 1.2, "Plexiglas Sample", ha='center', color='white')
                
                current_time_ps = (i / frames) * total_time_to_simulate
                
                # 1. Start Photon Flight (instantaneous for simplicity, just flashes)
                if current_time_ps < 100:
                    ax_phys.plot([-1, -9], [0, 0], color='mediumorchid', lw=3, linestyle='--')
                
                # 2. Positron living in sample
                if 0 <= current_time_ps < t_life:
                    ax_phys.scatter([0], [0], color=comp['color'], s=150, zorder=10) # Glowing positron
                    
                # 3. Annihilation / Stop Photon Flight
                if current_time_ps >= t_life:
                    ax_phys.plot([1, 9], [0, 0], color=comp['color'], lw=3, linestyle='--')
                    ax_phys.scatter([0], [0], color='white', marker='x', s=100) # Annihilation flash
                    
                # --- Timeline ---
                ax_time.set_xlim(-200, 3000)
                ax_time.set_ylim(0, 3)
                ax_time.set_yticks([1, 2])
                ax_time.set_yticklabels(["Detector 2\n(Stop)", "Detector 1\n(Start)"])
                ax_time.set_xlabel("Time (ps)")
                ax_time.set_title("Electronic Timelines (Building the Decay Curve)", color='white')
                
                # Draw past triggers
                for p_s, p_e, p_c in zip(past_start, past_stop, past_colors):
                    ax_time.axvline(p_s, ymin=0.55, ymax=0.95, color='mediumorchid', alpha=0.2, lw=1)
                    ax_time.axvline(p_e, ymin=0.05, ymax=0.45, color=p_c, alpha=0.2, lw=1)
                    # Connection line showing dt
                    ax_time.plot([p_s, p_e], [2, 1], color=p_c, alpha=0.1, lw=1)
                
                # Draw current triggers based on time
                t_start_elec = j1_arr[ev]
                t_stop_elec = t_life + j2_arr[ev]
                
                if current_time_ps >= 0: # Start triggers around t=0
                    ax_time.axvline(t_start_elec, ymin=0.55, ymax=0.95, color='mediumorchid', alpha=1.0, lw=2.5)
                if current_time_ps >= t_life:
                    ax_time.axvline(t_stop_elec, ymin=0.05, ymax=0.45, color=comp['color'], alpha=1.0, lw=2.5)
                    ax_time.plot([t_start_elec, t_stop_elec], [2, 1], color=comp['color'], alpha=0.8, lw=2, linestyle=':')

                plt.tight_layout()
                with placeholder_t2.container():
                    st.pyplot(fig, transparent=True)
                plt.close(fig)
                
            past_start.append(t_start_elec)
            past_stop.append(t_stop_elec)
            past_colors.append(comp['color'])
            
    else:
        fig, (ax_phys, ax_time) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1]})
        fig.patch.set_facecolor('none')
        ax_phys.set_facecolor('none')
        ax_time.set_facecolor('none')
        
        ax_phys.set_xlim(-15, 15)
        ax_phys.set_ylim(-2, 2)
        ax_phys.axis('off')
        ax_phys.set_title("Physical Space: Setup Geometry", color='white')
        
        ax_phys.add_patch(patches.Rectangle((-10, -1), 2, 2, color='mediumorchid', alpha=0.4))
        ax_phys.text(-9, -1.5, "Det 1 (Start)\n1275 keV", ha='center', color='mediumorchid')
        ax_phys.add_patch(patches.Rectangle((8, -1), 2, 2, color='white', alpha=0.4))
        ax_phys.text(9, -1.5, "Det 2 (Stop)\n511 keV", ha='center', color='white')
        ax_phys.add_patch(patches.Rectangle((-1, -1), 2, 2, color='gray', alpha=0.5))
        ax_phys.text(0, 1.2, "Plexiglas Sample", ha='center', color='white')
        
        ax_time.set_xlim(-200, 3000)
        ax_time.set_ylim(0, 3)
        ax_time.set_yticks([1, 2])
        ax_time.set_yticklabels(["Detector 2\n(Stop)", "Detector 1\n(Start)"])
        ax_time.set_xlabel("Time (ps)")
        ax_time.set_title("Electronic Timelines (Awaiting Experiment...)", color='white')
        
        plt.tight_layout()
        with placeholder_t2.container():
            st.pyplot(fig, transparent=True)
            plt.close(fig)

    st.markdown("---")
    st.subheader("Measured Lifetime Spectrum (Log Scale)")

    if trigger_lt_bulk:
        bulk_n = 5000
        states = np.random.choice([0, 1, 2], size=bulk_n, p=[alpha1, alpha2, alpha3])
        lifetimes = [np.random.exponential(components[s]["tau"]) for s in states]
        j1 = np.random.normal(0, time_res_sigma, bulk_n)
        j2 = np.random.normal(0, time_res_sigma, bulk_n)
        measured_dts = np.array(lifetimes) + j2 - j1
        
        st.session_state.lifetime_data.extend(measured_dts)
        st.session_state.lifetime_components.extend(states)

    st.markdown(f"**Total recorded decay events:** {len(st.session_state.lifetime_data)}")

    if len(st.session_state.lifetime_data) > 0:
        fig_hist2, ax_hist2 = plt.subplots(figsize=(10, 5))
        fig_hist2.patch.set_facecolor('none')
        ax_hist2.set_facecolor('none')
        
        # Calculate theoretical curves via convolution
        t_theory = np.linspace(-500, 5000, 1000)
        dt_bin = t_theory[1] - t_theory[0]
        gaussian_kernel = np.exp(-(t_theory**2)/(2*time_res_sigma**2)) / (np.sqrt(2*np.pi)*time_res_sigma)
        
        # Ideal exponential decays
        decay1 = np.where(t_theory > 0, (alpha1/tau1) * np.exp(-t_theory/tau1), 0)
        decay2 = np.where(t_theory > 0, (alpha2/tau2) * np.exp(-t_theory/tau2), 0)
        decay3 = np.where(t_theory > 0, (alpha3/tau3) * np.exp(-t_theory/tau3), 0)
        
        # Convoluted with detector resolution
        conv1 = np.convolve(decay1, gaussian_kernel, mode='same') * dt_bin
        conv2 = np.convolve(decay2, gaussian_kernel, mode='same') * dt_bin
        conv3 = np.convolve(decay3, gaussian_kernel, mode='same') * dt_bin
        conv_total = conv1 + conv2 + conv3
        
        # Plot Histogram
        bins = np.linspace(-500, 5000, 120)
        counts, bin_edges = np.histogram(st.session_state.lifetime_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Scale theoretical curves to match histogram area
        area = len(st.session_state.lifetime_data) * (bins[1] - bins[0])
        
        ax_hist2.step(bin_centers, counts, where='mid', color='white', alpha=0.7, label="Measured Data")
        
        # Plot theory lines
        ax_hist2.plot(t_theory, conv1 * area, color='cyan', linestyle='--', label=f"Para-Ps ($\\tau$={tau1}ps)")
        ax_hist2.plot(t_theory, conv2 * area, color='lime', linestyle='--', label=f"Free ($\\tau$={tau2}ps)")
        ax_hist2.plot(t_theory, conv3 * area, color='gold', linestyle='--', label=f"Ortho-Ps ($\\tau$={tau3}ps)")
        ax_hist2.plot(t_theory, conv_total * area, color='red', lw=2, label="Total Theoretical Sum")
        
        ax_hist2.set_yscale('log')
        ax_hist2.set_ylim(0.5, max(counts) * 2)
        ax_hist2.set_xlim(-500, 5000)
        
        ax_hist2.set_title("Positron Lifetime Spectrum (Logarithmic Scale)", color='white')
        ax_hist2.set_xlabel("Time Difference $t$ (ps)")
        ax_hist2.set_ylabel("Counts")
        ax_hist2.legend(loc='upper right')
        
        st.pyplot(fig_hist2, transparent=True)
        plt.close(fig_hist2)
    else:
        st.info("Click 'Simulate Experiment' or 'Record 5000 Events Rapidly' to build the 3-component decay spectrum.")

# --- Setup 3: Angular Efficiency ---
with tab3:
    st.header("Setup 3: Angular Efficiency Calibration")
    st.markdown("Uses **NaI(Tl) detectors** (good energy resolution, poor time resolution).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: Collimator & Rotation")
        angle_calib = st.slider("Calib. Angle", min_value=145, max_value=180, value=180, step=5, key="t3_angle")
        st.markdown("Detector 1 is moved back and collimated to simulate a point source. Detector 2 rotates from 180° to 145° in 25° steps.")
        
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax3, "22Na Source", "Det 1 (Point)\n511 keV", "Det 2 (Rotated)\n511 keV", det2_pos='rotate', value=angle_calib, has_collimator=True)
        st.pyplot(fig3)

    with col2:
        st.subheader("Principle: Solid Angle Detection")
        st.markdown(r"Real detectors cover a finite solid angle ($\Delta\Omega > 0$). This step determines the angular detection efficiency $\epsilon(\vartheta')$.")
        
        theta_eff = np.linspace(140, 180, 100)
        efficiency = np.exp(-((theta_eff - 180)**2) / 50)  # Mock efficiency curve
        
        fig3b, ax3b = plt.subplots()
        ax3b.plot(theta_eff, efficiency, 'b-')
        ax3b.axvline(angle_calib, color='orange', linestyle='--', label="Current Position")
        ax3b.set_xlabel("Angle $\\vartheta'$ (Degrees)")
        ax3b.set_ylabel("Detection Efficiency $\\epsilon(\\vartheta')$")
        ax3b.legend()
        st.pyplot(fig3b)

# --- Setup 4: Angular Correlation ---
with tab4:
    st.header("Setup 4: Gamma-Gamma Angular Correlation")
    st.markdown("NaI(Tl) detectors measure the anisotropic emission of the 4-2-0 cascade from 60Ni.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: 60Co Measurement")
        angle_meas = st.slider("Measurement Angle", min_value=90, max_value=180, value=180, step=15, key="t4_angle")
        st.markdown("The detector rotates from 180° to 90° in 15° steps.")
        
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax4, "60Co Source", "Det 1\n>2800 ch", "Det 2\n>2800 ch", det2_pos='rotate', value=angle_meas)
        st.pyplot(fig4)

    with col2:
        st.subheader("Principle: Smeared Anisotropy")
        st.markdown(r"The true theoretical angular distribution $W(\vartheta)$ is smoothed out into the effective correlation function $\tilde{W}(\vartheta)$ due to the detection efficiency measured in Setup 3.")
        st.latex(r"\tilde{W}(\vartheta) = \frac{\int \epsilon(\vartheta') \cdot W(\vartheta, \vartheta', \varphi') d\Omega'}{\int \epsilon(\vartheta') d\Omega'}")
        
        theta_corr = np.linspace(90, 180, 100)
        W_theta = 1 + 0.125 * np.cos(np.radians(theta_corr))**2 + 0.041 * np.cos(np.radians(theta_corr))**4
        W_eff = 1 + 0.09 * np.cos(np.radians(theta_corr))**2 + 0.02 * np.cos(np.radians(theta_corr))**4
        
        fig4b, ax4b = plt.subplots()
        ax4b.plot(theta_corr, W_theta, 'k--', label="Theoretical $W(\\vartheta)$")
        ax4b.plot(theta_corr, W_eff, 'r-', label="Effective $\\tilde{W}(\\vartheta)$")
        
        current_W = 1 + 0.09 * np.cos(np.radians(angle_meas))**2 + 0.02 * np.cos(np.radians(angle_meas))**4
        ax4b.plot(angle_meas, current_W, 'bo', markersize=8, label="Current Setup")
        
        ax4b.set_xlabel("Angle $\\vartheta$ (Degrees)")
        ax4b.set_ylabel("Correlation $W(\\vartheta)$")
        ax4b.legend()
        st.pyplot(fig4b)
