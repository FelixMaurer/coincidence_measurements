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
# --- Setup 2: Positron Lifetime ---
with tab2:
    st.header("Setup 2: Positron Lifetime Measurement")
    st.markdown("Both BaF2 detectors are now kept at a fixed distance.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: Fixed Position")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax2, "22Na Source", "Det 1 (Start)\n1275 keV", "Det 2 (Stop)\n511 keV", det2_pos='shift', value=6)
        st.pyplot(fig2)
        st.markdown("The start signal is the 1275 keV emission from 22Ne, and the stop signal is the 511 keV annihilation quantum.")

    with col2:
        st.subheader("Principle: Convoluted Decay")
        st.markdown(r"The true lifetime decay $e^{-t/\tau}$ is convoluted with the detector's Gaussian time resolution.")
        st.latex(r"N(t) \propto \sum \alpha_i e^{-t/\tau_i} \ast P_{Det}(t)")
        
        t2 = np.linspace(-5, 20, 400)
        dt2 = 1.5
        gaussian2 = (1 / (np.sqrt(2 * np.pi) * dt2)) * np.exp(-(t2**2) / (2 * dt2**2))
        tau = 3.0
        decay = np.where(t2 > 0, np.exp(-t2/tau), 0)
        convoluted = np.convolve(decay, gaussian2, mode='same') / sum(gaussian2)
        
        fig2b, ax2b = plt.subplots()
        ax2b.plot(t2, decay, 'k--', label="Theoretical Decay")
        ax2b.plot(t2, convoluted, 'r-', label="Measured Spectrum (Convoluted)")
        ax2b.set_xlabel("Time (ps)")
        ax2b.set_yscale('log')
        ax2b.set_ylim(1e-3, 1.5)
        ax2b.legend()
        st.pyplot(fig2b)

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
