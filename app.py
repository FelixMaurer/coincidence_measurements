import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

st.set_page_config(page_title="Coincidence Experiments", layout="wide")

st.title("Experimental Methods: Gamma and Positron Coincidence")
st.markdown("""
This application outlines the four distinct experimental setups required to measure positron lifetimes and gamma-gamma angular correlations. 
The experiments rely on coincidence measurements, where two related nuclear events are detected simultaneously (or with a measurable delay).
""")

# --- Helper Functions for Drawing ---
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
        # Linear shift
        distance = value
        det2 = patches.Rectangle((distance, -2), 4, 4, color='green', alpha=0.6)
        ax.add_patch(det2)
        ax.text(distance + 2, -3.5, det2_label, ha='center')
        ax.plot([1, distance], [0, 0], 'k--', alpha=0.5)
        ax.plot([-1, -6], [0, 0], 'k--', alpha=0.5)
    elif det2_pos == 'rotate':
        # Rotation
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

# --- Setup 1: Time Calibration ---
with tab1:
    st.header("Setup 1: Time Calibration & Resolution")
    st.markdown("Uses **BaF2 detectors** (good time resolution, poor energy resolution)[cite: 80].")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: Distance Variation")
        distance = st.slider("Shift Detector 2 (cm)", min_value=4, max_value=40, value=4, step=6)
        st.markdown("Detector 2 is shifted from 4 cm to 40 cm in 6 cm steps to use the time-of-flight difference for time calibration[cite: 374, 375].")
        
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax1, "22Na Source", "Det 1 (Start)\n511 keV", "Det 2 (Stop)\n511 keV", det2_pos='shift', value=distance)
        st.pyplot(fig1)

    with col2:
        st.subheader("Principle: Time-of-Flight & Resolution Kernel")
        st.markdown(r"Due to finite time resolution $\Delta t$, the detectors yield a Gaussian probability distribution[cite: 150, 151]:")
        st.latex(r"P_{Det}(t') = \frac{1}{\sqrt{2\pi}\Delta t} \cdot e^{-\frac{t'^2}{2\Delta t^2}}")
        st.markdown("Changing the distance shifts the mean detection time due to the speed of light.")
        
        t = np.linspace(-5, 20, 400)
        dt = 1.5
        time_delay = distance * 0.33  # Approximation for visualizing speed of light delay
        gaussian = (1 / (np.sqrt(2 * np.pi) * dt)) * np.exp(-((t - time_delay)**2) / (2 * dt**2))
        
        fig1b, ax1b = plt.subplots()
        ax1b.plot(t, gaussian, color='green')
        ax1b.fill_between(t, gaussian, color='green', alpha=0.3)
        ax1b.axvline(0, color='blue', linestyle='--', label="Start Signal (t=0)")
        ax1b.set_xlabel("Time (ps)")
        ax1b.set_yticks([])
        ax1b.legend()
        st.pyplot(fig1b)

# --- Setup 2: Positron Lifetime ---
with tab2:
    st.header("Setup 2: Positron Lifetime Measurement")
    st.markdown("Both BaF2 detectors are now kept at a fixed distance[cite: 389, 391].")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: Fixed Position")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax2, "22Na Source", "Det 1 (Start)\n1275 keV", "Det 2 (Stop)\n511 keV", det2_pos='shift', value=6)
        st.pyplot(fig2)
        st.markdown("The start signal is the 1275 keV emission from 22Ne, and the stop signal is the 511 keV annihilation quantum[cite: 116, 124, 125, 389].")

    with col2:
        st.subheader("Principle: Convoluted Decay")
        st.markdown(r"The true lifetime decay $e^{-t/\tau}$ is convoluted with the detector's Gaussian time resolution[cite: 153, 154].")
        st.latex(r"N(t) \propto \sum \alpha_i e^{-t/\tau_i} \ast P_{Det}(t)")
        
        t2 = np.linspace(-5, 20, 400)
        tau = 3.0
        decay = np.where(t2 > 0, np.exp(-t2/tau), 0)
        convoluted = np.convolve(decay, gaussian, mode='same') / sum(gaussian)
        
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
    st.markdown("Uses **NaI(Tl) detectors** (good energy resolution, poor time resolution)[cite: 81].")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: Collimator & Rotation")
        angle_calib = st.slider("Calib. Angle", min_value=145, max_value=180, value=180, step=5)
        st.markdown("Detector 1 is moved back and collimated to simulate a point source. Detector 2 rotates from 180° to 145° in 25° steps[cite: 382].")
        
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax3, "22Na Source", "Det 1 (Point)\n511 keV", "Det 2 (Rotated)\n511 keV", det2_pos='rotate', value=angle_calib, has_collimator=True)
        st.pyplot(fig3)

    with col2:
        st.subheader("Principle: Solid Angle Detection")
        st.markdown(r"Real detectors cover a finite solid angle ($\Delta\Omega > 0$). This step determines the angular detection efficiency $\epsilon(\vartheta')$[cite: 316, 318, 382].")
        
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
    st.markdown("NaI(Tl) detectors measure the anisotropic emission of the 4-2-0 cascade from 60Ni[cite: 242, 243, 246, 404].")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Geometry: 60Co Measurement")
        angle_meas = st.slider("Measurement Angle", min_value=90, max_value=180, value=180, step=15)
        st.markdown("The detector rotates from 180° to 90° in 15° steps[cite: 404].")
        
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        draw_detector_setup(ax4, "60Co Source", "Det 1\n>2800 ch", "Det 2\n>2800 ch", det2_pos='rotate', value=angle_meas)
        st.pyplot(fig4)

    with col2:
        st.subheader("Principle: Smeared Anisotropy")
        st.markdown(r"The true theoretical angular distribution $W(\vartheta)$ is smoothed out into the effective correlation function $\tilde{W}(\vartheta)$ due to the detection efficiency measured in Setup 3[cite: 300, 324, 325].")
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
