import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

st.set_page_config(page_title="Nuclear Coincidence Experiments", layout="wide")

st.title("Experimental Methods: Coincidence Measurements")
st.markdown("""
This application visualizes the two primary experiments utilizing coincidence measurement: **Positron Lifetime** and **Gamma-Gamma Angular Correlation**.
""")

# --- Sidebar Controls ---
st.sidebar.header("Experiment Settings")
experiment_type = st.sidebar.radio(
    "Select Experiment Type:",
    ("Time Resolution (Positron Lifetime)", "Angle Resolution (Angular Correlation)")
)

st.sidebar.markdown("---")

# --- Interactive Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detector Setup")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw Central Source
    source = patches.Circle((0, 0), 1, color='red', zorder=5)
    ax.add_patch(source)
    ax.text(0, 1.5, "Radioactive\nSource", ha='center', fontsize=10)

    # Fixed Detector 1 (Left)
    det1 = patches.Rectangle((-10, -2), 4, 4, color='blue', alpha=0.6)
    ax.add_patch(det1)
    ax.text(-8, -3.5, "Detector 1\n(Start / Fixed)", ha='center')

    # Moveable Detector 2 setup based on experiment
    if experiment_type == "Time Resolution (Positron Lifetime)":
        distance = st.sidebar.slider("Detector 2 Distance (cm)", min_value=4.0, max_value=20.0, value=10.0, step=1.0)
        
        # Detector 2 shifts linearly
        det2 = patches.Rectangle((distance, -2), 4, 4, color='green', alpha=0.6)
        ax.add_patch(det2)
        ax.text(distance + 2, -3.5, "Detector 2\n(Stop / Shifted)", ha='center')
        
        # Draw path
        ax.plot([1, distance], [0, 0], 'k--', alpha=0.5)
        ax.plot([-1, -6], [0, 0], 'k--', alpha=0.5)

    else:
        angle_deg = st.sidebar.slider("Detector 2 Angle (Degrees)", min_value=90, max_value=180, value=180, step=5)
        
        # Detector 2 rotates
        radius = 8
        angle_rad = np.radians(angle_deg)
        x_center = radius * np.cos(angle_rad)
        y_center = radius * np.sin(angle_rad)
        
        # Create a rectangle and rotate it
        det2 = patches.Rectangle((x_center - 2, y_center - 2), 4, 4, color='orange', alpha=0.6)
        t = transforms.Affine2D().rotate_deg_around(x_center, y_center, angle_deg) + ax.transData
        det2.set_transform(t)
        ax.add_patch(det2)
        
        # Draw path
        ax.plot([0, x_center], [0, y_center], 'k--', alpha=0.5)
        ax.plot([0, -radius], [0, 0], 'k--', alpha=0.5)
        
        # Draw angle arc
        arc = patches.Arc((0, 0), 4, 4, theta1=angle_deg, theta2=180, color='black')
        ax.add_patch(arc)
        ax.text(0, 3, f"{angle_deg}°", ha='center', fontsize=12)

    st.pyplot(fig)

with col2:
    if experiment_type == "Time Resolution (Positron Lifetime)":
        st.subheader("Limited Time Resolution")
        st.markdown(r"""
        Because the detectors and electronics cannot trigger infinitely fast, they have a limited time resolution $\Delta t$. 
        Instead of a perfect sharp spike for the start and stop time, the detector produces a Gaussian probability kernel:
        """)
        st.latex(r"P_{Det}(t') = \frac{1}{\sqrt{2\pi}\Delta t} \cdot e^{-\frac{t'^2}{2\Delta t^2}}")
        st.markdown("The further away the detector, the more the time delay changes (time-of-flight). Try shifting the detector distance to see the signal delay.")

        # Plot Gaussian Kernel
        t = np.linspace(-10, 30, 400)
        time_delay = distance * 0.5  # arbitrary scale for visualization
        dt = 2.0  # constant time resolution
        
        gaussian = (1 / (np.sqrt(2 * np.pi) * dt)) * np.exp(-((t - time_delay)**2) / (2 * dt**2))
        
        fig2, ax2 = plt.subplots()
        ax2.plot(t, gaussian, color='green')
        ax2.fill_between(t, gaussian, color='green', alpha=0.3)
        ax2.set_xlabel("Time (ps)")
        ax2.set_ylabel("Detection Probability")
        ax2.set_title("Gaussian Time Kernel Shifting with Distance")
        ax2.axvline(0, color='blue', linestyle='--', label="Start Signal (Det 1)")
        ax2.legend()
        st.pyplot(fig2)

    else:
        st.subheader("Limited Angular Resolution")
        st.markdown(r"""
        Because the detectors cover a finite solid angle ($\Delta\Omega > 0$), they record a smeared out average of the true angular correlation.
        This modifies the theoretical distribution $W(\vartheta)$ by the detection efficiency $\epsilon(\vartheta')$.
        """)
        st.latex(r"\tilde{W}(\vartheta) = \frac{\int \epsilon(\vartheta') \cdot W(\vartheta, \vartheta', \varphi') d\Omega'}{\int \epsilon(\vartheta') d\Omega'}")
        
        # Plot Angular Correlation vs Effective
        theta = np.linspace(90, 180, 100)
        # Theoretical 60Co curve simplified: ~ 1 + 0.125 * cos^2(theta) + 0.041 * cos^4(theta)
        W_theta = 1 + 0.125 * np.cos(np.radians(theta))**2 + 0.041 * np.cos(np.radians(theta))**4
        
        # Simulated effective (smoothed) curve
        W_eff = 1 + 0.09 * np.cos(np.radians(theta))**2 + 0.02 * np.cos(np.radians(theta))**4

        fig3, ax3 = plt.subplots()
        ax3.plot(theta, W_theta, 'k--', label="Theoretical $W(\\vartheta)$")
        ax3.plot(theta, W_eff, 'r-', label="Effective $\\tilde{W}(\\vartheta)$ (Finite Angle)")
        
        # Point indicating current slider position
        current_W = 1 + 0.09 * np.cos(np.radians(angle_deg))**2 + 0.02 * np.cos(np.radians(angle_deg))**4
        ax3.plot(angle_deg, current_W, 'bo', markersize=8, label="Current Setup")
        
        ax3.set_xlabel("Angle $\\vartheta$ (Degrees)")
        ax3.set_ylabel("Coincidence Rate / W($\\vartheta$)")
        ax3.set_title("Angular Correlation Curve")
        ax3.legend()
        st.pyplot(fig3)
