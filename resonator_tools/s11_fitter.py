"""
Interactive S11 resonance fitter (1-port reflection).

Model (Probst et al., Eq. 1):
    S11 = a * exp(i*alpha) * exp(-2*pi*i*f*tau) * [1 - (Ql/|Qc|)*exp(i*phi0) / (1 + 2i*Ql*x)]
    x = (f - f0) / f0
"""
import sys
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better window management on macOS
import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler  # Enables Ctrl+C / Cmd+C to copy figure to clipboard
from matplotlib.widgets import Slider, Button
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d
from scipy.linalg import svd
from io import StringIO
# ==============================================================================
# Data loading
# ==============================================================================

def _normalize_header_name(name):
    name = name.strip().lower()
    for ch in ("(", ")", "[", "]", "{", "}", " "):
        name = name.replace(ch, "")
    return name

def load_s11_csv(path, delimiter=","):
    """
    Load S11 CSV data with headers and comments.

    Expected columns: frequency and Re/Im of S11. We trust Re/Im even if
    magnitude/phase columns are present. If the header is missing, we fall back
    to common column positions.
    """
    with open(path, "r") as f:
        raw_lines = f.readlines()

    lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("No data lines found in file.")

    first = lines[0].strip()
    tokens = [t.strip() for t in first.split(delimiter)]
    has_header = any(any(ch.isalpha() for ch in t) for t in tokens)

    header = None
    data_lines = lines
    if has_header:
        header = [_normalize_header_name(t) for t in tokens]
        data_lines = lines[1:]

    data = np.genfromtxt(StringIO("".join(data_lines)), delimiter=delimiter)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if header:
        name_to_idx = {name: i for i, name in enumerate(header)}
        # Trust Re/Im even if magnitude/phase columns are present.
        freq_idx = None
        for key in ("f", "freq", "frequency", "hz"):
            if key in name_to_idx:
                freq_idx = name_to_idx[key]
                break
        if freq_idx is None:
            for name, idx in name_to_idx.items():
                if "freq" in name:
                    freq_idx = idx
                    break

        # Re/Im
        re_idx = None
        im_idx = None
        for key in ("re", "real", "realpart"):
            if key in name_to_idx:
                re_idx = name_to_idx[key]
                break
        for key in ("im", "imag", "imaginary", "imagpart"):
            if key in name_to_idx:
                im_idx = name_to_idx[key]
                break
        # Handle common labels like "re(s11)" or "im(s11)".
        if re_idx is None:
            for name, idx in name_to_idx.items():
                if name.startswith("re"):
                    re_idx = idx
                    break
        if im_idx is None:
            for name, idx in name_to_idx.items():
                if name.startswith("im"):
                    im_idx = idx
                    break

        if freq_idx is None:
            raise ValueError(f"Could not find frequency column in header: {header}")

        # If Re/Im not found, fall back to mag(dB) + phase(deg) if available.
        if re_idx is None or im_idx is None:
            mag_idx = None
            deg_idx = None

            # Common labels: "s11", "s21" (dB magnitude), "deg" (phase in degrees)
            for key in ("s11", "s21", "mag", "magnitude", "db"):
                if key in name_to_idx:
                    mag_idx = name_to_idx[key]
                    break
            for key in ("deg", "phase", "phasedeg", "ang"):
                if key in name_to_idx:
                    deg_idx = name_to_idx[key]
                    break

            if mag_idx is None or deg_idx is None:
                raise ValueError(
                    f"Could not find Re/Im OR (mag_dB + phase_deg) columns in header: {header}"
                )

            freq = data[:, freq_idx]
            mag_db = data[:, mag_idx]
            phase_deg = data[:, deg_idx]
            mag_lin = 10 ** (mag_db / 20.0)
            phase_rad = np.deg2rad(phase_deg)
            S11_complex = mag_lin * np.exp(1j * phase_rad)

            # Drop NaN rows
            good = np.isfinite(freq) & np.isfinite(mag_db) & np.isfinite(phase_deg)
            if not np.all(good):
                dropped = np.count_nonzero(~good)
                print(f"Warning: dropped {dropped} rows with NaNs.")
                freq = freq[good]
                S11_complex = S11_complex[good]

            # Sort by frequency
            if np.any(np.diff(freq) < 0):
                order = np.argsort(freq)
                freq = freq[order]
                S11_complex = S11_complex[order]

            return freq, S11_complex
    else:
        # No header: assume common column layouts.
        if data.shape[1] >= 5:
            freq_idx, re_idx, im_idx = 0, 3, 4
        elif data.shape[1] >= 3:
            freq_idx, re_idx, im_idx = 0, 1, 2
            print("Warning: no header found; assuming columns [f, Re, Im].")
        else:
            raise ValueError("Data format error: expected at least 3 columns.")

    freq = data[:, freq_idx]
    re = data[:, re_idx]
    im = data[:, im_idx]

    good = np.isfinite(freq) & np.isfinite(re) & np.isfinite(im)
    if not np.all(good):
        dropped = np.count_nonzero(~good)
        print(f"Warning: dropped {dropped} rows with NaNs in freq/Re/Im.")
        freq = freq[good]
        re = re[good]
        im = im[good]

    S11_complex = re + 1j * im

    # Sorting keeps the optimizer sane if the sweep is reversed.
    if np.any(np.diff(freq) < 0):
        order = np.argsort(freq)
        freq = freq[order]
        S11_complex = S11_complex[order]

    return freq, S11_complex

# ==============================================================================
# Cable delay (UNUSED)
# ==============================================================================

# def estimate_cable_delay(freq, S11, n_edge=None):
#     """Estimate cable delay from phase slope at edges."""
#     if n_edge is None:
#         n_edge = max(20, len(freq) // 10)
    
#     phase = np.unwrap(np.angle(S11))
#     edge_idx = np.concatenate([np.arange(n_edge), np.arange(len(freq) - n_edge, len(freq))])
    
#     coeffs = np.polyfit(freq[edge_idx], phase[edge_idx], 1)
#     tau = -coeffs[0] / (2 * np.pi)
#     alpha_offset = coeffs[1]
    
#     return tau, alpha_offset

# def remove_cable_delay(freq, S11, tau):
#     """Remove cable delay from S11."""
#     return S11 * np.exp(2j * np.pi * freq * tau)

# ==============================================================================
# Background Calibration (complex polynomial baseline)
# ==============================================================================

def make_off_res_mask(freq, f0, Ql, exclude_n_lw=6.0, min_off_pts=80):
    """Mask that excludes ±exclude_n_lw * FWHM around resonance."""
    fwhm = f0 / max(Ql, 100.0)
    mask = np.abs(freq - f0) > exclude_n_lw * fwhm
    if np.count_nonzero(mask) < min_off_pts:
        # Fallback: edges only
        edge = max(20, len(freq) // 10)
        mask = np.zeros_like(freq, dtype=bool)
        mask[:edge] = True
        mask[-edge:] = True
    return mask

def estimate_delay_from_mask(freq, S, mask):
    """Estimate τ from unwrapped phase slope on off-res points."""
    phase = np.unwrap(np.angle(S[mask]))
    p = np.polyfit(freq[mask], phase, 1)  # phase ≈ m*f + b
    tau = -p[0] / (2 * np.pi)
    return tau

def fit_complex_poly_baseline(freq, S_delay_removed, mask, order=2):
    """
    Fit complex polynomial baseline B(f) = Σ c_k * x^k.
    
    Uses scaled variable x = (f - fc)/scale to avoid ill-conditioning
    (frequencies are GHz, and powers explode otherwise).
    """
    fc = np.median(freq)
    x = freq - fc
    scale = np.max(np.abs(x))
    if scale == 0:
        scale = 1.0
    xs = x / scale

    X = np.vstack([xs**k for k in range(order + 1)]).T  # (N, order+1)
    coeffs = np.linalg.lstsq(X[mask], S_delay_removed[mask], rcond=None)[0]
    return fc, scale, coeffs

def eval_complex_poly(freq, fc, scale, coeffs):
    """Evaluate complex polynomial baseline at given frequencies."""
    xs = (freq - fc) / scale
    out = np.zeros_like(freq, dtype=np.complex128)
    for k, ck in enumerate(coeffs):
        out += ck * (xs ** k)
    return out

def calibrate_background(freq, S, f0, Ql, order=2, exclude_n_lw=6.0):
    """
    Calibrate frequency-dependent background (delay + complex polynomial).
    """
    mask_off = make_off_res_mask(freq, f0, Ql, exclude_n_lw=exclude_n_lw)
    tau_bg = estimate_delay_from_mask(freq, S, mask_off)
    S_dt = S * np.exp(2j * np.pi * freq * tau_bg)  # remove delay
    fc, scale, coeffs = fit_complex_poly_baseline(freq, S_dt, mask_off, order=order)
    return {
        "tau_bg": tau_bg,
        "fc": fc,
        "scale": scale,
        "coeffs": coeffs,
        "order": order,
        "mask_off": mask_off,
    }

def eval_background_factor(freq, bg, tau_res=0.0, a=1.0, alpha=0.0):
    """
    Full background multiplier in raw domain.
    """
    B = eval_complex_poly(freq, bg["fc"], bg["scale"], bg["coeffs"])
    return (a * np.exp(1j * alpha)) * np.exp(-2j * np.pi * freq * (bg["tau_bg"] + tau_res)) * B

# ==============================================================================
# Initial guesses
# ==============================================================================

def estimate_baseline(freq, S11, f0_est, Ql_est):
    """Estimate baseline a and alpha from off-resonance region."""
    fwhm = f0_est / max(Ql_est, 100)
    off_res = np.abs(freq - f0_est) > 3 * fwhm
    
    if np.sum(off_res) < 10:
        n_edge = max(10, len(freq) // 10)
        off_res = np.zeros(len(freq), dtype=bool)
        off_res[:n_edge] = True
        off_res[-n_edge:] = True
    
    S11_off = S11[off_res]
    a = np.median(np.abs(S11_off))
    alpha = np.median(np.angle(S11_off))
    
    return a, alpha

def find_resonance(freq, S11_dB):
    """Find resonance frequency and estimate Ql from linewidth."""
    # Light smoothing makes the dip pick stable on noisy traces.
    S11_smooth = uniform_filter1d(S11_dB, size=max(3, len(freq)//100))
    
    min_idx = np.argmin(S11_smooth)
    f0_est = freq[min_idx]
    
    # Crude linewidth-based Ql estimate
    S11_min = S11_smooth[min_idx]
    S11_max = np.max(S11_smooth)
    half_depth = (S11_max + S11_min) / 2
    
    above_half = S11_smooth > half_depth
    crossings = np.where(np.diff(above_half.astype(int)))[0]
    
    if len(crossings) >= 2:
        f_low = freq[crossings[0]]
        f_high = freq[crossings[-1]]
        fwhm = f_high - f_low
        Ql_est = f0_est / fwhm if fwhm > 0 else 5e3
    else:
        Ql_est = 5e3
    
    dip_depth = S11_max - S11_min
    
    return f0_est, max(100, min(Ql_est, 1e7)), dip_depth

def estimate_Qc(dip_depth_dB, Ql_est):
    """Estimate |Qc| from dip depth.
    
    At resonance for reflection: |S11|^2 = |1 - Ql/Qc|^2
    Dip depth ≈ 20*log10(1 - Ql/Qc) for undercoupled
    """
    # For shallow dips (undercoupled): |S11_min| ≈ 1 - Ql/Qc
    # dip_depth_dB = -20*log10(|S11_min|) ≈ 20*log10(1/(1-Ql/Qc))
    
    if dip_depth_dB > 0.1:
        # |S11_min| in linear
        S11_min_lin = 10**(-dip_depth_dB / 20)
        # Ql/Qc = 1 - |S11_min| (undercoupled case)
        Ql_over_Qc = 1 - S11_min_lin
        if Ql_over_Qc > 0.01:
            Qc_est = Ql_est / Ql_over_Qc
        else:
            Qc_est = Ql_est * 100
    else:
        Qc_est = Ql_est * 100
    
    return max(Ql_est, min(Qc_est, 1e9))

def estimate_phi0(freq, S11_norm, f0_est, Ql_est):
    """Estimate asymmetry angle from IQ circle."""
    fwhm = f0_est / max(Ql_est, 100)
    near_res = np.abs(freq - f0_est) < 2 * fwhm
    
    if np.sum(near_res) < 5:
        return 0.0
    
    S11_near = S11_norm[near_res]
    
    # Off-center circle gives phi0. This is a rough guess only.
    center_re = np.mean(S11_near.real)
    center_im = np.mean(S11_near.imag)
    
    # phi0 from asymmetry
    phi0_est = np.arctan2(center_im, 1 - center_re)
    
    return phi0_est

# ==============================================================================
# S11 Reflection Model (Probst et al. Eq. 1)
# ==============================================================================

def model_S11(f, f0, Ql, Qc_abs, phi0, tau, a, alpha):
    """S11 reflection model for a notch-type resonator."""
    x = (f - f0) / f0  # Reduced frequency (dimensionless detuning)
    
    S11 = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau) * \
          (1 - (Ql / Qc_abs) * np.exp(1j * phi0) / (1 + 2j * Ql * x))
    
    return S11

def model_resonator_S11(f, f0, Ql, Qc_abs, phi0):
    """Pure resonator response without environment effects."""
    x = (f - f0) / f0
    return 1 - (Ql / Qc_abs) * np.exp(1j * phi0) / (1 + 2j * Ql * x)

def model_S11_with_bg(f, f0, Ql, Qc_abs, phi0, tau_res, a, alpha, bg):
    """
    S11 model with frequency-dependent background calibration.
    
    S_meas(f) = background(f) * S_res(f)
    where background includes delay, polynomial baseline, and residual corrections.
    """
    return eval_background_factor(f, bg, tau_res=tau_res, a=a, alpha=alpha) * \
           model_resonator_S11(f, f0, Ql, Qc_abs, phi0)

def model_S11_general(f, f0, Ql, tau, A_re, A_im, B_re, B_im, D_re, D_im):
    """
    General complex S11 model for ugly circles.

    We use this when the physics model struggles, then map A/B/D back to
    Qc/Qi as a best-effort estimate.
    """
    A = A_re + 1j * A_im
    B = B_re + 1j * B_im
    D = D_re + 1j * D_im
    
    x = (f / f0) - 1.0
    denom = 1.0 + 2j * Ql * x
    
    return A * np.exp(-2j * np.pi * f * tau) * (B - D / denom)

def calculate_Qi(Ql, Qc_abs, phi0):
    """Internal Q from loaded Q and complex coupling Q."""
    Qi_inv = 1.0 / Ql - np.cos(phi0) / Qc_abs
    return 1.0 / Qi_inv if Qi_inv > 0 else np.inf

def extract_physics_from_general(f0, Ql, A, B, D):
    """Back out Qc/Qi and baseline from the general model."""
    # Baseline: a*exp(iα) = A*B
    baseline = A * B
    a = np.abs(baseline)
    alpha = np.angle(baseline)
    
    # Coupling: (Ql/Qc)*exp(iφ0) = D/B
    coupling = D / B
    kappa = np.abs(coupling)  # Ql/Qc
    phi0 = np.angle(coupling)
    
    # Qc and Qi
    Qc_abs = Ql / kappa if kappa > 1e-10 else np.inf
    Qi = calculate_Qi(Ql, Qc_abs, phi0)
    
    return {
        'a': a, 'alpha': alpha,
        'Qc_abs': Qc_abs, 'phi0': phi0,
        'Qi': Qi, 'kappa': kappa
    }

def calculate_rms(freq, S11_data, S11_model):
    """Calculate RMS residual between data and model."""
    diff = S11_data - S11_model
    return np.sqrt(np.mean(np.abs(diff)**2))


def calculate_fit_errors(freq, S11_data, params, res_result):
    """Local uncertainty estimate from the Jacobian."""
    try:
        # Get Jacobian from least_squares result
        J = res_result.jac
        # Residual variance estimate
        residuals = res_result.fun
        n_data = len(residuals)
        n_params = len(res_result.x)
        dof = max(1, n_data - n_params)
        mse = np.sum(residuals**2) / dof
        
        # Covariance matrix from J^T J
        JtJ = J.T @ J
        try:
            cov = np.linalg.inv(JtJ) * mse
            param_errs = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            # Use SVD pseudo-inverse for singular matrix
            U, s, Vt = svd(JtJ)
            s_inv = np.where(s > 1e-10 * s[0], 1.0/s, 0)
            cov = (Vt.T @ np.diag(s_inv) @ U.T) * mse
            param_errs = np.sqrt(np.abs(np.diag(cov)))
        
        # param order: [f0, log_Ql, log_Qc, phi0, tau, a, alpha]
        f0_err = param_errs[0]
        log_Ql_err = param_errs[1]
        log_Qc_err = param_errs[2]
        phi0_err = param_errs[3]
        tau_err = param_errs[4]
        a_err = param_errs[5]
        alpha_err = param_errs[6]
        
        # Convert log errors to linear: d(10^x) = 10^x * ln(10) * dx
        Ql = 10**res_result.x[1]
        Qc = 10**res_result.x[2]
        Ql_err = Ql * np.log(10) * log_Ql_err
        Qc_err = Qc * np.log(10) * log_Qc_err
        
        # Propagate to Qi error
        phi0_fit = res_result.x[3]
        Qi = calculate_Qi(Ql, Qc, phi0_fit)
        # dQi/dQl, dQi/dQc, dQi/dphi0 via error propagation
        if np.isfinite(Qi):
            dQi_dQl = Qi**2 / (Ql**2)
            dQi_dQc = Qi**2 * np.cos(phi0_fit) / (Qc**2)
            dQi_dphi0 = Qi**2 * np.sin(phi0_fit) / Qc
            Qi_err = np.sqrt(
                (dQi_dQl * Ql_err)**2 +
                (dQi_dQc * Qc_err)**2 +
                (dQi_dphi0 * phi0_err)**2
            )
        else:
            Qi_err = np.inf
        
        # RMS residual
        rms = np.sqrt(np.mean(residuals**2))
        
        return {
            'f0_err': f0_err,
            'Ql_err': Ql_err,
            'Qc_err': Qc_err,
            'Qi_err': Qi_err,
            'phi0_err': phi0_err,
            'tau_err': tau_err,
            'a_err': a_err,
            'alpha_err': alpha_err,
            'rms': rms
        }
    except Exception as e:
        return {
            'f0_err': np.nan, 'Ql_err': np.nan, 'Qc_err': np.nan, 'Qi_err': np.nan,
            'phi0_err': np.nan, 'tau_err': np.nan, 'a_err': np.nan, 'alpha_err': np.nan,
            'rms': np.nan
        }


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    
    # File selection dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.overrideredirect(True)  # Remove window decorations
    root.update()  # Process the geometry change
    
    # Bring dialog to front on macOS.
    root.lift()
    root.attributes('-topmost', True)
    root.focus_force()
    
    filename = filedialog.askopenfilename(
        parent=root,
        title="Select S11 data file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialdir=os.path.dirname(os.path.abspath(__file__))
    )
    
    # destroy root window after selection
    root.destroy()
    
    if not filename:
        print("No file selected. Exiting.")
        sys.exit(0)
    
    try:
        freq, S11_complex = load_s11_csv(filename)
    except Exception as e:
        sys.exit(f"Error reading file '{filename}': {e}")

    S11_dB = 20 * np.log10(np.abs(S11_complex) + 1e-12)
    
    # Find resonance (still OK on raw magnitude)
    f0_init, Ql_init, dip_depth = find_resonance(freq, S11_dB)

    # Background calibration (delay + complex quadratic baseline)
    bg_state = {"bg": calibrate_background(freq, S11_complex, f0_init, Ql_init, order=2, exclude_n_lw=6.0)}

    # Build a roughly "background-removed" trace for initial phi0 guess
    bg = bg_state["bg"]
    B = eval_complex_poly(freq, bg["fc"], bg["scale"], bg["coeffs"])
    S_flat = (S11_complex * np.exp(2j * np.pi * freq * bg["tau_bg"])) / B

    # Residual baseline parameters start near identity
    tau_init = 0.0
    a_init = 1.0
    alpha_init = 0.0

    # Use background-removed data for phi0 guess
    S11_norm = S_flat / (np.median(np.abs(S_flat[bg["mask_off"]])) + 1e-12)
    Qc_init = estimate_Qc(dip_depth, Ql_init)
    phi0_init = estimate_phi0(freq, S11_norm, f0_init, Ql_init)

    # Compute background flatness (sanity check)
    off_std = np.std(S_flat[bg["mask_off"]])
    if off_std > 0.05:
        print(f"Warning: off-res std = {off_std:.3f} > 0.05, trying order=3 baseline...")
        bg_state["bg"] = calibrate_background(freq, S11_complex, f0_init, Ql_init, order=3, exclude_n_lw=6.0)
        bg = bg_state["bg"]
        B = eval_complex_poly(freq, bg["fc"], bg["scale"], bg["coeffs"])
        S_flat = (S11_complex * np.exp(2j * np.pi * freq * bg["tau_bg"])) / B
        off_std = np.std(S_flat[bg["mask_off"]])

    print(f"\nFile: {os.path.basename(filename)}")
    print(f"Background calibration:")
    print(f"  tau_bg ≈ {bg['tau_bg']*1e9:.2f} ns")
    print(f"  poly order = {bg['order']}")
    print(f"  off-res flatness (std) = {off_std:.3e}")
    print(f"Initial estimates:")
    print(f"  f0   = {f0_init/1e9:.6f} GHz")
    print(f"  Ql   = {Ql_init:.2e}")
    print(f"  |Qc| = {Qc_init:.2e}")
    print(f"  phi0 = {np.rad2deg(phi0_init):.1f}°")
    print(f"  tau_res = {tau_init*1e9:.2f} ns (residual)")
    print(f"  dip  = {dip_depth:.2f} dB")

    # Interactive plot.
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"S11 Fitter: {os.path.basename(filename)}", fontsize=12)
    
    # Position window on screen (fixes macOS off-screen issue with TkAgg)
    try:
        mngr = fig.canvas.manager
        mngr.window.wm_geometry("+100+50")  # Position at x=100, y=50 pixels from top-left
    except Exception:
        pass  # Ignore if positioning fails

    # Info box for live readout.
    ax_info = fig.add_axes([0.02, 0.40, 0.12, 0.50])
    ax_info.axis('off')
    info_text = ax_info.text(0.0, 0.95, '', fontsize=9, va='top', ha='left',
                             transform=ax_info.transAxes, family='monospace')

    # Main plots.
    ax_mag = fig.add_axes([0.18, 0.56, 0.35, 0.32])
    ax_phase = fig.add_axes([0.58, 0.56, 0.32, 0.32])
    ax_iq = fig.add_axes([0.18, 0.12, 0.32, 0.38])

    # Crop sliders for trimming junk at sweep edges.
    ax_crop_left = fig.add_axes([0.92, 0.56, 0.015, 0.32])
    ax_crop_right = fig.add_axes([0.96, 0.56, 0.015, 0.32])
    slider_crop_left = Slider(ax=ax_crop_left, label="", valmin=0.0, valmax=0.45, valinit=0.0,
                              valfmt="%.2f", orientation="vertical")
    slider_crop_right = Slider(ax=ax_crop_right, label="", valmin=0.0, valmax=0.45, valinit=0.0,
                               valfmt="%.2f", orientation="vertical")
    fig.text(0.927, 0.89, "L", fontsize=9, ha="center")
    fig.text(0.967, 0.89, "R", fontsize=9, ha="center")

    # Current parameters and errors
    params = {
        'f0': f0_init, 'Ql': Ql_init, 'Qc_abs': Qc_init, 'phi0': phi0_init,
        'tau': tau_init, 'a': a_init, 'alpha': alpha_init
    }
    param_errors = {
        'f0_err': np.nan, 'Ql_err': np.nan, 'Qc_err': np.nan, 'Qi_err': np.nan,
        'phi0_err': np.nan, 'tau_err': np.nan, 'a_err': np.nan, 'alpha_err': np.nan, 'rms': np.nan
    }
    # Use a mutable container so nested functions can modify these
    crop_state = {'left': 0.0, 'right': 0.0}

    def get_crop_indices():
        """Get crop indices based on slider values."""
        n = len(freq)
        i_left = int(crop_state['left'] * n)
        i_right = n - int(crop_state['right'] * n)
        i_right = max(i_right, i_left + 10)  # Ensure at least 10 points
        return i_left, i_right

    def get_cropped_data():
        """Get cropped frequency and S11 data based on slider values."""
        i_left, i_right = get_crop_indices()
        return freq[i_left:i_right], S11_complex[i_left:i_right]

    def correct_phase(freq_data, S11_data, tau_res):
        """Phase with total delay (tau_bg + tau_res) removed and linear trend flattened."""
        tau_total = bg_state["bg"]["tau_bg"] + tau_res
        S11_corr = S11_data * np.exp(2j * np.pi * freq_data * tau_total)
        phase = np.unwrap(np.angle(S11_corr))
        coeffs = np.polyfit(freq_data, phase, 1)
        phase_flat = phase - np.polyval(coeffs, freq_data)
        return np.rad2deg(phase_flat)

    def correct_model_phase(freq_data, S11_model, tau_res):
        """Phase correction that mirrors the data path for fair comparison."""
        tau_total = bg_state["bg"]["tau_bg"] + tau_res
        S11_corr = S11_model * np.exp(2j * np.pi * freq_data * tau_total)
        phase = np.unwrap(np.angle(S11_corr))
        freq_crop, S11_crop = get_cropped_data()
        S11_data_corr = S11_crop * np.exp(2j * np.pi * freq_crop * tau_total)
        phase_data = np.unwrap(np.angle(S11_data_corr))
        coeffs = np.polyfit(freq_crop, phase_data, 1)
        phase_flat = phase - np.polyval(coeffs, freq_data)
        return np.rad2deg(phase_flat)

    # First draw uses cropped data only.
    freq_crop, S11_crop = get_cropped_data()
    S11_dB_crop = 20 * np.log10(np.abs(S11_crop) + 1e-12)
    phase_crop = correct_phase(freq_crop, S11_crop, params['tau'])

    # Data plots.
    line_mag_data, = ax_mag.plot(freq_crop/1e9, S11_dB_crop, 'b.', ms=3, alpha=0.6, label='Data')
    line_phase_data, = ax_phase.plot(freq_crop/1e9, phase_crop, 'b.', ms=3, alpha=0.6, label='Data')
    line_iq_data, = ax_iq.plot(S11_crop.real, S11_crop.imag, 'b.', ms=3, alpha=0.6, label='Data')

    # Model plots (using background-aware model).
    S11_model_init = model_S11_with_bg(freq_crop, f0_init, Ql_init, Qc_init, phi0_init, 
                                        tau_init, a_init, alpha_init, bg_state["bg"])
    phase_model = correct_model_phase(freq_crop, S11_model_init, params['tau'])
    line_mag_fit, = ax_mag.plot(freq_crop/1e9, 20*np.log10(np.abs(S11_model_init)+1e-12), 'r-', lw=2, label='Model')
    line_phase_fit, = ax_phase.plot(freq_crop/1e9, phase_model, 'r-', lw=2, label='Model')
    line_iq_fit, = ax_iq.plot(S11_model_init.real, S11_model_init.imag, 'r-', lw=2, label='Model')

    ax_mag.set_xlabel('Frequency (GHz)')
    ax_mag.set_ylabel('|S11| (dB)')
    ax_mag.legend(loc='upper right', fontsize=8)
    ax_mag.grid(True, alpha=0.3)

    ax_phase.set_xlabel('Frequency (GHz)')
    ax_phase.set_ylabel('Phase (deg, corrected)')
    ax_phase.legend(loc='upper right', fontsize=8)
    ax_phase.grid(True, alpha=0.3)

    ax_iq.set_xlabel('Re(S11)')
    ax_iq.set_ylabel('Im(S11)')
    ax_iq.legend(loc='upper right', fontsize=8)
    ax_iq.set_aspect('equal')
    ax_iq.grid(True, alpha=0.3)

    def update_info():
        Qi = calculate_Qi(params['Ql'], params['Qc_abs'], params['phi0'])
        
        txt = f"f0   = {params['f0']/1e9:.6f} GHz\n"
        if np.isfinite(param_errors['f0_err']):
            txt += f"       ± {param_errors['f0_err']/1e3:.1f} kHz\n"
        
        txt += f"Ql   = {params['Ql']:.2e}\n"
        if np.isfinite(param_errors['Ql_err']):
            txt += f"       ± {param_errors['Ql_err']:.1e}\n"
        
        txt += f"|Qc| = {params['Qc_abs']:.2e}\n"
        if np.isfinite(param_errors['Qc_err']):
            txt += f"       ± {param_errors['Qc_err']:.1e}\n"
        
        txt += f"Qi   = {Qi:.2e}\n"
        if np.isfinite(param_errors['Qi_err']):
            txt += f"       ± {param_errors['Qi_err']:.1e}\n"
        
        txt += f"φ0   = {np.rad2deg(params['phi0']):.1f}°\n"
        if np.isfinite(param_errors['phi0_err']):
            txt += f"       ± {np.rad2deg(param_errors['phi0_err']):.1f}°\n"
        
        tau_total = bg_state["bg"]["tau_bg"] + params['tau']
        txt += f"τ_res = {params['tau']*1e9:.2f} ns\n"
        txt += f"τ_tot = {tau_total*1e9:.2f} ns\n"
        txt += f"a    = {params['a']:.4f}\n"
        txt += f"α    = {np.rad2deg(params['alpha']):.1f}°\n"
        
        if np.isfinite(param_errors['rms']):
            txt += f"\nRMS  = {param_errors['rms']:.2e}"
        
        info_text.set_text(txt)

    update_info()

    # === PARAMETER SLIDERS (bottom right area) ===
    slider_color = 'lightgoldenrodyellow'
    sliders = {}

    ax_f0 = fig.add_axes([0.58, 0.38, 0.34, 0.025], facecolor=slider_color)
    sliders['f0'] = Slider(ax_f0, 'f0 (GHz)', freq.min()/1e9, freq.max()/1e9, 
                           valinit=params['f0']/1e9, valfmt='%.6f')

    ax_Ql = fig.add_axes([0.58, 0.33, 0.34, 0.025], facecolor=slider_color)
    sliders['Ql'] = Slider(ax_Ql, 'log₁₀(Ql)', 2, 7, 
                           valinit=np.log10(params['Ql']), valfmt='%.2f')

    ax_Qc = fig.add_axes([0.58, 0.28, 0.34, 0.025], facecolor=slider_color)
    sliders['Qc'] = Slider(ax_Qc, 'log₁₀(|Qc|)', 2, 9, 
                           valinit=np.log10(params['Qc_abs']), valfmt='%.2f')

    ax_phi = fig.add_axes([0.58, 0.23, 0.34, 0.025], facecolor=slider_color)
    sliders['phi0'] = Slider(ax_phi, 'φ0 (deg)', -180, 180, 
                             valinit=np.rad2deg(params['phi0']), valfmt='%.1f')

    ax_tau = fig.add_axes([0.58, 0.18, 0.34, 0.025], facecolor=slider_color)
    sliders['tau'] = Slider(ax_tau, 'τ_res (ns)', -50, 50, 
                            valinit=params['tau']*1e9, valfmt='%.2f')

    ax_a = fig.add_axes([0.58, 0.13, 0.34, 0.025], facecolor=slider_color)
    sliders['a'] = Slider(ax_a, 'a', 0.01, 1.5, 
                          valinit=params['a'], valfmt='%.4f')

    ax_alpha = fig.add_axes([0.58, 0.08, 0.34, 0.025], facecolor=slider_color)
    sliders['alpha'] = Slider(ax_alpha, 'α (deg)', -180, 180, 
                              valinit=np.rad2deg(params['alpha']), valfmt='%.1f')

    def update_display():
        """Update all plots with current cropped data and model."""
        crop_state['left'] = slider_crop_left.val
        crop_state['right'] = slider_crop_right.val
        
        # Use current crop to avoid fitting edges with junk.
        freq_crop, S11_crop = get_cropped_data()
        S11_dB_crop = 20 * np.log10(np.abs(S11_crop) + 1e-12)
        phase_crop = correct_phase(freq_crop, S11_crop, params['tau'])
        
        # Update plots.
        line_mag_data.set_data(freq_crop/1e9, S11_dB_crop)
        line_phase_data.set_data(freq_crop/1e9, phase_crop)
        line_iq_data.set_data(S11_crop.real, S11_crop.imag)
        
        # Model on the same cropped grid (using background-aware model).
        S11_model = model_S11_with_bg(
            freq_crop,
            params['f0'], params['Ql'], params['Qc_abs'], params['phi0'],
            params['tau'], params['a'], params['alpha'],
            bg_state["bg"]
        )
        phase_model = correct_model_phase(freq_crop, S11_model, params['tau'])
        
        line_mag_fit.set_data(freq_crop/1e9, 20*np.log10(np.abs(S11_model)+1e-12))
        line_phase_fit.set_data(freq_crop/1e9, phase_model)
        line_iq_fit.set_data(S11_model.real, S11_model.imag)
        
        # Autoscale to the current crop.
        ax_mag.relim()
        ax_mag.autoscale_view()
        ax_phase.relim()
        ax_phase.autoscale_view()
        ax_iq.relim()
        ax_iq.autoscale_view()
        ax_iq.set_aspect('equal')

    def update_plot(val=None):
        params['f0'] = sliders['f0'].val * 1e9
        params['Ql'] = 10**sliders['Ql'].val
        params['Qc_abs'] = 10**sliders['Qc'].val
        params['phi0'] = np.deg2rad(sliders['phi0'].val)
        params['tau'] = sliders['tau'].val * 1e-9
        params['a'] = sliders['a'].val
        params['alpha'] = np.deg2rad(sliders['alpha'].val)
        
        # Calculate RMS for current slider values
        freq_crop, S11_crop = get_cropped_data()
        S11_model = model_S11_with_bg(
            freq_crop,
            params['f0'], params['Ql'], params['Qc_abs'], params['phi0'],
            params['tau'], params['a'], params['alpha'],
            bg_state["bg"]
        )
        rms = calculate_rms(freq_crop, S11_crop, S11_model)
        param_errors['rms'] = rms
        
        update_display()
        update_info()
        fig.canvas.draw_idle()

    def on_crop_change(val=None):
        crop_state['left'] = slider_crop_left.val
        crop_state['right'] = slider_crop_right.val
        
        # Recalculate RMS for new crop region
        freq_crop, S11_crop = get_cropped_data()
        S11_model = model_S11_with_bg(
            freq_crop,
            params['f0'], params['Ql'], params['Qc_abs'], params['phi0'],
            params['tau'], params['a'], params['alpha'],
            bg_state["bg"]
        )
        rms = calculate_rms(freq_crop, S11_crop, S11_model)
        param_errors['rms'] = rms
        
        update_display()
        update_info()
        fig.canvas.draw_idle()

    for s in sliders.values():
        s.on_changed(update_plot)

    slider_crop_left.on_changed(on_crop_change)
    slider_crop_right.on_changed(on_crop_change)

    # Buttons
    ax_fit = fig.add_axes([0.58, 0.02, 0.10, 0.04])
    btn_fit = Button(ax_fit, 'Fit', color=slider_color)

    ax_fit_gen = fig.add_axes([0.69, 0.02, 0.10, 0.04])
    btn_fit_gen = Button(ax_fit_gen, 'Fit General', color='lightblue')

    ax_bg = fig.add_axes([0.80, 0.02, 0.10, 0.04])
    btn_bg = Button(ax_bg, 'Recal BG', color='lightgreen')

    ax_reset = fig.add_axes([0.91, 0.02, 0.08, 0.04])
    btn_reset = Button(ax_reset, 'Reset', color=slider_color)

    def on_fit(event):
        """Multi-stage least squares fit on cropped data with background model."""
        # Get cropped data for fitting
        freq_fit, S11_fit = get_cropped_data()
        bg = bg_state["bg"]
        
        # Stage 1: Fit (f0, Ql, Qc, phi0) with fixed residual baseline (tau_res=0, a=1, alpha=0)
        def residual_s1(p):
            f0_t, log_Ql, log_Qc, phi0_t = p
            Ql_t = 10**log_Ql
            Qc_t = 10**log_Qc
            
            S11_m = model_S11_with_bg(freq_fit, f0_t, Ql_t, Qc_t, phi0_t, 
                                       0.0, 1.0, 0.0, bg)
            
            # Weight on-resonance more
            fwhm = f0_t / Ql_t
            dist = np.abs(freq_fit - f0_t) / max(fwhm, 1e3)
            weights = 1 + 2 * np.exp(-0.5 * dist**2)
            
            diff = S11_fit - S11_m
            return np.concatenate([weights * diff.real, weights * diff.imag])
        
        p0_s1 = [params['f0'], np.log10(params['Ql']), np.log10(params['Qc_abs']), params['phi0']]
        bounds_s1 = ([freq_fit.min(), 2, 2, -np.pi], [freq_fit.max(), 8, 10, np.pi])
        
        res1 = least_squares(residual_s1, p0_s1, bounds=bounds_s1, method='trf', ftol=1e-10)
        f0_fit, log_Ql, log_Qc, phi0_fit = res1.x
        Ql_fit = 10**log_Ql
        Qc_fit = 10**log_Qc
        
        # Stage 2: Full refinement (allow small residual corrections)
        def residual_s2(p):
            f0_t, log_Ql, log_Qc, phi0_t, tau_t, a_t, alpha_t = p
            Ql_t = 10**log_Ql
            Qc_t = 10**log_Qc
            
            S11_m = model_S11_with_bg(freq_fit, f0_t, Ql_t, Qc_t, phi0_t, 
                                       tau_t, a_t, alpha_t, bg)
            diff = S11_fit - S11_m
            return np.concatenate([diff.real, diff.imag])
        
        p0_s2 = [f0_fit, np.log10(Ql_fit), np.log10(Qc_fit), phi0_fit,
                 params['tau'], params['a'], params['alpha']]
        # Tighter tau_res bounds (±50 ns) since background already captures main delay
        bounds_s2 = ([freq_fit.min(), 2, 2, -np.pi, -5e-8, 0.5, -np.pi],
                     [freq_fit.max(), 8, 10, np.pi, 5e-8, 1.5, np.pi])
        
        res2 = least_squares(residual_s2, p0_s2, bounds=bounds_s2, method='trf', ftol=1e-12)
        f0_fit, log_Ql, log_Qc, phi0_fit, tau_fit, a_fit, alpha_fit = res2.x
        Ql_fit = 10**log_Ql
        Qc_fit = 10**log_Qc
        Qi_fit = calculate_Qi(Ql_fit, Qc_fit, phi0_fit)
        
        # Calculate errors from covariance
        param_errors.update(calculate_fit_errors(freq_fit, S11_fit, params, res2))
        
        tau_total = bg_state["bg"]["tau_bg"] + tau_fit
        print(f"\nFit results:")
        print(f"  f0   = {f0_fit/1e9:.6f} GHz ± {param_errors['f0_err']/1e3:.1f} kHz")
        print(f"  Ql   = {Ql_fit:.2e} ± {param_errors['Ql_err']:.1e}")
        print(f"  |Qc| = {Qc_fit:.2e} ± {param_errors['Qc_err']:.1e}")
        print(f"  Qi   = {Qi_fit:.2e} ± {param_errors['Qi_err']:.1e}")
        print(f"  phi0 = {np.rad2deg(phi0_fit):.1f}° ± {np.rad2deg(param_errors['phi0_err']):.1f}°")
        print(f"  tau_res = {tau_fit*1e9:.2f} ns (residual)")
        print(f"  tau_total = {tau_total*1e9:.2f} ns (bg + residual)")
        print(f"  RMS  = {param_errors['rms']:.2e}")
        
        # Update sliders
        sliders['f0'].set_val(f0_fit / 1e9)
        sliders['Ql'].set_val(np.log10(Ql_fit))
        sliders['Qc'].set_val(np.log10(Qc_fit))
        sliders['phi0'].set_val(np.rad2deg(phi0_fit))
        sliders['tau'].set_val(tau_fit * 1e9)
        sliders['a'].set_val(a_fit)
        sliders['alpha'].set_val(np.rad2deg(alpha_fit))

    def on_reset(event):
        param_errors.update({
            'f0_err': np.nan, 'Ql_err': np.nan, 'Qc_err': np.nan, 'Qi_err': np.nan,
            'phi0_err': np.nan, 'tau_err': np.nan, 'a_err': np.nan, 'alpha_err': np.nan, 'rms': np.nan
        })
        slider_crop_left.set_val(0.0)
        slider_crop_right.set_val(0.0)
        sliders['f0'].set_val(f0_init / 1e9)
        sliders['Ql'].set_val(np.log10(Ql_init))
        sliders['Qc'].set_val(np.log10(Qc_init))
        sliders['phi0'].set_val(np.rad2deg(phi0_init))
        sliders['tau'].set_val(tau_init * 1e9)
        sliders['a'].set_val(a_init)
        sliders['alpha'].set_val(np.rad2deg(alpha_init))

    def on_recal_bg(event):
        """Recalibrate background using current crop and slider values."""
        freq_crop, S11_crop = get_cropped_data()
        bg_state["bg"] = calibrate_background(
            freq_crop, S11_crop,
            params["f0"], params["Ql"],
            order=2, exclude_n_lw=6.0
        )
        bg = bg_state["bg"]
        
        # Compute flatness metric
        B = eval_complex_poly(freq_crop, bg["fc"], bg["scale"], bg["coeffs"])
        S_flat = (S11_crop * np.exp(2j * np.pi * freq_crop * bg["tau_bg"])) / B
        off_std = np.std(S_flat[bg["mask_off"]])
        
        # Auto-upgrade to order=3 if needed
        if off_std > 0.05:
            print(f"Off-res std = {off_std:.3f} > 0.05, trying order=3...")
            bg_state["bg"] = calibrate_background(
                freq_crop, S11_crop,
                params["f0"], params["Ql"],
                order=3, exclude_n_lw=6.0
            )
            bg = bg_state["bg"]
            B = eval_complex_poly(freq_crop, bg["fc"], bg["scale"], bg["coeffs"])
            S_flat = (S11_crop * np.exp(2j * np.pi * freq_crop * bg["tau_bg"])) / B
            off_std = np.std(S_flat[bg["mask_off"]])
        
        print(f"\nRecalibrated background:")
        print(f"  tau_bg ≈ {bg['tau_bg']*1e9:.2f} ns")
        print(f"  poly order = {bg['order']}")
        print(f"  off-res flatness (std) = {off_std:.3e}")
        
        update_display()
        update_info()
        fig.canvas.draw_idle()

    btn_fit.on_clicked(on_fit)
    btn_bg.on_clicked(on_recal_bg)
    btn_reset.on_clicked(on_reset)


    def on_fit_general(event):
        """
        Fit using the general S11 model: S11 = A*exp(-2πifτ)*[B - D/(1+2iQl*x)].
        
        More flexible than physics-based model, then extracts Qi/Qc from result.
        """
        
        freq_fit, S11_fit = get_cropped_data()
        N = len(freq_fit)
        
        # Edge-based baseline estimate (robust)
        edge = max(5, int(0.08 * N))
        S_edge = np.concatenate([S11_fit[:edge], S11_fit[-edge:]])
        B0 = np.mean(S_edge)  # Complex baseline
        
        # Resonance location and depth
        mag = np.abs(S11_fit)
        idx0 = np.argmin(mag)
        f0_guess = freq_fit[idx0]
        S_min = S11_fit[idx0]
        D0 = B0 - S_min  # Complex depth vector
        
        # Q and delay guesses
        Ql_guess = params['Ql'] if params['Ql'] > 100 else 1e4
        tau_guess = params['tau']
        A0 = 1.0 + 0.0j
        
        # Initial parameter vector
        p0 = np.array([
            f0_guess, Ql_guess, tau_guess,
            np.real(A0), np.imag(A0),
            np.real(B0), np.imag(B0),
            np.real(D0), np.imag(D0)
        ], dtype=float)
        
        # Bounds
        fmin, fmax = freq_fit.min(), freq_fit.max()
        bounds_lower = [fmin, 100, -1e-6, -10, -10, -10, -10, -10, -10]
        bounds_upper = [fmax, 1e9, 1e-6, 10, 10, 10, 10, 10, 10]
        
        # Residual function with resonance weighting
        def residual_gen(p):
            f0_t, Ql_t, tau_t, A_re, A_im, B_re, B_im, D_re, D_im = p
            S_model = model_S11_general(freq_fit, f0_t, Ql_t, tau_t,
                                         A_re, A_im, B_re, B_im, D_re, D_im)
            
            # Weight resonance region more
            fwhm = f0_t / max(Ql_t, 100)
            sigma = 0.15 * (fmax - fmin)
            weights = 1.0 + 2.0 * np.exp(-((freq_fit - f0_t) / sigma)**2)
            
            diff = S11_fit - S_model
            return np.concatenate([weights * diff.real, weights * diff.imag])
        
        res = least_squares(residual_gen, p0, bounds=(bounds_lower, bounds_upper),
                            method='trf', ftol=1e-12, max_nfev=20000)
        
        f0_fit, Ql_fit, tau_fit, A_re, A_im, B_re, B_im, D_re, D_im = res.x
        A = A_re + 1j * A_im
        B = B_re + 1j * B_im
        D = D_re + 1j * D_im
        
        # Extract physics from general model
        phys = extract_physics_from_general(f0_fit, Ql_fit, A, B, D)
        Qc_fit = phys['Qc_abs']
        Qi_fit = phys['Qi']
        phi0_fit = phys['phi0']
        a_fit = phys['a']
        alpha_fit = phys['alpha']
        kappa_fit = phys['kappa']
        
        # RMS error
        S_model = model_S11_general(freq_fit, f0_fit, Ql_fit, tau_fit,
                                     A_re, A_im, B_re, B_im, D_re, D_im)
        rms = np.sqrt(np.mean(np.abs(S11_fit - S_model)**2))
        
        # Update param_errors with RMS (full error estimation would need Jacobian analysis)
        param_errors.update({
            'f0_err': np.nan, 'Ql_err': np.nan, 'Qc_err': np.nan, 'Qi_err': np.nan,
            'phi0_err': np.nan, 'tau_err': np.nan, 'a_err': np.nan, 'alpha_err': np.nan,
            'rms': rms
        })
        
        print(f"\n===== Fit Results (General S11 Model) =====")
        print(f"  f0   = {f0_fit/1e9:.6f} GHz")
        print(f"  Ql   = {Ql_fit:.2e}")
        print(f"  τ    = {tau_fit*1e9:.2f} ns")
        print(f"  A    = {A:.4f}")
        print(f"  B    = {B:.4f} (baseline)")
        print(f"  D    = {D:.4f} (depth)")
        print(f"  --- Derived physics ---")
        print(f"  |Qc| = {Qc_fit:.2e}")
        print(f"  Qi   = {Qi_fit:.2e}")
        print(f"  φ0   = {np.rad2deg(phi0_fit):.1f}°")
        print(f"  κ    = {kappa_fit:.3f}")
        print(f"  a    = {a_fit:.4f}")
        print(f"  RMS  = {rms:.2e}")
        
        # Update sliders with derived physics parameters
        sliders['f0'].set_val(f0_fit / 1e9)
        sliders['Ql'].set_val(np.log10(Ql_fit))
        sliders['Qc'].set_val(np.log10(Qc_fit) if Qc_fit < 1e9 else 9)
        sliders['phi0'].set_val(np.rad2deg(phi0_fit))
        sliders['tau'].set_val(tau_fit * 1e9)
        sliders['a'].set_val(min(a_fit, 1.5))
        sliders['alpha'].set_val(np.rad2deg(alpha_fit))


    btn_fit_gen.on_clicked(on_fit_general)

    plt.show()
