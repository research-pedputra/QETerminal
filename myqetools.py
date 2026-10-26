## to Plot Band Structure
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import os
import re
from ase.io import read

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import sys

sys.path.append('API')



# ==============================================================
# CLASS 1 — BAND STRUCTURE (your original code)
# ==============================================================

class BandStructure:
    def __init__(self):
        self.data = None
        self.fermi_energy = None
        self.k = None
        self.bands = None
        self.high_symmetry_points = None
        self.fig, self.ax = None, None
    def read_scf_info(file):
        Ef, Ne = None, None
        with open(file) as f:
            for line in f:
                if "the Fermi energy is" in line: Ef = float(line.split()[4])
                if "number of electrons" in line: Ne = float(line.split()[-1])
        return Ef, Ne
    
    def effmass(E, k, hole=False):
        d2E_dk2 = np.gradient(np.gradient(E, k), k)
        m = 1 / np.mean(d2E_dk2) * (6.5821220E-16**2) / 9.1093897E-31
        return -m if hole else m
    
    def get_band_gap(valence_band, conduction_band):
        Ev_max = np.max(valence_band)   
        Ec_min = np.min(conduction_band) 
        Eg = Ec_min - Ev_max
        return Eg, Ev_max, Ec_min
    
    def get_bandstructure(self, data, fermi_energy, kpoints):
        self.data = data
        self.fermi_energy = fermi_energy
        self.k = np.unique(data[:, 0])
        self.bands = np.reshape(data[:, 1], (-1, len(self.k)))
        self.high_symmetry_points = kpoints

    def plot_band_structure(self):
        if self.data is None:
            raise ValueError("Band structure data has not been loaded.")
        self.fig, self.ax = plt.subplots(1, 1, figsize=(3, 6))
        for i, band in enumerate(self.bands):
            self.ax.plot(self.k, band - self.fermi_energy, linewidth=1, alpha=0.5, color='r')
        self.ax.set_xlim(min(self.k), max(self.k))
        self.ax.set_ylim(-5, 5)

        if self.high_symmetry_points is not None:
            x_coords = [x_coord for _, x_coord in self.high_symmetry_points]
            x_labels = [symmetry_point for symmetry_point, _ in self.high_symmetry_points]
            self.ax.set_xticks(x_coords)
            self.ax.set_xticklabels(x_labels, fontsize=14, fontweight='bold')

            for _, x_coordinate in self.high_symmetry_points:
                self.ax.axvline(x=x_coordinate, color='k', linestyle='-', linewidth=1)
        self.ax.set_ylabel("E-E$_f$ (eV)", fontsize=14, fontweight='bold')

    def plot_band_number(self, band_number, overlay=True):
        if band_number < 1 or band_number > len(self.bands):
            raise ValueError("Invalid band number.")
        
        if overlay:
            if self.fig is None or self.ax is None:
                self.plot_band_structure()
            self.ax.plot(self.k, self.bands[band_number - 1] - self.fermi_energy, linewidth=1, color='b')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(4, 6))
            for i, band in enumerate(self.bands):
                ax.plot(self.k, band - self.fermi_energy, linewidth=1, alpha=0.5, color='r')
            ax.plot(self.k, self.bands[band_number - 1] - self.fermi_energy, linewidth=1, color='b')
            ax.set_xlim(min(self.k), max(self.k))
            ax.set_ylim(-5, 5)
            if self.high_symmetry_points is not None:
                x_coords = [x_coord for _, x_coord in self.high_symmetry_points]
                x_labels = [symmetry_point for symmetry_point, _ in self.high_symmetry_points]
                ax.set_xticks(x_coords)
                ax.set_xticklabels(x_labels)
                for _, x_coordinate in self.high_symmetry_points:
                    ax.axvline(x=x_coordinate, color='k', linestyle='-', linewidth=1)
            ax.set_ylabel("E-E$_f$ (eV)", fontsize=14, fontweight='bold')
            ax.set_title(f"Band {band_number}")
            plt.tick_params(axis="both", which="major", labelsize=12, width=1.5)
            plt.xticks(fontsize=12, fontweight="bold")
            plt.yticks(fontsize=12, fontweight="bold")
            plt.show()

    def shift_band(self, start_band_number, shift_value):
        if start_band_number < 1 or start_band_number > len(self.bands):
            raise ValueError("Invalid band number.")
        for i in range(start_band_number - 1, len(self.bands)):
            self.bands[i] += shift_value

    def plot_shifted_band_structure(self, start_band_number, shift_value):
        self.shift_band(start_band_number, shift_value)
        self.plot_band_structure()

    def get_band_data_point(self, band_number):
        if band_number < 1 or band_number > len(self.bands):
            raise ValueError("Invalid band number.")
        return self.k, self.bands[band_number - 1] - self.fermi_energy

    def save_figure(self, filename="band_structure.png", dpi=300, transparent=True):
        if self.fig is None:
            raise ValueError("No figure to save. Please plot the band structure first.")
        self.fig.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches='tight')


# ==============================================================
# CLASS 2 — EFFECTIVE MASS ANALYZER
# ==============================================================

class EffectiveMassAnalyzer:
    def __init__(self, band_structure):
        self.bs = band_structure

    @staticmethod
    def effmass_inverse_avg(E_list, k, idx_range, hole=False, use_3d_trace=False):
        """
        Computes degeneracy-aware effective mass using:
        <1/m*> = (1/N ħ²) Σ_n |∂²E_n/∂k²|
        m* = 1 / <1/m*>
        """
        a, b = idx_range
        curvatures = []
        for E in E_list:
            d2E = np.gradient(np.gradient(E[a:b], k[a:b]), k[a:b])
            curvatures.append(np.mean(np.abs(d2E)))
        mean_curv = np.mean(curvatures)
        if use_3d_trace:
            mean_curv /= 3.0

        hbar = 6.5821220e-16  # eV·s
        m_e = 9.1093897e-31   # kg
        m_eff = (hbar**2 / m_e) / mean_curv
        return -m_eff if hole else m_eff

    @staticmethod
    def get_band_gap(valence_band, conduction_band):
        Ev_max = np.max(valence_band)
        Ec_min = np.min(conduction_band)
        Eg = Ec_min - Ev_max
        return Eg, Ev_max, Ec_min

    def analyze(self, VBM_main, CBM_main, VBM_degenerate=None, CBM_degenerate=None,
                range_e=(35, 45), range_h=(35, 45)):
        if VBM_degenerate is None:
            VBM_degenerate = []
        if CBM_degenerate is None:
            CBM_degenerate = []

        # ==== ELECTRON SIDE ====
        a, b = range_e
        k_cbm, E_cbm = self.bs.get_band_data_point(CBM_main)
        electron_k = k_cbm
        electron_E_list = [E_cbm]
        for idx in CBM_degenerate:
            _, Ei = self.bs.get_band_data_point(idx)
            electron_E_list.append(Ei)

        me = self.effmass_inverse_avg(electron_E_list, electron_k, (a, b), hole=False)
        

        E_e_mean = np.mean(np.vstack(electron_E_list), axis=0)
        k_e_fit = electron_k[a:b]
        E_e_fit = E_e_mean[a:b]
        spline_e = UnivariateSpline(k_e_fit, E_e_fit, s=0)
        k_smooth_e = np.linspace(k_e_fit.min(), k_e_fit.max(), 600)

        # ==== HOLE SIDE ====
        c, d = range_h
        k_vbm, E_vbm = self.bs.get_band_data_point(VBM_main)
        hole_k = k_vbm
        hole_E_list = [E_vbm]
        for idx in VBM_degenerate:
            _, Ei = self.bs.get_band_data_point(idx)
            hole_E_list.append(Ei)

        mh = self.effmass_inverse_avg(hole_E_list, hole_k, (c, d), hole=True)
        

        E_h_mean = np.mean(np.vstack(hole_E_list), axis=0)
        k_h_fit = hole_k[c:d]
        E_h_fit = E_h_mean[c:d]
        spline_h = UnivariateSpline(k_h_fit, E_h_fit, s=0)
        k_smooth_h = np.linspace(k_h_fit.min(), k_h_fit.max(), 600)

        # ==== PLOTTING ====
        self._plot_effective_mass(electron_E_list, electron_k, a, b, E_e_fit, spline_e,
                                  k_smooth_e, [CBM_main] + CBM_degenerate, "Electron", "red")
        self._plot_effective_mass(hole_E_list, hole_k, c, d, E_h_fit, spline_h,
                                  k_smooth_h, [VBM_main] + VBM_degenerate, "Hole", "green")

        # ==== BAND GAP ====
        Eg, Ev_max, Ec_min = self.get_band_gap(E_vbm, E_cbm)
        print(f"Band gap = {Eg:.3f} eV")
        print(f"Electron effective mass ({len(electron_E_list)} band(s)) = {me:.3f} m_e")
        print(f"Hole effective mass ({len(hole_E_list)} band(s)) = {mh:.3f} m_e")

        return me, mh, Eg

    def _plot_effective_mass(self, band_list, k, start, end, E_fit, spline, k_smooth,
                             bands, title, color):
        plt.figure(figsize=(6, 5))
        for Ei in band_list:
            plt.scatter(k[start:end], Ei[start:end], s=32, alpha=0.35, edgecolors='none')
        plt.scatter(k[start:end], E_fit, s=64, facecolors=color,
                    edgecolors='black', linewidths=0.8, label=f'{title} data')
        plt.plot(k_smooth, spline(k_smooth), linewidth=1.6, label=f'{title} fit')
        plt.xlabel("k-path")
        plt.ylabel("E - E$_f$ (eV)")
        plt.title(f"{title} side (bands: {bands})")
        plt.legend()
        plt.grid(True)
        plt.show()


# ==============================================================
# EXAMPLE USAGE
# ==============================================================

#if __name__ == "__main__":
#    # === Input setup ===
#    fermi_energy = 5.6925
#    bands_file = '../Works/Screening/part_1/CsGeCl.bands.gnu'
#    data = np.loadtxt(bands_file)
#    high_sym_pts = [("Γ", 0.0000), ("M", 0.7071), ("R", 1.2071),
#                    ("X", 1.9142), ("X1", 2.6213)]

    # === Build and plot bandstructure ===
#    bs = BandStructure()
#    bs.get_bandstructure(data, fermi_energy, high_sym_pts)
#    bs.plot_shifted_band_structure(23, 2.84 - 1.013)
#    for bnum in [22, 23, 24, 25]:
#        bs.plot_band_number(bnum)
#    bs.save_figure("../Works/Figure/CGC.png")

#    # === Analyze effective masses ===
#    analyzer = EffectiveMassAnalyzer(bs)
#    analyzer.analyze(VBM_main=22, CBM_main=23,
#                     VBM_degenerate=[],
#                     CBM_degenerate=[24, 25],
#                     range_e=(35, 45), range_h=(35, 45))


# to plot PDOS

class PDOSPlotter:
    def __init__(self, fermi_energy, orientation='vertical'):
        self.fermi = fermi_energy
        self.orientation = orientation 

    def read_pdos(self, filepath):
        with open(filepath) as f:
            for i, line in enumerate(f):
                if line.startswith("#"):
                    break
        data = np.loadtxt(filepath, skiprows=i + 1)
        energy = data[:, 0] - self.fermi
        pdos = data[:, 1:]
        return energy, pdos
    def read_scf_info(file):
        Ef, Ne = None, None
        with open(file) as f:
            for line in f:
                if "the Fermi energy is" in line: Ef = float(line.split()[4])
                if "number of electrons" in line: Ne = float(line.split()[-1])
        return Ef, Ne
    def infer_label(self, filepath):
        name = os.path.basename(filepath)
        match = re.search(r"\(?([A-Za-z]+)\)?_\(?([spdfg])\)?\.dat", name)
        return f"{match[1]}({match[2]})" if match else name

    def plot(self, files, xlim=None, ylim=None, orientation=None, figsize=(8,6)):
        orientation = orientation or self.orientation
        plt.figure(figsize=figsize)
    
        for label, fileinfo in files.items():
            # --- Flexible unpacking ---
            # Format can be:
            #   "Br_s": ("blue", "(Br)_(s).dat")     → no style
            #   "Br_s": ("blue", "--", "(Br)_(s).dat") → with style
            if len(fileinfo) == 2:
                color, filepath = fileinfo
                style = None
            elif len(fileinfo) == 3:
                color, style, filepath = fileinfo
            else:
                raise ValueError(f"Invalid file tuple for {label}: {fileinfo}")
    
            energy, pdos = self.read_pdos(filepath)
            display_label = self.infer_label(filepath)
    
            # --- Plot logic ---
            if pdos.shape[1] == 1:
                x, y = (energy, pdos[:, 0]) if orientation == 'vertical' else (pdos[:, 0], energy)
                if orientation == 'vertical':
                    if style:
                        plt.plot(x, y, style, color=color, label=display_label, lw=1.5)
                    else:
                        plt.fill_between(x, y, color=color, alpha=0.5, label=display_label)
                else:
                    if style:
                        plt.plot(y, x, style, color=color, label=display_label, lw=1.5)
                    else:
                        plt.fill_betweenx(y, x, color=color, alpha=0.5, label=display_label)
            else:
                up = pdos[:, 0]
                down = -pdos[:, 1]
                if orientation == 'vertical':
                    if style:
                        plt.plot(energy, up, style, color=color, label=display_label + " ↑", lw=1.5)
                        plt.plot(energy, down, style, color=color, label=display_label + " ↓", lw=1.5)
                    else:
                        plt.fill_between(energy, up, color=color, alpha=0.5, label=display_label + " ↑")
                        plt.fill_between(energy, down, color=color, alpha=0.5, label=display_label + " ↓")
                else:
                    if style:
                        plt.plot(up, energy, style, color=color, label=display_label + " ↑", lw=1.5)
                        plt.plot(down, energy, style, color=color, label=display_label + " ↓", lw=1.5)
                    else:
                        plt.fill_betweenx(energy, up, color=color, alpha=0.5, label=display_label + " ↑")
                        plt.fill_betweenx(energy, down, color=color, alpha=0.5, label=display_label + " ↓")
    
        # --- Axis & label formatting ---
        if orientation == 'vertical':
            plt.axvline(0, color='gray', linestyle='--')
            plt.xlabel("E - E$_f$ (eV)", fontweight="bold")
            plt.ylabel("Atomic Orbital Projected DOS (a.u.)", fontweight="bold")
            if xlim: plt.xlim(xlim)
            if ylim: plt.ylim(ylim)
        else:
            plt.axhline(0, color='gray', linestyle='--')
            plt.ylabel("E - E$_f$ (eV)", fontweight="bold")
            plt.xlabel("Atomic Orbital Projected DOS (a.u.)", fontweight="bold")
            if xlim: plt.ylim(xlim)
            if ylim: plt.xlim(ylim)
    
        plt.yticks([])
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()


# Example to use
#fermi_energy = 3.5926
#plotter = PDOSPlotter(fermi_energy)

#files = {
#    "Br_s": ("blue", "(Br)_(s).dat"),
#    "Br_p": ("green", "(Br)_(p).dat"),
#    "Cs_s": ("orange", "(Cs)_(s).dat"),
#    "Pb_p": ("red", "(Pb)_(p).dat"),
#}

#plotter.plot(files, xlim=(-5, 5), ylim=(-50, 50), orientation='vertical',  figsize=(8,6))
#plotter.plot(files, xlim=(-5, 5), ylim=(0, 100), orientation='horizontal',  figsize=(3,6))


#===============
import numpy as np
import matplotlib.pyplot as plt


class PDOSPlottershift:
    """
    Object-Oriented PDOS Plotter

    Features
    --------
    ✓ Read Fermi energy automatically from QE SCF output
    ✓ Read number of electrons
    ✓ Shift only conduction-band PDOS (scissor correction)
    ✓ Spin-up / spin-down plotting
    ✓ Filled or line-only PDOS
    ✓ Easy multiple subplot support
    """

    def __init__(self):

        self.fermi_energy = None
        self.num_electrons = None

    # =====================================================
    # Read SCF information
    # =====================================================
    def read_scf_info(self, scf_file):
        """
        Read Fermi energy and number of electrons from QE output.
        """

        Ef = None
        Ne = None

        with open(scf_file, "r") as f:

            for line in f:

                if "the Fermi energy is" in line:
                    Ef = float(line.split()[4])

                if "number of electrons" in line:
                    Ne = float(line.split()[-1])

        self.fermi_energy = Ef
        self.num_electrons = Ne

        print(f"Fermi Energy : {Ef:.4f} eV")
        print(f"Electrons    : {Ne}")

        return Ef, Ne

    # =====================================================
    # Shift conduction band only
    # =====================================================
    @staticmethod
    def shift_pdos_band_edge(
        energy,
        cbm_energy=0.0,
        shift_value=0.0,
    ):
        """
        Apply scissor correction only above CBM.
        """

        energy = np.asarray(energy).copy()

        energy[energy >= cbm_energy] += shift_value

        return energy

    # =====================================================
    # Read PDOS file
    # =====================================================
    @staticmethod
    def load_pdos(file):

        data = np.loadtxt(file, skiprows=1)

        energy = data[:, 0]
        up = data[:, 1]

        if data.shape[1] >= 3:
            down = -data[:, 2]
        else:
            down = np.zeros_like(up)

        return energy, up, down

    # =====================================================
    # Plot PDOS
    # =====================================================
    def plot(
        self,
        ax,
        data_dict,
        title="",
        Ef=None,
        xlim=(-5, 5),
        ylim=(-15, 15),
        fill=True,
        cbm_energy=None,
        shift_value=0,
        linewidth=1.5,
        alpha=0.30,
        fontsize=12,
        grid=True,
        legend=True,
        legend_loc="upper right",
        xlabel=r"$E-E_F$ (eV)",
        ylabel="PDOS (States/eV)",
        show_fermi=True,
    ):

        if Ef is None:

            if self.fermi_energy is None:
                raise ValueError(
                    "Fermi energy not found. "
                    "Run read_scf_info() or pass Ef."
                )

            Ef = self.fermi_energy

        for label, values in data_dict.items():

            color, linestyle, file = values

            try:

                energy, up, down = self.load_pdos(file)

                energy = energy - Ef

                if (
                    cbm_energy is not None
                    and shift_value != 0
                ):

                    energy = self.shift_pdos_band_edge(
                        energy,
                        cbm_energy,
                        shift_value,
                    )

                ax.plot(
                    energy,
                    up,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                )

                ax.plot(
                    energy,
                    down,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                )

                if fill:

                    ax.fill_between(
                        energy,
                        0,
                        up,
                        color=color,
                        alpha=alpha,
                    )

                    ax.fill_between(
                        energy,
                        0,
                        down,
                        color=color,
                        alpha=alpha,
                    )

            except FileNotFoundError:

                print(f"Cannot find {file}")

        if show_fermi:

            ax.axvline(
                0,
                color="black",
                linestyle="--",
                linewidth=0.8,
            )

        if grid:

            ax.grid(alpha=0.3)

        if legend:

            ax.legend(
                fontsize=fontsize,
                loc=legend_loc,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel(
            xlabel,
            fontsize=fontsize,
            fontweight="bold",
        )

        ax.set_ylabel(
            ylabel,
            fontsize=fontsize,
            fontweight="bold",
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=fontsize,
        )

        if title:

            ax.text(
                xlim[0] + 0.2,
                ylim[1] - 0.12 * (ylim[1] - ylim[0]),
                title,
                fontsize=fontsize,
                fontweight="bold",
            )

        return ax

    # =====================================================
    # Save figure
    # =====================================================
    @staticmethod
    def save(
        filename="PDOS.png",
        dpi=500,
        transparent=True,
    ):

        plt.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

        print(f"Saved -> {filename}")


# ==========================================================
# Example
# ==========================================================
#if __name__ == "__main__":

#    pdos = PDOSPlotter()

#    Ef, Ne = pdos.read_scf_info(
#        "scf.out"
#    )

#    pdos_data = {

#        "Pb-s": (
#            "royalblue",
#            "-",
#            "Pb.s.pdos",
#        ),

#        "Pb-p": (
#            "red",
#            "-",
#            "Pb.p.pdos",
#        ),

#        "Br-p": (
#            "green",
#            "-",
#            "Br.p.pdos",
#        ),

#        "Cs-s": (
#            "orange",
#            "-",
#            "Cs.s.pdos",
#        ),
#    }

#    fig, ax = plt.subplots(
#        figsize=(8, 6)
#    )

#    pdos.plot(
#        ax=ax,
#        data_dict=pdos_data,
#        title="CsPbBr$_3$",
#        xlim=(-5, 5),
#        ylim=(-15, 15),
#        fill=True,
#        cbm_energy=1.2,
#        shift_value=0.35,
#    )

 #   plt.tight_layout()

#    pdos.save(
#        "PDOS.png",
#        dpi=500,
#    )

#    plt.show()
#===============

class OrbitalCenter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.fermi_energy = fermi_energy
        self.energy = None
        self.pdos = None
        self._load_data()

    def _load_data(self):
        data = np.loadtxt(self.file_path, comments="#")
        self.energy = data[:, 0] - self.fermi_energy
        self.pdos = data[:, 1:]

    def get_extreme_positions(self, x_min=None, x_max=None):
        energy = self.energy
        pdos = self.pdos

        if x_min is not None and x_max is not None:
            mask = (energy >= x_min) & (energy <= x_max)
            energy = energy[mask]
            pdos = pdos[mask]

        total_pdos = pdos.sum(axis=1)
        min_pos = energy[np.argmin(total_pdos)]
        max_pos = energy[np.argmax(total_pdos)]
        return min_pos, max_pos


# Example usage
#fermi_energy = 3.5926
#file_path = "../Works/Data/PDOS_CPB_Cubic_SUC/(Pb)_(p).dat"
#oc = OrbitalCenter(file_path)
#min_pos, max_pos = oc.get_extreme_positions(x_min=0, x_max=4)

#print("Minimum position on x-axis:", min_pos)
#print("Maximum position on x-axis:", max_pos)


class EffectiveMass:
    def __init__(self, bandstructure):
        self.bandstructure = bandstructure

    def effmass(self, E, k):
        # Effective mass in units of electron mass
        hbar = 6.5821220E-16  # eV·s
        m_e = 9.1093897E-31   # kg
        second_derivative = np.gradient(np.gradient(E, k), k)
        return 1 / np.mean(second_derivative) * hbar**2 / m_e

    def calculate_average_mass(self, band_indices, k_range=None, plot=False):
        masses = []
        for idx in band_indices:
            k, E = self.bandstructure.get_band_data_point(idx)
            if k_range:
                k = k[k_range[0]:k_range[1]]
                E = E[k_range[0]:k_range[1]]
            if plot:
                plt.plot(k, E, "o", label=f"Band {idx}")
            masses.append(self.effmass(E, k))
        
        if plot:
            plt.xlabel("k-path")
            plt.ylabel("Energy (eV)")
            plt.title("Band Segments for Effective Mass")
            plt.legend()
            plt.show()

        return np.mean(masses)

# Example usage
# Assuming the class Band Structure is called as bs
#em_calc = EffectiveMass(bs)

# Calculate average effective mass using bands numbers
#me = em_calc.calculate_average_mass([23], k_range=(25, 50), plot=True)

#print(f"Effective mass = {'{:.3f}'.format(me)} me")


import numpy as np
import matplotlib.pyplot as plt


class ChargeTransfer:
    """
    Planar-integrated charge-density difference analysis.

    Parameters
    ----------
    system : ndarray
        Charge density of the combined system.
    substrate : ndarray
        Charge density of the clean substrate.
    adsorbate : ndarray
        Charge density of the isolated adsorbate.
    atoms : ase.Atoms
        Atoms object of the combined system.
    substrate_atoms : ase.Atoms
        Atoms object of the substrate.
    adsorbate_atoms : ase.Atoms
        Atoms object of the adsorbate.
    """

    def __init__(
        self,
        system,
        substrate,
        adsorbate,
        atoms,
        substrate_atoms,
        adsorbate_atoms,
    ):

        self.system = system
        self.substrate = substrate
        self.adsorbate = adsorbate

        self.atoms = atoms
        self.substrate_atoms = substrate_atoms
        self.adsorbate_atoms = adsorbate_atoms

        # Charge-density difference
        self.diff = system - substrate - adsorbate

        # Cell
        self.cell = atoms.get_cell()

        # Grid
        self.nx, self.ny, self.nz = self.diff.shape

        self.dx = np.linalg.norm(self.cell[0]) / self.nx
        self.dy = np.linalg.norm(self.cell[1]) / self.ny
        self.dz = np.linalg.norm(self.cell[2]) / self.nz

        # Cross-sectional area
        self.dA = self.dx * self.dy

        # Planar-integrated charge density
        self.rho_z = np.sum(self.diff, axis=(0, 1)) * self.dA

        # z coordinate
        self.z = np.arange(self.nz) * self.dz

        # Interface definition
        self.substrate_top = np.max(
            self.substrate_atoms.get_positions()[:, 2]
        )

        self.adsorbate_bottom = np.min(
            self.adsorbate_atoms.get_positions()[:, 2]
        )

        self.interface_z = (
            self.substrate_top + self.adsorbate_bottom
        ) / 2

    # =====================================================
    # Data
    # =====================================================

    def get_charge_profile(self):
        """Return z coordinate and planar-integrated charge density."""
        return self.z, self.rho_z

    # =====================================================
    # Plot
    # =====================================================

    def plot(
        self,
        figsize=(8, 6),
        orientation="vertical",
        color="C0",
        linewidth=2,
        show_interface=True,
        xlim=None,
        ylim=None,
        grid=False,
        ax=None,
        savefig=False,
        filename="charge_transfer.png",
        dpi=500,
        transparent=True,
    ):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        orientation = orientation.lower()

        if orientation == "vertical":

            ax.plot(
                self.z,
                self.rho_z,
                color=color,
                lw=linewidth,
                label="Charge Density Difference",
            )

            ax.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=0.8,
            )

            if show_interface:

                ax.axvline(
                    self.substrate_top,
                    color="blue",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Substrate top = {self.substrate_top:.2f} Å",
                )

                ax.axvline(
                    self.adsorbate_bottom,
                    color="green",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Adsorbate bottom = {self.adsorbate_bottom:.2f} Å",
                )

                ax.axvline(
                    self.interface_z,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Interface = {self.interface_z:.2f} Å",
                )

            ax.set_xlabel(
                "z Distance (Å)",
                fontsize=14,
                fontweight="bold",
            )

            ax.set_ylabel(
                r"$\int\!\!\int\Delta\rho(x,y,z)\,dx\,dy$",
                fontsize=14,
                fontweight="bold",
            )

        elif orientation == "horizontal":

            ax.plot(
                self.rho_z,
                self.z,
                color=color,
                lw=linewidth,
                label="Charge Density Difference",
            )

            ax.axvline(
                0,
                color="k",
                linestyle="--",
                linewidth=0.8,
            )

            if show_interface:

                ax.axhline(
                    self.substrate_top,
                    color="blue",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Substrate top = {self.substrate_top:.2f} Å",
                )

                ax.axhline(
                    self.adsorbate_bottom,
                    color="green",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Adsorbate bottom = {self.adsorbate_bottom:.2f} Å",
                )

                ax.axhline(
                    self.interface_z,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Interface = {self.interface_z:.2f} Å",
                )

            ax.set_ylabel(
                "z Distance (Å)",
                fontsize=14,
                fontweight="bold",
            )

            ax.set_xlabel(
                r"$\int\!\!\int\Delta\rho(x,y,z)\,dx\,dy$",
                fontsize=14,
                fontweight="bold",
            )

        else:
            raise ValueError(
                "orientation must be 'vertical' or 'horizontal'."
            )

        # Limits
        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        # Grid
        if grid:
            ax.grid(alpha=0.3)

        ax.legend()

        plt.tight_layout()

        if savefig:
            plt.savefig(
                filename,
                dpi=dpi,
                transparent=transparent,
                bbox_inches="tight",
            )

        plt.show()

        return ax

    # =====================================================
    # Integrations
    # =====================================================

    def integrated_charge_above(self):
        """
        Integrated charge above the interface.
        """

        mask = self.z >= self.interface_z

        return np.trapezoid(
            self.rho_z[mask],
            self.z[mask],
        )

    def integrated_charge_below(self):
        """
        Integrated charge below the interface.
        """

        mask = self.z <= self.interface_z

        return np.trapezoid(
            self.rho_z[mask],
            self.z[mask],
        )

    def integrated_charge_between(self, zmin, zmax):
        """
        Integrated charge between two z positions.
        """

        mask = (self.z >= zmin) & (self.z <= zmax)

        return np.trapezoid(
            self.rho_z[mask],
            self.z[mask],
        )

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        charge_above = self.integrated_charge_above()
        charge_below = self.integrated_charge_below()

        print("=" * 60)
        print("Charge Transfer Summary")
        print("=" * 60)
        print(f"Top substrate atom      : {self.substrate_top:.3f} Å")
        print(f"Bottom adsorbate atom   : {self.adsorbate_bottom:.3f} Å")
        print(f"Interface position      : {self.interface_z:.3f} Å")
        print("-" * 60)
        print(f"Charge above interface  : {charge_above:.6f} e")
        print(f"Charge below interface  : {charge_below:.6f} e")
        print("=" * 60)



class QeInputFile:
    def __init__(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = file_paths
        self.atoms_list = [read(f) for f in file_paths]

    def get_atom_indices(self, atoms, symbol):
        indices = [i for i, atom in enumerate(atoms) if atom.symbol == symbol]
        if not indices:
            raise ValueError(f"Atomic symbol '{symbol}' not found in the system.")
        return indices

    def get_all_distances(self, atoms, symbol1, symbol2):
        indices1 = self.get_atom_indices(atoms, symbol1)
        indices2 = self.get_atom_indices(atoms, symbol2)

        distances = []
        for index1 in indices1:
            for index2 in indices2:
                if index1 != index2:
                    dist = atoms.get_distance(index1, index2)
                    distances.append(dist)
        return distances

    def print_average_distances(self, symbol1, symbol2):
        for file_path, atoms in zip(self.file_paths, self.atoms_list):
            distances = self.get_all_distances(atoms, symbol1, symbol2)
            if distances:
                avg_dist = sum(distances) / len(distances)
                print(f"{file_path}: Average distance between {symbol1} and {symbol2} = {avg_dist:.3f} Å")
            else:
                print(f"{file_path}: No distances found between {symbol1} and {symbol2}")

# Usage:
#qe_input = QeInputFile([
#    '../Works/Data/PDOS_CPB_Cubic_SUC/scf.in',
#    '../Works/Data/PDOS_CPB_Cubic_SUC/scf.in',
#    '../Works/Data/PDOS_CPB_Cubic_SUC/scf.in'
#])

#qe_input.print_average_distances('Pb', 'Br')




class TotalEnergy:
    def __init__(self, filepath):
        self.filepath = filepath
        self.last_total_energy_ry = None

    def extract(self):
        with open(self.filepath, 'r') as file:
            for line in file:
                if "!    total energy" in line:
                    self.last_total_energy_ry = float(line.split()[-2])

    def last_TotalEnergy(self):
        if self.last_total_energy_ry is not None:
            print(f"Last Total Energy : {self.last_total_energy_ry:.8f} Ry")
            print(f"Last Total Energy : {self.last_total_energy_ry * 13.605698:.8f} eV")
        else:
            print("Total energy not found in the output file.")


class Minimaout:
    def __init__(self, file_path):
        self.file_path = file_path
        self.atoms = read(file_path, format="espresso-out")
    
    def get_atomic_positions(self, exclude_element="O"):
        cartesian_positions = [
            (atom.symbol, *atom.position)
            for atom in self.atoms if atom.symbol != exclude_element
        ]
        fractional_positions = [
            (atom.symbol, *self.atoms.get_scaled_positions()[i])
            for i, atom in enumerate(self.atoms) if atom.symbol != exclude_element
        ]
        return cartesian_positions, fractional_positions
    
    def find_highest_z_coordinates(self, exclude_element="O"):
        cartesian_positions, fractional_positions = self.get_atomic_positions(exclude_element)
        sorted_cartesian = sorted(cartesian_positions, key=lambda atom: atom[3], reverse=True)
        sorted_fractional = sorted(fractional_positions, key=lambda atom: atom[3], reverse=True)
        
        highest_cartesian = sorted_cartesian[0]
        second_highest_cartesian = sorted_cartesian[1]
        highest_fractional = sorted_fractional[0]
        second_highest_fractional = sorted_fractional[1]
        
        z_distance_cartesian = abs(highest_cartesian[3] - second_highest_cartesian[3])
        z_distance_fractional = abs(highest_fractional[3] - second_highest_fractional[3])
        
        return {
            "highest": {
                "symbol": highest_cartesian[0],
                "cartesian": highest_cartesian[1:],
                "fractional": highest_fractional[1:]
            },
            "second_highest": {
                "symbol": second_highest_cartesian[0],
                "cartesian": second_highest_cartesian[1:],
                "fractional": second_highest_fractional[1:]
            },
            "z_distances": {
                "cartesian": z_distance_cartesian,
                "fractional": z_distance_fractional
            }
        }

class Minimain:
    def __init__(self, file_path):
        self.file_path = file_path
        self.atoms = read(file_path, format="espresso-in")
    
    def get_atomic_positions(self, exclude_element="O"):
        cartesian_positions = [
            (atom.symbol, *atom.position)
            for atom in self.atoms if atom.symbol != exclude_element
        ]
        fractional_positions = [
            (atom.symbol, *self.atoms.get_scaled_positions()[i])
            for i, atom in enumerate(self.atoms) if atom.symbol != exclude_element
        ]
        return cartesian_positions, fractional_positions
    
    def find_highest_z_coordinates(self, exclude_element="O"):
        cartesian_positions, fractional_positions = self.get_atomic_positions(exclude_element)
        sorted_cartesian = sorted(cartesian_positions, key=lambda atom: atom[3], reverse=True)
        sorted_fractional = sorted(fractional_positions, key=lambda atom: atom[3], reverse=True)
        
        highest_cartesian = sorted_cartesian[0]
        second_highest_cartesian = sorted_cartesian[1]
        highest_fractional = sorted_fractional[0]
        second_highest_fractional = sorted_fractional[1]
        
        z_distance_cartesian = abs(highest_cartesian[3] - second_highest_cartesian[3])
        z_distance_fractional = abs(highest_fractional[3] - second_highest_fractional[3])
        
        return {
            "highest": {
                "symbol": highest_cartesian[0],
                "cartesian": highest_cartesian[1:],
                "fractional": highest_fractional[1:]
            },
            "second_highest": {
                "symbol": second_highest_cartesian[0],
                "cartesian": second_highest_cartesian[1:],
                "fractional": second_highest_fractional[1:]
            },
            "z_distances": {
                "cartesian": z_distance_cartesian,
                "fractional": z_distance_fractional
            }
        }


class SurfaceEnergyCalculator:
    def __init__(self, ry_to_jm2=217.9863):
        self.ry_to_jm2 = ry_to_jm2
    def surf(self, data, bulk, N, area):
        data = np.array(data)
        area = np.array(area)
        surface_energies_ry = (data - N * bulk) / (2 * area)
        surface_energies_jm2 = surface_energies_ry * self.ry_to_jm2
        return surface_energies_jm2

class AdsorptionCalculator:
    def __init__(self, Eadsorbate, ry_to_eV=13.605698066, kBT=0.0258519):
        self.Eadsorbate = Eadsorbate
        self.ry_to_eV = ry_to_eV
        self.kBT = kBT

    def Eadsorption(self, Esys, Eadsorbant):
        Esys = np.array(Esys)
        return (Esys - (Eadsorbant + self.Eadsorbate)) * self.ry_to_eV

    def desorption_time(self, t, Esys, Eadsorbant):
        Eads = self.Eadsorption(Esys, Eadsorbant)
        return t * np.exp(-1 * Eads / self.kBT)


#############################################################################
# Simulated XRD from QE Input
#############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator


class SimulatedXRD:

    def __init__(self,
                 wavelength="CuKa",
                 fwhm=0.20):

        self.calc = XRDCalculator(wavelength=wavelength)
        self.fwhm = fwhm

        self.structures = []      # (path,label,structure)
        self.last_fig = None

    # ==================================================
    # Load structures
    # ==================================================
    def add_structure(self,
                      *filepaths,
                      labels=None,
                      file_format="espresso-in"):

        if labels is not None and len(labels) != len(filepaths):
            raise ValueError("labels must match filepaths")

        for i, path in enumerate(filepaths):

            atoms = read(path, format=file_format)

            structure = AseAtomsAdaptor.get_structure(atoms)

            label = labels[i] if labels else os.path.basename(path)

            self.structures.append(
                (path, label, structure)
            )

    # ==================================================
    # Simulate one pattern
    # ==================================================
    def _simulate_pattern(self,
                          structure,
                          x_min,
                          x_max,
                          npoints):

        pattern = self.calc.get_pattern(
            structure,
            scaled=True
        )

        x = np.linspace(x_min, x_max, npoints)

        y = np.zeros_like(x)

        sigma = self.fwhm / 2.355

        for pos, inten in zip(pattern.x, pattern.y):

            y += inten * np.exp(
                -(x-pos)**2/(2*sigma**2)
            )

        if np.max(y) > 0:
            y /= np.max(y)

        return x, y

    # ==================================================
    # Plot
    # ==================================================
    def plot(self,
             xlim=(10,80),
             zoom_xlim=None,
             separation=1.2,
             npoints=5000):

        if len(self.structures) == 0:
            raise RuntimeError("No structures loaded.")

        fig = plt.figure(figsize=(8,5))

        if zoom_xlim is None:

            ax = fig.add_subplot(111)

            offset = 0

            for _, label, structure in self.structures:

                x, y = self._simulate_pattern(
                    structure,
                    xlim[0],
                    xlim[1],
                    npoints
                )

                ax.plot(
                    x,
                    y+offset,
                    lw=2
                )

                ax.text(
                    xlim[0]+0.5,
                    offset+0.85,
                    label,
                    fontsize=11
                )

                offset += separation

            ax.set_xlim(xlim)
            ax.set_ylim(-0.1, offset)

            ax.set_xlabel(r"2$\theta$ (°)", fontsize=14)
            ax.set_ylabel("Intensity (a.u.)", fontsize=14)

            ax.set_yticks([])

        else:

            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[4,1],
                wspace=0.05
            )

            ax = fig.add_subplot(gs[0])
            ax_zoom = fig.add_subplot(gs[1])

            offset = 0

            for _, label, structure in self.structures:

                x, y = self._simulate_pattern(
                    structure,
                    xlim[0],
                    xlim[1],
                    npoints
                )

                ax.plot(
                    x,
                    y+offset,
                    lw=2
                )

                ax_zoom.plot(
                    x,
                    y+offset,
                    lw=2
                )

                ax.text(
                    xlim[0]+0.5,
                    offset+0.85,
                    label,
                    fontsize=11
                )

                offset += separation

            ax.set_xlim(xlim)
            ax_zoom.set_xlim(zoom_xlim)

            ax.set_ylim(-0.1, offset)
            ax_zoom.set_ylim(-0.1, offset)

            ax.set_xlabel(r"2$\theta$ (°)", fontsize=14)
            ax_zoom.set_xlabel(r"2$\theta$ (°)", fontsize=14)

            ax.set_ylabel("Intensity (a.u.)", fontsize=14)

            ax.set_yticks([])
            ax_zoom.set_yticks([])

        plt.tight_layout()

        self.last_fig = fig

        plt.show()

    # ==================================================
    # Save
    # ==================================================
    def save(self,
             filename,
             dpi=500):

        if self.last_fig is None:
            raise RuntimeError("Run plot() first.")

        self.last_fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches="tight"
        )

#xrd = SimulatedXRD(
#    wavelength="CuKa",
#    fwhm=0.2
#)

#xrd.add_structure(
#    "/media/mse/backup/Back_up_wildfly/Analysis/Screening/PDOS/scf_CPB_cubic.in",
#    "/media/mse/backup/Back_up_wildfly/Analysis/Screening/PDOS/scf_CPB_ortho.in",
#    "/media/mse/backup/Back_up_wildfly/Analysis/Screening/PDOS/scf_CPB_tetra.in",
#    labels=[
#        "CsPbBr$_3$ Cubic",
#        "CsPbBr$_3$ Orthorhombic",
#        "CsPbBr$_3$ Tetragonal"
#    ]
#)

#xrd.plot(
#    xlim=(10,40),
#    zoom_xlim=(14,17),
#    separation=1.3
#)


import os
import copy
import numpy as np

from ase.io import read, write


class QEManipulator:

    def __init__(self, input_file):

        self.input_file = input_file
        self.atoms = read(input_file, format="espresso-in")

    # ===========================================
    # Cell
    # ===========================================

    def get_cell(self):
        return self.atoms.get_cell()

    def get_positions(self):
        return self.atoms.get_positions()

    def get_fractional_positions(self):
        return self.atoms.get_scaled_positions()

    # ===========================================
    # Lattice manipulation
    # ===========================================

    def scale_lattice(self, scale):

        atoms = self.atoms.copy()

        if np.isscalar(scale):
            scale = np.array([scale, scale, scale])

        cell = atoms.get_cell().array
        new_cell = cell * scale[:, None]

        atoms.set_cell(
            new_cell,
            scale_atoms=True
        )

        self.atoms = atoms

    def deform(self, matrix):

        atoms = self.atoms.copy()

        cell = atoms.get_cell().array
        pos = atoms.get_positions()

        atoms.set_cell(
            matrix @ cell,
            scale_atoms=False
        )

        atoms.set_positions(
            (matrix @ pos.T).T
        )

        self.atoms = atoms

    # ===========================================
    # Atomic manipulation
    # ===========================================

    def translate(self, vector):

        atoms = self.atoms.copy()

        atoms.translate(vector)

        self.atoms = atoms

    def wrap(self):

        self.atoms.wrap()

    # ===========================================
    # Save
    # ===========================================

    def save(self, filename, pseudopotentials):
    
        write(
            filename,
            self.atoms,
            format="espresso-in",
            pseudopotentials=pseudopotentials
        )

    # ===========================================
    # Automatic strain scan
    # ===========================================

    def strain_scan(
        self,
        scales,
        output_dir="strain_scan"
    ):

        os.makedirs(output_dir, exist_ok=True)

        original = self.atoms.copy()

        for s in scales:

            self.atoms = original.copy()

            self.scale_lattice(s)

            folder = os.path.join(
                output_dir,
                f"scale_{s:.3f}"
            )

            os.makedirs(folder, exist_ok=True)

            self.save(
                os.path.join(
                    folder,
                    "scf.in"
                )
            )

        self.atoms = original

#F = np.array([
#    [1.03,0,0],
#    [0,1.03,0],
#    [0,0,1.03]
#])

#qe = QEManipulator("/media/mse/backup/Back_up_wildfly/Analysis/Screening/PDOS/scf_CPB_cubic.in")

#qe.deform(F)

#qe.save("/media/mse/backup/Back_up_wildfly/Analysis/Screening/other/scf_CPB_cubic3.in",
#    pseudopotentials={
#        "Cs":"Cs.upf",
#        "Pb":"Pb.upf",
#        "Br":"Br.upf"
#    }
#)

#########################################################################################

import numpy as np
import matplotlib.pyplot as plt
import xraydb


class AtomicScatteringPlot:
    def __init__(
        self,
        factor_type="f0",
        color_mode="default",
        colors=None,
        cmap="rainbow",
        linewidth=2
    ):
        """
        factor_type options:
            "f0"       -> normal atomic scattering factor vs q = sin(theta)/lambda
            "f1"       -> real anomalous scattering factor vs energy
            "f2"       -> imaginary anomalous scattering factor vs energy
            "f1_f2"    -> plot both f1 and f2 vs energy

        color_mode options:
            "default"  -> matplotlib default colors
            "colors"   -> use specific colors from colors list
            "gradient" -> use matplotlib colormap gradient

        Example colormaps:
            "rainbow", "viridis", "plasma", "turbo", "coolwarm"
        """

        self.factor_type = factor_type
        self.color_mode = color_mode
        self.colors = colors
        self.cmap = cmap
        self.linewidth = linewidth

        self.atoms = []  # list of (symbol, label)
        self.last_fig = None

        plt.rcParams.update({
            "font.family": "Times New Roman",
        })

    # =========================
    # --- ADD ATOMS ---
    # =========================
    def add_atoms(self, *symbols, labels=None):
        """
        Add one or more atoms.

        Example:
            plotter.add_atoms("Pb", "Br", "Cs", "Sn")

        Or with custom labels:
            plotter.add_atoms("Pb", "Br", labels=["Lead", "Bromine"])
        """

        if labels is not None and len(labels) != len(symbols):
            raise ValueError("Labels must match number of atoms")

        for i, symbol in enumerate(symbols):
            label = labels[i] if labels is not None else symbol
            self.atoms.append((symbol, label))

    # =========================
    # --- COLOR HANDLING ---
    # =========================
    def _get_plot_colors(self, n_atoms, color_mode=None, colors=None, cmap=None):
        color_mode = color_mode if color_mode is not None else self.color_mode
        colors = colors if colors is not None else self.colors
        cmap = cmap if cmap is not None else self.cmap

        if color_mode is None:
            color_mode = "default"

        color_mode = color_mode.lower()

        if color_mode == "default":
            return [None] * n_atoms

        if color_mode == "colors":
            if colors is None:
                raise ValueError(
                    "color_mode='colors' requires a colors list, "
                    "for example colors=['red', 'blue', 'green']"
                )

            if len(colors) != n_atoms:
                raise ValueError(
                    f"Number of colors ({len(colors)}) must match number of atoms ({n_atoms})"
                )

            return colors

        if color_mode == "gradient":
            colormap = plt.get_cmap(cmap)

            if n_atoms == 1:
                return [colormap(0.5)]

            return [colormap(i / (n_atoms - 1)) for i in range(n_atoms)]

        raise ValueError("color_mode must be 'default', 'colors', or 'gradient'")

    # =========================
    # --- XRAYDB CALCULATIONS ---
    # =========================
    def _calculate_f0(self, symbol, q):
        """
        f0 is calculated as a function of q = sin(theta) / lambda.
        """

        try:
            return np.asarray(xraydb.f0(symbol, q), dtype=float)
        except Exception:
            return np.array([xraydb.f0(symbol, value) for value in q], dtype=float)

    def _calculate_f1(self, symbol, energy):
        """
        f1 is calculated as a function of energy in eV.
        """

        try:
            return np.asarray(xraydb.f1_chantler(symbol, energy), dtype=float)
        except Exception:
            return np.array([xraydb.f1_chantler(symbol, value) for value in energy], dtype=float)

    def _calculate_f2(self, symbol, energy):
        """
        f2 is calculated as a function of energy in eV.
        """

        try:
            return np.asarray(xraydb.f2_chantler(symbol, energy), dtype=float)
        except Exception:
            return np.array([xraydb.f2_chantler(symbol, value) for value in energy], dtype=float)

    # =========================
    # --- PLOT ---
    # =========================
    def plot(
        self,
        x_range=None,
        n_points=500,
        factor_type=None,
        color_mode=None,
        colors=None,
        cmap=None,
        figsize=(7, 5),
        show=True
    ):
        """
        Plot atomic scattering factor.

        For factor_type="f0":
            x_range is q range, usually q = sin(theta) / lambda
            default: x_range=(0, 6)

        For factor_type="f1", "f2", or "f1_f2":
            x_range is energy range in eV
            default: x_range=(1000, 30000)
        """

        if len(self.atoms) == 0:
            raise RuntimeError("No atoms added. Use add_atoms() first.")

        factor_type = factor_type if factor_type is not None else self.factor_type
        factor_type = factor_type.lower()

        if factor_type == "f0":
            if x_range is None:
                x_range = (0, 6)
            xlabel = r"$q = \sin(\theta) / \lambda$"
            ylabel = r"$f_0$"
        elif factor_type in ["f1", "f2", "f1_f2"]:
            if x_range is None:
                x_range = (1000, 30000)
            xlabel = "Energy (eV)"
            ylabel = "Atomic scattering factor"
        else:
            raise ValueError("factor_type must be 'f0', 'f1', 'f2', or 'f1_f2'")

        x = np.linspace(x_range[0], x_range[1], n_points)

        plot_colors = self._get_plot_colors(
            len(self.atoms),
            color_mode=color_mode,
            colors=colors,
            cmap=cmap
        )

        fig, ax = plt.subplots(figsize=figsize)

        for i, (symbol, label) in enumerate(self.atoms):
            color = plot_colors[i]

            if factor_type == "f0":
                y = self._calculate_f0(symbol, x)
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=self.linewidth
                )

            elif factor_type == "f1":
                y = self._calculate_f1(symbol, x)
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=self.linewidth
                )

            elif factor_type == "f2":
                y = self._calculate_f2(symbol, x)
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=self.linewidth
                )

            elif factor_type == "f1_f2":
                f1 = self._calculate_f1(symbol, x)
                f2 = self._calculate_f2(symbol, x)

                ax.plot(
                    x,
                    f1,
                    label=f"{label} f1",
                    color=color,
                    linewidth=self.linewidth
                )

                ax.plot(
                    x,
                    f2,
                    label=f"{label} f2",
                    color=color,
                    linestyle="--",
                    linewidth=self.linewidth
                )

        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        self.last_fig = fig

        if show:
            plt.show()

    # =========================
    # --- SAVE ---
    # =========================
    def save_plot(self, filename="atomic_scattering_factor.png", dpi=500):
        if self.last_fig is None:
            raise RuntimeError("Call plot() first")

        self.last_fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        print(f"Saved: {filename}")

#plotter = AtomicScatteringPlot(
#    factor_type="f0",
#    color_mode="gradient",
#    cmap="rainbow"
#)

#plotter.add_atoms("Cs", "Br", "Pb")

#plotter.plot(
#    x_range=(0, 6),
#    n_points=500
#)
###########################################################################################

import numpy as np
import xraydb


class FhklCalculator:
    def __init__(
        self,
        wavelength=1.5406,
        energy=None,
        use_anomalous=False
    ):
        self.wavelength = wavelength
        self.energy = energy
        self.use_anomalous = use_anomalous

        self.sites = []

        self.lattice = {
            "a": None,
            "b": None,
            "c": None,
            "alpha": 90,
            "beta": 90,
            "gamma": 90
        }

    # =========================
    # --- CLEAR ATOMS ---
    # =========================
    def clear_atoms(self):
        self.sites = []

    # =========================
    # --- SET LATTICE ---
    # =========================
    def set_lattice(
        self,
        a,
        b=None,
        c=None,
        alpha=90,
        beta=90,
        gamma=90,
        clear_atoms=True
    ):
        """
        Set unit cell parameters.

        clear_atoms=True prevents atoms from previous notebook runs/cells
        from staying inside the calculator.
        """

        if clear_atoms:
            self.clear_atoms()

        if b is None:
            b = a

        if c is None:
            c = a

        self.lattice = {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma
        }

    # =========================
    # --- ADD ATOMIC SITE ---
    # =========================
    def add_atom(
        self,
        symbol,
        x,
        y,
        z,
        occupancy=1.0,
        B=0.0,
        label=None,
        skip_duplicate=True
    ):
        """
        Add one atomic site.

        skip_duplicate=True prevents the same atom from being added again
        when you re-run a notebook cell.
        """

        new_site = {
            "symbol": symbol,
            "x": x,
            "y": y,
            "z": z,
            "occupancy": occupancy,
            "B": B,
            "label": label if label is not None else symbol
        }

        if skip_duplicate:
            for site in self.sites:
                same_atom = (
                    site["symbol"] == symbol and
                    np.isclose(site["x"], x) and
                    np.isclose(site["y"], y) and
                    np.isclose(site["z"], z) and
                    np.isclose(site["occupancy"], occupancy) and
                    np.isclose(site["B"], B)
                )

                if same_atom:
                    return

        self.sites.append(new_site)

    # =========================
    # --- NUMBER OF ATOMS ---
    # =========================
    def number_of_atoms(self):
        return len(self.sites)

    # =========================
    # --- CALCULATE d_hkl ---
    # =========================
    def d_spacing(self, h, k, l):
        a = self.lattice["a"]
        b = self.lattice["b"]
        c = self.lattice["c"]
        alpha = self.lattice["alpha"]
        beta = self.lattice["beta"]
        gamma = self.lattice["gamma"]

        if a is None:
            raise RuntimeError("Set lattice first using set_lattice().")

        if alpha != 90 or beta != 90 or gamma != 90:
            raise NotImplementedError(
                "This simple d_spacing method supports only alpha=beta=gamma=90."
            )

        inv_d2 = (h ** 2 / a ** 2) + (k ** 2 / b ** 2) + (l ** 2 / c ** 2)

        if inv_d2 == 0:
            raise ValueError("(000) reflection is not valid.")

        return 1 / np.sqrt(inv_d2)

    # =========================
    # --- CALCULATE SCATTERING VECTOR ---
    # =========================
    def q_from_hkl(self, h, k, l):
        d = self.d_spacing(h, k, l)
        return 1 / (2 * d)

    # =========================
    # --- ATOMIC SCATTERING FACTOR ---
    # =========================
    def atomic_scattering_factor(self, symbol, h, k, l):
        q = self.q_from_hkl(h, k, l)

        f0 = xraydb.f0(symbol, q)

        if not self.use_anomalous:
            return f0

        if self.energy is None:
            raise ValueError("energy must be set when use_anomalous=True")

        f1 = xraydb.f1_chantler(symbol, self.energy)
        f2 = xraydb.f2_chantler(symbol, self.energy)

        return f0 + f1 + 1j * f2

    # =========================
    # --- CALCULATE F_hkl ---
    # =========================
    def calculate_Fhkl(self, h, k, l, verbose=False):
        if len(self.sites) == 0:
            raise RuntimeError("No atoms added. Use add_atom() first.")

        Fhkl = 0 + 0j

        for site in self.sites:
            symbol = site["symbol"]
            x = site["x"]
            y = site["y"]
            z = site["z"]
            occupancy = site["occupancy"]
            B = site["B"]

            f = self.atomic_scattering_factor(symbol, h, k, l)
            phase = np.exp(2j * np.pi * (h * x + k * y + l * z))

            q = self.q_from_hkl(h, k, l)
            debye_waller = np.exp(-B * q ** 2)

            contribution = occupancy * f * phase * debye_waller
            Fhkl += contribution

            if verbose:
                print(
                    f"{site['label']:8s} "
                    f"f={f:.4f} "
                    f"phase={phase:.4f} "
                    f"contribution={contribution:.4f}"
                )

        return Fhkl

    # =========================
    # --- INTENSITY ---
    # =========================
    def intensity(self, h, k, l):
        Fhkl = self.calculate_Fhkl(h, k, l)
        return np.abs(Fhkl) ** 2

    # =========================
    # --- MULTIPLE HKL ---
    # =========================
    def calculate_many(self, hkls):
        results = []

        for h, k, l in hkls:
            d = self.d_spacing(h, k, l)
            q = self.q_from_hkl(h, k, l)
            Fhkl = self.calculate_Fhkl(h, k, l)
            I = np.abs(Fhkl) ** 2

            results.append({
                "h": h,
                "k": k,
                "l": l,
                "d": d,
                "q": q,
                "Fhkl": Fhkl,
                "F_real": Fhkl.real,
                "F_imag": Fhkl.imag,
                "F_abs": np.abs(Fhkl),
                "intensity": I
            })

        return results

    # =========================
    # --- PRINT TABLE ---
    # =========================
    def print_table(self, hkls):
        results = self.calculate_many(hkls)

        print(
            f"{'hkl':>10s} "
            f"{'d(A)':>10s} "
            f"{'q':>10s} "
            f"{'Re(F)':>12s} "
            f"{'Im(F)':>12s} "
            f"{'|F|':>12s} "
            f"{'I=|F|^2':>12s}"
        )

        for r in results:
            hkl = f"({r['h']} {r['k']} {r['l']})"

            print(
                f"{hkl:>10s} "
                f"{r['d']:10.4f} "
                f"{r['q']:10.4f} "
                f"{r['F_real']:12.4f} "
                f"{r['F_imag']:12.4f} "
                f"{r['F_abs']:12.4f} "
                f"{r['intensity']:12.4f}"
            )

#calc = FhklCalculator(
#    wavelength=1.5406,
#    use_anomalous=False
#)

#calc.set_lattice(a=5.87)

#calc.add_atom("Cs", 0.0, 0.0, 0.0)
#calc.add_atom("Pb", 0.5, 0.5, 0.5)
#calc.add_atom("Br", 0.5, 0.5, 0.0)
#calc.add_atom("Br", 0.5, 0.0, 0.5)
#calc.add_atom("Br", 0.0, 0.5, 0.5)


#F100 = calc.calculate_Fhkl(1, 0, 0)
#I100 = calc.intensity(1, 0, 0)
#F110 = calc.calculate_Fhkl(1, 1, 0)
#I110 = calc.intensity(1,1, 0)
#print("F100 =", F100)
#print("I100 =", I100)
#print("F110 =", F110)
#print("I110 =", I110)

######################################################################################################
# Band Structure Projection within PDOS
######################################################################################################

import re
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


class ProjectedBandPlotter:
    """
    OOP projected-band plotter for a Qepy ProjwfcXML object.

    Supports:
    - total atom projections
    - total s/p/d/f orbital projections
    - atom + orbital projections
    - QE wfc-index selections, such as 2p or 3d
    - multiple selected components
    - continuous background band structure
    """

    orbital_labels = {
        0: "s",
        1: "p",
        2: "d",
        3: "f",
    }

    orbital_to_l = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
    }

    def __init__(self, band, path_kpoints=None):
        self.band = band
        self.path_kpoints = path_kpoints

    def print_projected_states(self):
        """Print available QE projection states."""
        print("Index | Atom | QE wfc index | Orbital")

        for state in self.band.states:
            orbital = self.orbital_labels.get(
                state["l"],
                str(state["l"]),
            )

            print(
                f"{state['istate'] - 1:5d} | "
                f"{state['atype']:4s} | "
                f"{state['wfc']:12d} | "
                f"{orbital}"
            )

    def select_states(
        self,
        atoms=None,
        orbitals=None,
        wfc=None,
    ):
        """
        Select QE projected states.

        Examples
        --------
        atoms=["Te"]
        atoms=["Te"], orbitals=["p"]
        atoms=["Te"], orbitals=["2p"]
        atoms=["Te"], orbitals=["p"], wfc=2
        """
        if atoms is None:
            atoms = ["all"]
        elif isinstance(atoms, str):
            atoms = [atoms]

        if orbitals is None:
            orbitals = ["s", "p", "d", "f"]
        elif isinstance(orbitals, str):
            orbitals = [orbitals]

        selected_states = []

        for state in self.band.states:
            atom_matches = (
                "all" in atoms
                or state["atype"] in atoms
            )

            if not atom_matches:
                continue

            for orbital in orbitals:
                match = re.fullmatch(
                    r"(?:(\d+))?([spdf])",
                    orbital.lower(),
                )

                if match is None:
                    raise ValueError(
                        f"Invalid orbital: {orbital}. "
                        "Use s, p, d, f, 2p, 3d, etc."
                    )

                requested_wfc = (
                    int(match.group(1))
                    if match.group(1) is not None
                    else None
                )

                requested_l = self.orbital_to_l[
                    match.group(2)
                ]

                if state["l"] != requested_l:
                    continue

                if (
                    requested_wfc is not None
                    and state["wfc"] != requested_wfc
                ):
                    continue

                if wfc is not None and state["wfc"] != wfc:
                    continue

                selected_states.append(
                    state["istate"] - 1
                )
                break

        selected_states = sorted(set(selected_states))

        if not selected_states:
            raise ValueError(
                "No matching QE projection states. "
                "Run plotter.print_projected_states() first."
            )

        return selected_states

    def _resolve_selections(self, selections):
        """Convert flexible selection dictionaries to QE state-index lists."""
        return {
            label: (
                self.select_states(
                    atoms=selection.get("atoms"),
                    orbitals=selection.get("orbitals"),
                    wfc=selection.get("wfc"),
                )
                if isinstance(selection, dict)
                else selection
            )
            for label, selection in selections.items()
        }

    def _get_kpoint_distance(self):
        kpoints = np.asarray(
            self.band.kpoints[:self.band.nkpoints],
            dtype=float,
        )

        return np.concatenate(
            [
                [0.0],
                np.cumsum(
                    np.linalg.norm(
                        np.diff(kpoints, axis=0),
                        axis=1,
                    )
                ),
            ]
        )

    def _add_path_labels(self, ax, kpoint_distance, path_kpoints):
        if path_kpoints is None:
            return

        if hasattr(path_kpoints, "get_indexes"):
            path_kpoints = path_kpoints.get_indexes()

        ticks, tick_labels = zip(*path_kpoints)

        ax.set_xticks(
            [kpoint_distance[index] for index in ticks]
        )
        ax.set_xticklabels(tick_labels)

        for index in ticks:
            ax.axvline(
                kpoint_distance[index],
                color="gray",
                linewidth=0.7,
                zorder=0,
            )

    def plot(
        self,
        ax,
        selections,
        path_kpoints=None,
        colors=None,
        bandmin=0,
        bandmax=None,
        min_weight=0.01,
        base_color="lightgray",
        base_linewidth=0.7,
        max_linewidth=5.0,
        alpha=1.0,
        y_offset=0.0,
        show_unprojected=False,
        show_full_bands=True,
        show_legend=True,
    ):
        """
        Plot projected bands as colored line segments.

        selections example:

        {
            "Cs": {"atoms": ["Cs"]},
            "Te p": {"atoms": ["Te"], "orbitals": ["p"]},
            "Cl 2p": {"atoms": ["Cl"], "orbitals": ["2p"]},
        }
        """
        band = self.band

        if path_kpoints is None:
            path_kpoints = self.path_kpoints

        if bandmax is None:
            bandmax = band.nbands

        if (
            bandmin < 0
            or bandmax > band.nbands
            or bandmin >= bandmax
        ):
            raise ValueError("Check bandmin and bandmax.")

        selections = self._resolve_selections(selections)
        labels = list(selections.keys())

        if not labels:
            raise ValueError("selections cannot be empty.")

        if colors is None:
            cmap = plt.get_cmap("tab20")

            colors = [
                cmap(index % 20)
                for index in range(len(labels))
            ]

        if len(colors) < len(labels):
            raise ValueError(
                "Provide one color for every selection."
            )

        colors = colors[:len(labels)]

        kpoint_distance = self._get_kpoint_distance()

        self._add_path_labels(
            ax,
            kpoint_distance,
            path_kpoints,
        )

        ax.axhline(
            0,
            color="gray",
            linewidth=0.8,
            zorder=0,
        )

        all_weights = [
            band.get_weights(
                selected_orbitals=state_indices
            )
            for state_indices in selections.values()
        ]

        # Non-spin-polarized or full-spinor calculation.
        if band.spin_components in (1, 4):
            eigen_sets = [band.eigen]
            weight_sets = [all_weights]

        # Collinear spin-polarized calculation.
        elif band.spin_components == 2:
            eigen_sets = [band.eigen1, band.eigen2]

            weight_sets = [
                [weight[0] for weight in all_weights],
                [weight[1] for weight in all_weights],
            ]

        else:
            raise ValueError(
                f"Unsupported spin_components: "
                f"{band.spin_components}"
            )

        # Continuous full background band structure.
        if show_full_bands:
            for eigenvalues in eigen_sets:
                for iband in range(bandmin, bandmax):
                    ax.plot(
                        kpoint_distance,
                        eigenvalues[:, iband] + y_offset,
                        color=base_color,
                        linewidth=base_linewidth,
                        zorder=1,
                    )

        # Colored projected line segments.
        for spin_index, eigenvalues in enumerate(eigen_sets):
            component_weights = weight_sets[spin_index]

            for iband in range(bandmin, bandmax):
                energy = eigenvalues[:, iband] + y_offset

                weights = np.array(
                    [
                        component_weight[:, iband]
                        for component_weight in component_weights
                    ]
                )

                total_weight = np.sum(weights, axis=0)
                dominant_component = np.argmax(
                    weights,
                    axis=0,
                )

                points = np.column_stack(
                    [kpoint_distance, energy]
                )

                segments = np.stack(
                    [points[:-1], points[1:]],
                    axis=1,
                )

                segment_weight = 0.5 * (
                    total_weight[:-1]
                    + total_weight[1:]
                )

                segment_component = dominant_component[:-1]

                if show_unprojected:
                    keep = np.ones(
                        len(segment_weight),
                        dtype=bool,
                    )

                    segment_colors = [
                        (
                            colors[component]
                            if weight >= min_weight
                            else base_color
                        )
                        for component, weight in zip(
                            segment_component,
                            segment_weight,
                        )
                    ]

                else:
                    keep = segment_weight >= min_weight

                    segment_colors = [
                        colors[component]
                        for component in segment_component[keep]
                    ]

                segments = segments[keep]
                segment_weight = segment_weight[keep]

                if len(segments) == 0:
                    continue

                line_widths = (
                    base_linewidth
                    + (max_linewidth - base_linewidth)
                    * np.clip(segment_weight, 0, 1)
                )

                lines = LineCollection(
                    segments,
                    colors=segment_colors,
                    linewidths=line_widths,
                    alpha=alpha,
                    zorder=2,
                )

                ax.add_collection(lines)

        ax.set_xlim(
            kpoint_distance.min(),
            kpoint_distance.max(),
        )

        ax.set_ylabel("E - $E_F$ (eV)")

        if show_legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=3,
                    label=label,
                )
                for label, color in zip(labels, colors)
            ]

            if show_unprojected:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=base_color,
                        linewidth=2,
                        label="Low selected weight",
                    )
                )

            ax.legend(
                handles=handles,
                title="Projection",
                frameon=False,
                loc="best",
            )

        return ax

#print_projected_states(band)

#fig, ax = plt.subplots(figsize=(4, 6))

#plot_multicomponent_projected_bands(
#    band,
#    ax,
#    selections={
    #    "Cs total": {
    #        "atoms": ["Cs"],
    #    },
#        "Te p": {
#            "atoms": ["Te"],
#            "orbitals": ["p"],
#        },
#        "Cl p": {
#            "atoms": ["Cl"],
#            "orbitals": ["p"],
#        },
  #      "Cs d": {
   #         "atoms": ["Cs"],
   #         "orbitals": ["d"],
   #     },
#    },
#    path_kpoints=path_kpoints,
#    colors=[
#        "tab:purple",
     #   "tab:orange",
#        "tab:green",
        #"tab:red",
#    ],
#    min_weight=0.02,
#    max_linewidth=5.0,
#    show_unprojected=False,
#)

#ax.set_ylim(-4, 2)
#plt.show()


###########################
# Band Gap Analyzer
###########################

import numpy as np


class BandGapAnalyzer:
    """Calculate direct and indirect band gaps from a ProjwfcXML band object."""

    def __init__(
        self,
        band,
        path_kpoints=None,
        n_valence_bands=None,
        zero_tolerance=1e-5,
    ):
        self.band = band
        self.path_kpoints = path_kpoints
        self.n_valence_bands = n_valence_bands
        self.zero_tolerance = zero_tolerance
        self.results = None

    def _kpoint_label(self, k_index, kpoint):
        if self.path_kpoints is not None:
            path = self.path_kpoints

            if hasattr(path, "get_indexes"):
                path = path.get_indexes()

            label_map = {
                index: label
                for index, label in path
            }

            if k_index in label_map:
                return label_map[k_index]

        return (
            f"k-index {k_index} "
            f"({kpoint[0]:.3f}, {kpoint[1]:.3f}, {kpoint[2]:.3f})"
        )

    def _analyze_eigenvalues(self, eigenvalues, kpoints):
        n_valence_bands = self.n_valence_bands

        # Automatic valence-band detection for semiconductors.
        if n_valence_bands is None:
            occupied_bands = np.where(
                np.nanmax(eigenvalues, axis=0)
                <= self.zero_tolerance
            )[0]

            if len(occupied_bands) == 0:
                raise ValueError(
                    "Could not identify occupied bands automatically. "
                    "Set n_valence_bands manually."
                )

            n_valence_bands = occupied_bands[-1] + 1

        if (
            n_valence_bands < 1
            or n_valence_bands >= eigenvalues.shape[1]
        ):
            raise ValueError("Check n_valence_bands.")

        valence = eigenvalues[:, :n_valence_bands]
        conduction = eigenvalues[:, n_valence_bands:]

        # Valence-band maximum.
        vbm_flat = np.nanargmax(valence)
        vbm_kindex, vbm_band = np.unravel_index(
            vbm_flat,
            valence.shape,
        )
        vbm_energy = valence[vbm_kindex, vbm_band]

        # Conduction-band minimum.
        cbm_flat = np.nanargmin(conduction)
        cbm_kindex, cbm_band_relative = np.unravel_index(
            cbm_flat,
            conduction.shape,
        )
        cbm_band = cbm_band_relative + n_valence_bands
        cbm_energy = conduction[cbm_kindex, cbm_band_relative]

        indirect_gap = cbm_energy - vbm_energy

        # Lowest direct transition at the same k point.
        local_vbm = np.nanmax(valence, axis=1)
        local_cbm = np.nanmin(conduction, axis=1)
        direct_gaps = local_cbm - local_vbm

        direct_kindex = np.nanargmin(direct_gaps)
        direct_gap = direct_gaps[direct_kindex]

        return {
            "n_valence_bands": n_valence_bands,
            "vbm_energy_eV": vbm_energy,
            "vbm_band_index": vbm_band,
            "vbm_kindex": vbm_kindex,
            "vbm_kpoint": kpoints[vbm_kindex],
            "vbm_label": self._kpoint_label(
                vbm_kindex,
                kpoints[vbm_kindex],
            ),
            "cbm_energy_eV": cbm_energy,
            "cbm_band_index": cbm_band,
            "cbm_kindex": cbm_kindex,
            "cbm_kpoint": kpoints[cbm_kindex],
            "cbm_label": self._kpoint_label(
                cbm_kindex,
                kpoints[cbm_kindex],
            ),
            "indirect_gap_eV": indirect_gap,
            "direct_gap_eV": direct_gap,
            "direct_kindex": direct_kindex,
            "direct_kpoint": kpoints[direct_kindex],
            "direct_label": self._kpoint_label(
                direct_kindex,
                kpoints[direct_kindex],
            ),
            "is_direct": vbm_kindex == cbm_kindex,
        }

    def calculate(self):
        """Calculate the direct and indirect gaps."""
        kpoints = np.asarray(
            self.band.kpoints[:self.band.nkpoints],
            dtype=float,
        )

        # Non-spin-polarized or spinor calculation.
        if self.band.spin_components in (1, 4):
            self.results = {
                "total": self._analyze_eigenvalues(
                    self.band.eigen,
                    kpoints,
                )
            }

        # Collinear spin-polarized calculation.
        elif self.band.spin_components == 2:
            self.results = {
                "spin up": self._analyze_eigenvalues(
                    self.band.eigen1,
                    kpoints,
                ),
                "spin down": self._analyze_eigenvalues(
                    self.band.eigen2,
                    kpoints,
                ),
            }

        else:
            raise ValueError(
                f"Unsupported spin_components: "
                f"{self.band.spin_components}"
            )

        return self.results

    def print_results(self):
        """Print direct/indirect band-gap results."""
        if self.results is None:
            self.calculate()

        for spin_label, result in self.results.items():
            print(f"\n{spin_label}")

            print(
                f"VBM = {result['vbm_energy_eV']:.4f} eV "
                f"at {result['vbm_label']} "
                f"(band {result['vbm_band_index']})"
            )

            print(
                f"CBM = {result['cbm_energy_eV']:.4f} eV "
                f"at {result['cbm_label']} "
                f"(band {result['cbm_band_index']})"
            )

            print(
                f"Indirect gap = "
                f"{result['indirect_gap_eV']:.4f} eV"
            )

            print(
                f"Minimum direct gap = "
                f"{result['direct_gap_eV']:.4f} eV "
                f"at {result['direct_label']}"
            )

            gap_type = (
                "direct"
                if result["is_direct"]
                else "indirect"
            )

            print(f"Fundamental gap: {gap_type}")

#gap_analyzer = BandGapAnalyzer(
#    band,
#    path_kpoints=path_kpoints,
#)

#gap_analyzer.calculate()
#gap_analyzer.print_results()