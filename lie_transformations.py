# Copyright (c) 2022, Colas Droin. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

""" This file contains functions used to compute, simulate and visualize Lie transformations for 
any multipole order."""

# ==================================================================================================
# --- Imports
# ==================================================================================================
import sympy as sym
from sympy import Array
import numpy as np
import plotly.graph_objects as go
import plotly.express as plx
from sklearn.decomposition import PCA
from ipywidgets import widgets
from plotly.subplots import make_subplots
import io
from PIL import Image
import moviepy.editor as mpy


# ==================================================================================================
# --- Plotly plotting functions and settings
# ==================================================================================================
# To visualize ellipse in a non-distorted way
plx.defaults.width = 600
plx.defaults.height = 600

# To generate animations
def plotly_fig2array(fig):
    # convert Plotly fig to  an array
    fig.update_layout(
        autosize=False,
        width=1280,
        height=720,
    )
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


# ==================================================================================================
# --- Functions to generate the Hamiltonian and the corresponding lie transformation
# ==================================================================================================
def poisson_brackets(f, g, X, P):
    """This function is used to compute the Poisson brackets between f and g, with respect to the
    variables in  X and P."""
    return sum([f.diff(xi) * g.diff(pi) - f.diff(pi) * g.diff(xi) for xi, pi in zip(X, P)])


def lie_transformation(f, g, X, P, max_order=4, verbose=False):
    """This function is used to compute the Lie transformation of f applied to g, truncated up to a
    given order."""
    # Special case for i = 0 as transform is always identity
    poisson_transform = g
    transform = poisson_transform
    for i in range(1, max_order, 1):
        poisson_transform = poisson_brackets(f, poisson_transform, X, P)
        transform += poisson_transform / sym.factorial(i)
        if poisson_transform == 0:
            if verbose:
                print("Exact transformation found at order", i)
            return sym.simplify(transform)
    if verbose:
        print("Exact transformation could not be found up to order", max_order)
    return sym.simplify(transform)


def lie_transformation_hamiltonian_4D(X, P, H, L, order=2, yoshida_coeff=1.0, verbose=False):
    """This function compute the Lie transformation of the Hamiltonian H for each of the 4 canonical
    variables in the transverse plane. Yoshida coefficients can be provided if the Hamiltonian is
    separated into kicks and drifts parts."""
    x_new, y_new, px_new, py_new = [
        lie_transformation(-L * yoshida_coeff * H, g, X, P, order, verbose=verbose)
        for g in X.tolist() + P.tolist()
    ]
    return x_new, px_new, y_new, py_new


def generate_hamiltonian(
    order_magnet,
    delta,
    X,
    P,
    skew=False,
    k=None,
    rho=None,
    approx=True,
    return_separate_hamiltonians=False,
):
    """This function is used to generate the Hamiltonian for a given order of magnet, with the
    possibility to include skew components.The Hamiltonian can be returned as the sum of drifts and
    kicks parts, or as a single expression.
    order -1 is drift
    order 0 is dipole
    order 1 is quadrupole
    order 2 is sextupole
    order 3 is octupole
    etc."""

    # Define symbols for magnet strength and bending radius if not provided
    if k is None:
        k = sym.Symbol("k" + "_" + str(order_magnet), real=True)
    if rho is None:
        rho = sym.Symbol("rho", real=True)

    # Take into account if skew components are present
    if skew:
        fact_norm = 0
        fact_skew = 1j
    else:
        fact_norm = 1
        fact_skew = 0

    # Drift is a special case
    if order_magnet == -1:
        if approx:
            Hd = (P[0] ** 2 + P[1] ** 2) / (2 * (1 + delta))
            Hk = 0
            H = Hd + Hk
        else:
            H = -(((1 + delta) ** 2 - px**2 - py**2) ** 0.5)
            if return_separate_hamiltonians:
                print("No separate hamiltonians available for exact drift")
                return sym.simplify(H)

    # Dipole is also a special case as Bx is not symmetrical with By, and k = 1/rho
    elif order_magnet == 0:
        Hd = (P[0] ** 2 + P[1] ** 2) / (2 * (1 + delta))
        Hk = x * delta / rho + (x**2) / (2 * rho**2)
        H = Hd + Hk

    # Multipoles
    else:
        Hd = (P[0] ** 2 + P[1] ** 2) / (2 * (1 + delta))
        Hk = (
            1
            / (1 + order_magnet)
            * sym.re((fact_norm * k + fact_skew * k) * (X[0] + 1j * X[1]) ** (order_magnet + 1))
        )
        H = Hd + Hk

    if return_separate_hamiltonians:
        return sym.simplify(Hd), sym.simplify(Hk)
    else:
        return sym.simplify(H)


# ==================================================================================================
# --- Functions to generate particles distributions
# ==================================================================================================
def generate_particle_distribution(std_u=10**-3, std_pu=10**-4, n_particles=500):
    """This function is used to generate a set of n_particles particles with given position and
    momenta."""

    # Define a set of particles with given position and momenta
    array_u = np.random.normal(0.0, std_u, size=n_particles)
    array_pu = np.random.normal(0.0, std_pu, size=n_particles)

    return array_u, array_pu


def get_twiss_and_emittance(array_u, array_pu):
    """This function is used to compute the emittance and Twiss parameters from a set of particles
    with given position and momenta."""
    # Compute the emittance
    std_u = array_u.std()
    std_pu = array_pu.std()
    cov_upu = np.nanmean((array_u - np.nanmean(array_u)) * (array_pu - np.nanmean(array_pu)))
    emittance = (std_u**2 * std_pu**2 - cov_upu**2) ** 0.5

    # Compute the Twiss parameters
    alpha = -cov_upu / emittance
    beta = std_u**2 / emittance
    gamma = std_pu**2 / emittance

    return emittance, alpha, beta, gamma


def get_contour_ellipse(array_u, array_pu, emittance, alpha, beta, gamma):
    """This function is used to compute the contour of the ellipse (representing the emittance) in
    the phase-space from a set of particles with given position and momenta."""
    # Compute the corresponding ellipse in the phase-spae
    u_ellipse = np.linspace(np.min(array_u), np.max(array_u), 100)
    p_ellipse = np.linspace(np.min(array_pu), np.max(array_pu), 100)
    u_ellipse, p_ellipse = np.meshgrid(u_ellipse, p_ellipse)
    ellipse = (
        gamma * u_ellipse**2
        + 2 * alpha * u_ellipse * p_ellipse
        + beta * p_ellipse**2
        - emittance
    )
    return u_ellipse, p_ellipse, ellipse


def get_color_according_to_PCA_projection(array_u, array_pu):
    """This function is used to compute a color for each particle according to its position in the
    phase-space, using a PCA projection."""

    # Define colors according to PCA projection in 1D
    pca = PCA(n_components=1)
    pca.fit(np.array([array_u, array_pu]).T)
    pca_projection = pca.transform(np.array([array_u, array_pu]).T)

    # Rescale pca projection to be between 0 and 1
    pca_projection = (pca_projection - np.min(pca_projection)) / (
        np.max(pca_projection) - np.min(pca_projection)
    )

    # Remove extra dimension and convert to list
    l_colors = list(np.squeeze(pca_projection))
    return l_colors


# ==================================================================================================
# --- Numerical integration functions
# ==================================================================================================
def simulate_transfer_truncated(
    array_u,
    array_pu,
    X,
    P,
    magnet_order=1,
    L=1.0,
    delta=0.0,
    k_n=0.1,
    max_order_lie_transform=10,
    y_map=False,
    array_u2=None,
    array_pu2=None,
):
    """This function is used to simulate the transfer of a set of particles with a truncated map
    obtained from a Lie transformation."""

    # Generate the hamiltonian and compute transformation
    H = generate_hamiltonian(magnet_order, delta, X, P, k=k_n)
    x_new, px_new, y_new, py_new = lie_transformation_hamiltonian_4D(
        X, P, H, L, order=max_order_lie_transform
    )

    # Get variables
    x, y = X
    px, py = P

    # Evaluate the result with lambdify
    # Case map is 1D (either x  or y, without coupling involved)
    if array_u2 is None and array_pu2 is None:
        if y_map:
            f = sym.lambdify((y, py), y_new)
            f_p = sym.lambdify((y, py), py_new)
        else:
            f = sym.lambdify((x, px), x_new)
            f_p = sym.lambdify((x, px), px_new)

        array_u_transformed = f(array_u, array_pu)
        array_pu_transformed = f_p(array_u, array_pu)
        return array_u_transformed, array_pu_transformed

    # Case map is 2D (with or without coupling involved)
    elif array_u2 is not None and array_pu2 is not None:
        f_x = sym.lambdify((x, px, y, py), x_new)
        f_px = sym.lambdify((x, px, y, py), px_new)
        f_y = sym.lambdify((x, px, y, py), y_new)
        f_py = sym.lambdify((x, px, y, py), py_new)
        array_x_transformed = f_x(array_u, array_pu, array_u2, array_pu2)
        array_px_transformed = f_px(array_u, array_pu, array_u2, array_pu2)
        array_y_transformed = f_y(array_u, array_pu, array_u2, array_pu2)
        array_py_transformed = f_py(array_u, array_pu, array_u2, array_pu2)
        return array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed

    else:
        raise ValueError("array_u2 and array_pu2 must be both None or both not None")


def return_exact_map_for_quadrupole(
    X, P, array_u, array_pu, L=1.0, delta=0.0, k_n=0.1, y_map=False
):
    """This function is used to return the exact (closed-form) map for a quadrupole."""
    # Get variables
    x, y = X
    px, py = P

    if y_map:
        y_new = sym.cosh(k_n**0.5 * L) * y + sym.sinh(k_n**0.5 * L) / k_n**0.5 * py
        py_new = (k_n**0.5) * sym.sinh(k_n**0.5 * L) * y + sym.cosh(k_n**0.5 * L) * py
        f = sym.lambdify((y, py), y_new)
        f_p = sym.lambdify((y, py), py_new)
    else:
        x_new = sym.cos(k_n**0.5 * L) * x + sym.sin(k_n**0.5 * L) / k_n**0.5 * px
        px_new = -(k_n**0.5) * sym.sin(k_n**0.5 * L) * x + sym.cos(k_n**0.5 * L) * px
        f = sym.lambdify((x, px), x_new)
        f_p = sym.lambdify((x, px), px_new)

    array_u_transformed = f(array_u, array_pu)
    array_pu_transformed = f_p(array_u, array_pu)
    return array_u_transformed, array_pu_transformed


def yoshida_coeff_calculator(order):
    """This function is used to compute the Yoshida coefficients of a Lie transformation for a
    given order."""

    S = [0.5, 1, 0.5]
    if order % 2 != 0:
        # print("Yoshida coefficients can only be computed for even orders.")
        return S

    for n in range(1, int(order / 2)):
        alpha = 2.0 ** (1.0 / (2 * n + 1))
        x0 = -alpha / (2.0 - alpha)
        x1 = 1 / (2.0 - alpha)
        TC = [i * x0 for i in S]
        TL = [i * x1 for i in S]
        T = []
        for i in TL[:-1]:
            T.append(i)
        T.append(TL[-1] + TC[0])
        for i in TC[1:-1]:
            T.append(i)
        T.append(TC[-1] + TL[0])
        for i in TL[1:]:
            T.append(i)
        S = T
    return S


def symplectic_analytical_integrator(order_integrator, Hd, Hk, X, P, L, max_order_lie_transform=20):
    """This function is used to compute the analytical, symplectic, maps representing the particle
    trajectory when going through a multipole (with the corresponding hamiltonian)."""

    S = yoshida_coeff_calculator(order_integrator)
    l_transforms = []
    for idx, coeff in enumerate(S):
        if idx % 2 == 0:
            x_new, px_new, y_new, py_new = lie_transformation_hamiltonian_4D(
                X, P, Hd, L, order=max_order_lie_transform, yoshida_coeff=coeff
            )
        else:
            x_new, px_new, y_new, py_new = lie_transformation_hamiltonian_4D(
                X, P, Hk, L, order=max_order_lie_transform, yoshida_coeff=coeff
            )
        l_transforms.append([x_new, px_new, y_new, py_new])
    return l_transforms


def symplectic_numerical_integrator(
    X, P, l_transforms, array_x=None, array_px=None, array_y=None, array_py=None
):
    """This function is used to integrate the particle trajectory when going through a multipole
    (with the corresponding hamiltonian), using a symplectic integrator."""

    # Evaluate the result of each successive transformation with lambdify
    array_x_transformed = np.copy(array_x)
    array_px_transformed = np.copy(array_px)
    array_y_transformed = np.copy(array_y)
    array_py_transformed = np.copy(array_py)

    # Check conditions
    x_ok = array_x is not None and array_px is not None
    y_ok = array_y is not None and array_py is not None

    # Get variables
    x, y = X
    px, py = P

    # Evaluate the result with lambdify
    for x_new, px_new, y_new, py_new in l_transforms:
        if x_ok and y_ok:
            f_x = sym.lambdify((x, px, y, py), x_new)
            f_px = sym.lambdify((x, px, y, py), px_new)
            f_y = sym.lambdify((x, px, y, py), y_new)
            f_py = sym.lambdify((x, px, y, py), py_new)
            array_x_transformed_temp = f_x(
                array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed
            )
            array_px_transformed_temp = f_px(
                array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed
            )
            array_y_transformed_temp = f_y(
                array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed
            )
            array_py_transformed_temp = f_py(
                array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed
            )
            array_x_transformed = np.copy(array_x_transformed_temp)
            array_px_transformed = np.copy(array_px_transformed_temp)
            array_y_transformed = np.copy(array_y_transformed_temp)
            array_py_transformed = np.copy(array_py_transformed_temp)

        elif x_ok:
            f_x = sym.lambdify((x, px), x_new)
            f_px = sym.lambdify((x, px), px_new)
            array_x_transformed_temp = f_x(array_x_transformed, array_px_transformed)
            array_px_transformed = f_px(array_x_transformed, array_px_transformed)
            array_x_transformed = np.copy(array_x_transformed_temp)
        elif y_ok:
            f_y = sym.lambdify((y, py), y_new)
            f_py = sym.lambdify((y, py), py_new)
            array_y_transformed_temp = f_y(array_y_transformed, array_py_transformed)
            array_py_transformed = f_py(array_y_transformed, array_py_transformed)
            array_y_transformed = np.copy(array_y_transformed_temp)

    return array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed


def simulate_transfer_symplectic_integrator(
    array_x,
    array_px,
    X,
    P,
    array_y=None,
    array_py=None,
    magnet_order=1,
    L=1.0,
    delta=0.0,
    k_n=0.1,
    order_symplectic_integrator=4,
    max_order_lie_transform=20,
):
    """This function is used to generate a Hamiltonian for a given multipole, generate the
    corresponding symplectic map, and compute the transformed coordinates of the particles."""

    # Get separate Hamiltonians
    Hd, Hk = generate_hamiltonian(
        magnet_order, delta, X, P, k=k_n, return_separate_hamiltonians=True
    )

    # Get list of analytical transformations
    l_transforms = symplectic_analytical_integrator(
        order_symplectic_integrator, Hd, Hk, X, P, L, max_order_lie_transform
    )

    # Compute the transformations
    (
        array_x_transformed,
        array_px_transformed,
        array_y_transformed,
        array_py_transformed,
    ) = symplectic_numerical_integrator(
        X, P, l_transforms, array_x, array_px, array_y=array_y, array_py=array_py
    )

    # Return the result
    if array_y is None:
        return array_x_transformed, array_px_transformed
    else:
        return array_x_transformed, array_px_transformed, array_y_transformed, array_py_transformed


def check_emittance_conservation_quadrupole(
    X,
    P,
    array_x,
    array_px,
    array_y,
    array_py,
    max_order_lie_transform=6,
    order_symplectic_integrator=4,
):
    """This function is used to check if the emittance is conserved when simulating trajectories
    through a quadrupole, using symplectic and non symplectic maps."""
    # Do integration
    l_emittance_x_truncated = []
    l_emittance_y_truncated = []
    l_emittance_x_symplectic = []
    l_emittance_y_symplectic = []
    l_emittance_x_exact = []
    l_emittance_y_exact = []
    l_strength = list(np.linspace(0.001, 6, 20))
    for k_n in l_strength:
        # Transformation for truncated map
        (
            array_x_truncated,
            array_px_truncated,
            array_y_truncated,
            array_py_truncated,
        ) = simulate_transfer_truncated(
            array_x,
            array_px,
            X,
            P,
            k_n=k_n,
            max_order_lie_transform=max_order_lie_transform,
            array_u2=array_y,
            array_pu2=array_py,
        )
        emittance_x_truncated, _, _, _ = get_twiss_and_emittance(
            array_x_truncated, array_px_truncated
        )
        emittance_y_truncated, _, _, _ = get_twiss_and_emittance(
            array_y_truncated, array_py_truncated
        )

        # Transformation for symplectic map
        (
            array_x_symplectic,
            array_px_symplectic,
            array_y_symplectic,
            array_py_symplectic,
        ) = simulate_transfer_symplectic_integrator(
            array_x,
            array_px,
            X,
            P,
            array_y=array_y,
            array_py=array_py,
            k_n=k_n,
            order_symplectic_integrator=order_symplectic_integrator,
            max_order_lie_transform=max_order_lie_transform,
        )
        emittance_x_symplectic, _, _, _ = get_twiss_and_emittance(
            array_x_symplectic, array_px_symplectic
        )
        emittance_y_symplectic, _, _, _ = get_twiss_and_emittance(
            array_y_symplectic, array_py_symplectic
        )

        # Transformation for exact map
        array_x_exact, array_px_exact = return_exact_map_for_quadrupole(
            X,
            P,
            array_x,
            array_px,
            k_n=k_n,
            y_map=False,
        )
        array_y_exact, array_py_exact = return_exact_map_for_quadrupole(
            X,
            P,
            array_y,
            array_py,
            k_n=k_n,
            y_map=True,
        )
        emittance_x_exact, _, _, _ = get_twiss_and_emittance(array_x_exact, array_px_exact)
        emittance_y_exact, _, _, _ = get_twiss_and_emittance(array_y_exact, array_py_exact)

        # Add all emittances to list
        l_emittance_x_truncated.append(emittance_x_truncated)
        l_emittance_y_truncated.append(emittance_y_truncated)
        l_emittance_x_symplectic.append(emittance_x_symplectic)
        l_emittance_y_symplectic.append(emittance_y_symplectic)
        l_emittance_x_exact.append(emittance_x_exact)
        l_emittance_y_exact.append(emittance_y_exact)

    return (
        l_emittance_x_truncated,
        l_emittance_y_truncated,
        l_emittance_x_symplectic,
        l_emittance_y_symplectic,
        l_emittance_x_exact,
        l_emittance_y_exact,
        l_strength,
    )


# ==================================================================================================
# --- Plotting functions
# ==================================================================================================

# Plot the particle distribution
def plot_distribution(
    array_u,
    array_pu,
    l_colors,
    max_std,
    u_ellipse=None,
    p_ellipse=None,
    ellipse=None,
    label_x=r"$u$",
    label_y=r"$p_u$",
    title=None,
):
    """This function is used to plot the initial particle distribution."""
    fig = plx.scatter(
        x=array_u,
        y=array_pu,
        labels={"x": label_x, "y": label_y},
        color_continuous_scale="spectral",
        color=l_colors,
        opacity=0.8,
    )
    if ellipse is not None:
        fig.add_trace(
            go.Contour(
                x=u_ellipse[0, :],
                y=p_ellipse[:, 0],
                z=ellipse,
                colorscale="Blues",
                showscale=False,
                opacity=0.5,
            )
        )
    fig.update_layout(yaxis_range=[-5 * max_std, 5 * max_std])
    fig.update_layout(xaxis_range=[-5 * max_std, 5 * max_std])
    fig.update_coloraxes(showscale=False)
    fig.update_layout(template="plotly_white")
    if title is not None:
        fig.update_layout(title_text=title, title_x=0.5)

    fig.show()


def plot_interactive_distribution_quadrupole(
    X,
    P,
    array_u,
    array_pu,
    l_colors,
    max_std,
    label_x=r"$u$",
    label_y=r"$p_u$",
    title=None,
    max_order_lie_transform=6,
    order_symplectic_integrator=4,
    integrator="truncated_map",
    animation=False,
    gif_path="test.gif",
    fps=60,
    duration=5,
):
    """This function is used to plot the transformation of the particle distribution when going
    through a quadrupole, with either a truncated or symplectic map, and compare it with the
    corresponding closed-form solution.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Truncated map integration"
            if integrator == "truncated_map"
            else "Symplectic integration",
            "Exact map integration",
        ),
    )
    fig.append_trace(
        go.Scatter(
            x=array_u,
            y=array_pu,
            marker_colorscale="spectral",
            marker_color=l_colors,
            marker_opacity=0.8,
            mode="markers",
        ),
        row=1,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=array_u,
            y=array_pu,
            marker_colorscale="spectral",
            marker_color=l_colors,
            marker_opacity=0.8,
            mode="markers",
        ),
        row=1,
        col=2,
    )

    # Update overall layout
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        width=1000,
        height=600,
        template="plotly_white",
        showlegend=False,
    )

    # Update yaxis properties
    fig.update_xaxes(title_text=label_x, range=[-5 * max_std, 5 * max_std], row=1, col=1)
    fig.update_yaxes(title_text=label_y, range=[-5 * max_std, 5 * max_std], row=1, col=1)
    fig.update_xaxes(title_text=label_x, range=[-5 * max_std, 5 * max_std], row=1, col=2)
    fig.update_yaxes(title_text=label_y, range=[-5 * max_std, 5 * max_std], row=1, col=2)
    fig.update_coloraxes(showscale=False, row=1, col=1)
    fig.update_coloraxes(showscale=False, row=1, col=2)

    # Slider for the knob
    k_min = 0.0001
    k_max = 10.0
    k_step = duration / fps
    step = duration / fps
    slider_range = np.linspace(k_min, k_max, duration * fps)
    slider = widgets.FloatSlider(
        value=0.0,
        min=k_min,
        max=k_max,
        step=k_step,
        description="k",
        continuous_update=True,
    )

    # Build interaction
    g = go.FigureWidget(fig)

    def response(change):
        if integrator == "truncated_map":
            array_x_transformed, array_px_transformed = simulate_transfer_truncated(
                array_u,
                array_pu,
                X,
                P,
                magnet_order=1,
                k_n=slider.value,
                max_order_lie_transform=max_order_lie_transform,
            )
        elif integrator == "symplectic_integrator":
            array_x_transformed, array_px_transformed = simulate_transfer_symplectic_integrator(
                array_u,
                array_pu,
                X,
                P,
                magnet_order=1,
                k_n=slider.value,
                order_symplectic_integrator=order_symplectic_integrator,
                max_order_lie_transform=max_order_lie_transform,
            )

        array_x_transformed_exact, array_px_transformed_exact = return_exact_map_for_quadrupole(
            X, P, array_u, array_pu, k_n=slider.value
        )

        with g.batch_update():
            # Update data
            g.data[0].x = array_x_transformed
            g.data[0].y = array_px_transformed
            g.data[1].x = array_x_transformed_exact
            g.data[1].y = array_px_transformed_exact

            # Update subplot titles
            g.layout["annotations"][0]["text"] = (
                f"Truncated map integration, k = {slider.value:.2f}"
                if integrator == "truncated_map"
                else f"Symplectic integration, k = {slider.value:.2f}"
            )
            g.layout["annotations"][1]["text"] = f"Exact map integration, k = {slider.value:.2f}"

    # Function to convert interaction to an animation
    def make_frame(t):
        idx = int(round(t * fps))
        slider.value = slider_range[idx]
        response(None)
        return plotly_fig2array(g)

    container = widgets.HBox(
        children=[
            slider,
        ]
    )

    slider.observe(response, names="value")
    vbox = widgets.VBox([container, g])
    if animation:
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_gif(gif_path, fps=fps)
    return vbox


def plot_interactive_distribution_x_y(
    X,
    P,
    array_x,
    array_px,
    array_y,
    array_py,
    l_colors_x,
    l_colors_y,
    max_std_x,
    max_std_y,
    label_x_1=r"$x$",
    label_y_1=r"$p_x$",
    label_x_2=r"$y$",
    label_y_2=r"$p_y$",
    title=None,
    max_order_lie_transform=6,
    order_symplectic_integrator=4,
    magnet_order=1,
    k_min=0.0001,
    k_max=10.0,
    exact_map_quadrupole=False,
    animation=False,
    gif_path="test.gif",
    fps=60,
    duration=5,
):
    """This function is used to compare all maps (truncated, symplectic, and exact if available) in
    phase-space  and real space for a given particle distribution going through a multipole."""

    n_rows = 3 if exact_map_quadrupole else 2
    fig = make_subplots(
        rows=n_rows,
        cols=3,
        subplot_titles=(
            "Truncated map integration x-px",
            "Truncated map integration y-py",
            "Truncated map integration x-y",
            "Symplectic integration x-px",
            "Symplectic integration y-py",
            "Symplectic integration x-y",
            "Exact map integration x-px",
            "Exact map integration y-py",
            "Exact map integration x-y",
        )
        if exact_map_quadrupole
        else (
            "Truncated map integration x-px",
            "Truncated map integration y-py",
            "Truncated map integration x-y",
            "Symplectic integration x-px",
            "Symplectic integration y-py",
            "Symplectic integration x-y",
        ),
    )

    # Normal phase-space representations
    for idx_row in range(1, n_rows + 1):
        for idx_col in range(1, 3):
            fig.append_trace(
                go.Scatter(
                    x=array_x if idx_col == 1 else array_y,
                    y=array_px if idx_col == 1 else array_py,
                    marker_colorscale="spectral",
                    marker_color=l_colors_x if idx_col == 1 else l_colors_y,
                    marker_opacity=0.8,
                    mode="markers",
                ),
                row=idx_row,
                col=idx_col,
            )

    # Actual space representation
    for idx_row in range(1, n_rows + 1):
        fig.append_trace(
            go.Scatter(
                x=array_x,
                y=array_y,
                marker_opacity=0.8,
                mode="markers",
                marker_color="teal",
            ),
            row=idx_row,
            col=3,
        )

    # Update overall layout
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        width=1000,
        height=1000 if exact_map_quadrupole else 700,
        template="plotly_white",
        showlegend=False,
    )

    # Update yaxis properties
    for row in range(1, n_rows + 1):
        fig.update_xaxes(
            title_text=label_x_1, range=[-15 * max_std_x, 15 * max_std_x], row=row, col=1
        )
        fig.update_yaxes(
            title_text=label_y_1, range=[-15 * max_std_x, 15 * max_std_x], row=row, col=1
        )
        fig.update_xaxes(
            title_text=label_x_2, range=[-15 * max_std_y, 15 * max_std_y], row=row, col=2
        )
        fig.update_yaxes(
            title_text=label_y_2, range=[-15 * max_std_y, 15 * max_std_y], row=row, col=2
        )
        fig.update_xaxes(
            title_text=label_x_1, range=[-15 * max_std_x, 15 * max_std_x], row=row, col=3
        )
        fig.update_yaxes(
            title_text=label_x_2, range=[-15 * max_std_y, 15 * max_std_y], row=row, col=3
        )
        for col in range(1, 4):
            fig.update_coloraxes(showscale=False, row=row, col=col)
            fig.update_coloraxes(showscale=False, row=row, col=col)

    # Slider for the knob
    k_min = k_min
    k_max = k_max
    k_step = duration / fps
    slider_range = np.linspace(k_min, k_max, duration * fps)
    slider = widgets.FloatSlider(
        value=0.0,
        min=k_min,
        max=k_max,
        step=k_step,
        description="k",
        continuous_update=True,
    )

    # Build interaction
    g = go.FigureWidget(fig)

    def response(change):
        # Transformation for truncated map
        (
            array_x_truncated,
            array_px_truncated,
            array_y_truncated,
            array_py_truncated,
        ) = simulate_transfer_truncated(
            array_x,
            array_px,
            X,
            P,
            magnet_order=magnet_order,
            k_n=slider.value,
            max_order_lie_transform=max_order_lie_transform,
            array_u2=array_y,
            array_pu2=array_py,
        )

        # Transformation for symplectic map
        (
            array_x_symplectic,
            array_px_symplectic,
            array_y_symplectic,
            array_py_symplectic,
        ) = simulate_transfer_symplectic_integrator(
            array_x,
            array_px,
            X,
            P,
            array_y=array_y,
            array_py=array_py,
            magnet_order=magnet_order,
            k_n=slider.value,
            order_symplectic_integrator=order_symplectic_integrator,
            max_order_lie_transform=max_order_lie_transform,
        )

        if exact_map_quadrupole:
            # Transformation for exact map
            array_x_exact, array_px_exact = return_exact_map_for_quadrupole(
                X,
                P,
                array_x,
                array_px,
                k_n=slider.value,
                y_map=False,
            )
            array_y_exact, array_py_exact = return_exact_map_for_quadrupole(
                X,
                P,
                array_y,
                array_py,
                k_n=slider.value,
                y_map=True,
            )

        with g.batch_update():
            #  Update title with the rounded value (2 decimals) of the slider
            g.layout.title.text = title + " k = " + str(round(slider.value, 2))

            # First 2 col of 1st row
            g.data[0].x = array_x_truncated
            g.data[0].y = array_px_truncated
            g.data[1].x = array_y_truncated
            g.data[1].y = array_py_truncated

            # First 2 col of 2nd row
            g.data[2].x = array_x_symplectic
            g.data[2].y = array_px_symplectic
            g.data[3].x = array_y_symplectic
            g.data[3].y = array_py_symplectic

            if exact_map_quadrupole:
                # First 2 col of 3rd row
                g.data[4].x = array_x_exact
                g.data[4].y = array_px_exact
                g.data[5].x = array_y_exact
                g.data[5].y = array_py_exact

                # Last col for each row
                g.data[6].x = array_x_truncated
                g.data[6].y = array_y_truncated
                g.data[7].x = array_x_symplectic
                g.data[7].y = array_y_symplectic
                g.data[8].x = array_x_exact
                g.data[8].y = array_y_exact

            else:
                # Last col for each row
                g.data[4].x = array_x_truncated
                g.data[4].y = array_y_truncated
                g.data[5].x = array_x_symplectic
                g.data[5].y = array_y_symplectic

    # Function to convert interaction to an animation
    def make_frame(t):
        idx = int(round(t * fps))
        slider.value = slider_range[idx]
        response(None)
        return plotly_fig2array(g)

    if animation:
        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_gif(gif_path, fps=fps)

    container = widgets.HBox(
        children=[
            slider,
        ]
    )
    slider.observe(response, names="value")
    vbox = widgets.VBox([container, g])
    return vbox


def plot_emittance_conservation(
    l_emittance_x_truncated,
    l_emittance_y_truncated,
    l_emittance_x_symplectic,
    l_emittance_y_symplectic,
    l_emittance_x_exact,
    l_emittance_y_exact,
    l_strength,
):
    """This function allows to plot the output of check_emittance_conservation_quadrupole()."""
    # Plot the result
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=l_strength, y=l_emittance_x_truncated, name="Truncated x emittance map")
    )
    fig.add_trace(
        go.Scatter(x=l_strength, y=l_emittance_y_truncated, name="Truncated y emittance map")
    )
    fig.add_trace(
        go.Scatter(x=l_strength, y=l_emittance_x_symplectic, name="Symplectic x emittance map")
    )
    fig.add_trace(
        go.Scatter(x=l_strength, y=l_emittance_y_symplectic, name="Symplectic y emittance map")
    )
    fig.add_trace(
        go.Scatter(
            x=l_strength,
            y=l_emittance_x_exact,
            name="Exact x emittance map",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=l_strength,
            y=l_emittance_y_exact,
            name="Exact y emittance map",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Emittance growth for different maps",
        xaxis_title="Quadrupole strength",
        yaxis_title="Emittance",
        template="plotly_white",
        title_x=0.5,
    )
    fig.show()
