"""Library for simulating Double Spring Pendulum (DSP)."""

import dataclasses
from typing import Sequence

import gin
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from pylab import figure, rcParams, rc, rc_context, subplot

@gin.configurable
@dataclasses.dataclass
class DSPConstants:
    """Constant four Double Spring Pendulum (DSP).

    Attributes:
        m1: Mass of bob 1.
        m2: Mass of bob 2.
        l1: Equilibrium length of spring 1.
        l2: Equilibrium length of spring 2.
        k1: Spring constant for spring 1.
        k2: Spring constant for spring 2.
        g: Gravitational constant, defaults to 9.8.
        process_noise_db: The level of the process noise in dB, defaults to -50.
        Ce: (Can not be set.) The covariance of the process noise, defined via `process_noise_db`.
    """

    m1: float
    m2: float
    l1: float
    l2: float
    k1: float
    k2: float
    g: float = 9.8
    process_noise_db: float = -50

    def __post_init__(self):
        self.Ce = 10 ** (self.process_noise_db / 10) * np.eye(8)


@dataclasses.dataclass
class DSPState:
    """Container for the state trajectory of Double Spring Pendulum (DSP)."""

    theta1_points: list[float] = dataclasses.field(default_factory=list)
    omega1_points: list[float] = dataclasses.field(default_factory=list)
    r1_points: list[float] = dataclasses.field(default_factory=list)
    v1_points: list[float] = dataclasses.field(default_factory=list)
    theta2_points: list[float] = dataclasses.field(default_factory=list)
    omega2_points: list[float] = dataclasses.field(default_factory=list)
    r2_points: list[float] = dataclasses.field(default_factory=list)
    v2_points: list[float] = dataclasses.field(default_factory=list)
    t_points: list[float] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t_points)

    def append_from_eta(self, eta: np.ndarray, t: float) -> None:
        self.theta1_points.append(eta[0])
        self.omega1_points.append(eta[1])
        self.r1_points.append(eta[2])
        self.v1_points.append(eta[3])
        self.theta2_points.append(eta[4])
        self.omega2_points.append(eta[5])
        self.r2_points.append(eta[6])
        self.v2_points.append(eta[7])
        self.t_points.append(t)

    def get_tip_point_positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the positions of both tip points as separate coordinate arrays.
        
        Returns:
            tuple: (x1_points, y1_points, x2_points, y2_points)
        """
        theta1_points, theta2_points = np.array(self.theta1_points), np.array(self.theta2_points)
        r1_points, r2_points = np.array(self.r1_points), np.array(self.r2_points)
        
        # First pendulum tip positions
        x1_points = r1_points * np.sin(theta1_points)
        y1_points = -r1_points * np.cos(theta1_points)
        
        # Second pendulum tip positions (relative to origin)
        x2_points = r1_points * np.sin(theta1_points) + r2_points * np.sin(theta2_points)
        y2_points = -r1_points * np.cos(theta1_points) - r2_points * np.cos(theta2_points)
        
        positions = np.stack(
            [x1_points, y1_points, x2_points, y2_points], axis=1
        )
        return positions

    def get_eta(self) -> np.ndarray:
        """Returns the state vector of the DSP."""
        return np.stack([
            self.theta1_points,
            self.omega1_points,
            self.r1_points,
            self.v1_points,
            self.theta2_points,
            self.omega2_points,
            self.r2_points,
            self.v2_points,
        ], axis=1)


def f(eta: np.ndarray, dsp_constants: DSPConstants) -> np.ndarray:
    m1 = dsp_constants.m1
    m2 = dsp_constants.m2
    l1 = dsp_constants.l1
    l2 = dsp_constants.l2
    k1 = dsp_constants.k1
    k2 = dsp_constants.k2
    g = dsp_constants.g
    theta1 = eta[0]
    omega1 = eta[1]
    r1 = eta[2]
    v1 = eta[3]
    theta2 = eta[4]
    omega2 = eta[5]
    r2 = eta[6]
    v2 = eta[7]

    f_theta1 = omega1
    f_r1 = v1
    f_theta2 = omega2
    f_r2 = v2

    f_omega1 = (k2 * (l2 - r2) * np.sin(theta1 - theta2) - 2 * m1 * v1 * omega1
                - m1 * g * np.sin(theta1)) / (m1 * r1)

    f_v1 = (m1 * g * np.cos(theta1) + k1 * (l1 - r1) - k2 * (l2 - r2) * np.cos(theta1 - theta2)
            + m1 * r1 * omega1 ** 2) / m1

    f_omega2 = (-k1 * (l1 - r1) * np.sin(theta1 - theta2) - 2 * m1 * v2 * omega2) / (m1 * r2)

    f_v2 = (k2 * (m1 + m2) * (l2 - r2) + m1 * m2 * r2 * omega2 ** 2 - m2 * k1 * (l1 - r1) * np.cos(theta1 - theta2)) / (
                m1 * m2)

    return np.array([f_theta1, f_omega1, f_r1, f_v1, f_theta2, f_omega2, f_r2, f_v2], float)


# define a function that takes initial angles and spring compressions in as
#  parameters and outputs array of angles and velocities of both masses
def elastic_double_pendulum(
    theta1_initial_deg: float,
    theta2_initial_deg: float,
    r1_initial: float,
    r2_initial: float,
    simulation_time: int,
    dsp_constants: DSPConstants,
    stochastic: bool = False
) -> DSPState:
    # Convert initial angles from degrees to radians
    theta1_initial = np.radians(theta1_initial_deg)
    theta2_initial = np.radians(theta2_initial_deg)
    
    # Initial conditions for angular velocities and radial velocities
    omega1_initial = 0.0
    omega2_initial = 0.0
    v1_initial = 0.0
    v2_initial = 0.0

    # Time domain setup
    dt = 0.02  # time step
    t_points = np.arange(0.0, dt * simulation_time, dt)

    # Initial state vector: (theta1, omega1, r1, v1, theta2, omega2, r2, v2)
    eta = np.array([theta1_initial, omega1_initial, r1_initial, v1_initial,
                      theta2_initial, omega2_initial, r2_initial, v2_initial], float)

    dsp_state = DSPState()

    for t in t_points:
        dsp_state.append_from_eta(eta, t)

        k1 = f(eta, dsp_constants)
        k2 = f(eta + 0.5 * dt * k1, dsp_constants)
        k3 = f(eta + 0.5 * dt * k2, dsp_constants)
        k4 = f(eta + dt * k3, dsp_constants)

        if stochastic:
            noise = np.random.multivariate_normal(np.zeros_like(eta), dsp_constants.Ce)
            eta += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6 + noise
        else:
            eta += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return dsp_state


def get_one_example(
    theta1_deg: float, 
    theta2_deg: float, 
    l1: float, 
    l2: float, 
    simulation_time: int,
    dsp_constants: DSPConstants, 
    animate: bool = False, 
    stochastic: bool = False
) -> tuple[list[float], list[float]]:
    """Get one example of a double pendulum simulation."""
    
    dsp_state = elastic_double_pendulum(
        theta1_deg,
        theta2_deg,
        l1,
        l2,
        simulation_time,
        dsp_constants, 
        stochastic
    )

    # x1_coords = [
    #    r1 * np.sin(theta1) 
    #     for r1, theta1 in zip(dsp_state.r1_points, dsp_state.theta1_points)
    # ]
    # y1_coords = [
    #     -r1 * np.cos(theta1) 
    #     for r1, theta1 in zip(dsp_state.r1_points, dsp_state.theta1_points)
    # ]
    x2_coords = [
        r1 * np.sin(theta1) + r2 * np.sin(theta2) 
        for r1, theta1, r2, theta2 in zip(dsp_state.r1_points, dsp_state.theta1_points, dsp_state.r2_points, dsp_state.theta2_points)
    ]
    y2_coords = [
        -r1 * np.cos(theta1) - r2 * np.cos(theta2) 
        for r1, theta1, r2, theta2 in zip(dsp_state.r1_points, dsp_state.theta1_points, dsp_state.r2_points, dsp_state.theta2_points)
    ]

    if animate:
        get_animate(
            dsp_state.theta1_points, 
            dsp_state.r1_points, 
            dsp_state.theta2_points, 
            dsp_state.r2_points, 
            dsp_state.t_points, 
            x2_coords, 
            y2_coords
        )
    
    return x2_coords, y2_coords


def get_animate(theta1_points, r1_points, theta2_points, r2_points, t_points, x2_points, y2_points):
    rcParams.update({'font.size': 18})
    rc('axes', linewidth=2)
    with rc_context({'axes.edgecolor': 'white', 'xtick.color': 'white',
                     'ytick.color': 'white', 'figure.facecolor': 'darkslategrey',
                     'axes.facecolor': 'darkslategrey', 'axes.labelcolor': 'white',
                     'axes.titlecolor': 'white'}):

        # print runtime of our code
        matplotlib.use('TkAgg')

        # set up a figure for pendulum
        fig = figure(figsize=(10, 8))

        # subplot for animation of pendulum
        ax_pend = subplot(1, 1, 1, aspect='equal')
        # get rid of axis ticks
        ax_pend.tick_params(axis='both', colors="darkslategrey")

        ### finally we animate ###
        # create a list to input images in for each time step
        ims = []
        index = 0
        # only show the first 80seconds or so in the gif
        while index <= len(t_points) - 1:
            ln1, = ax_pend.plot([0, r1_points[index] * np.sin(theta1_points[index])],
                                [0, -r1_points[index] * np.cos(theta1_points[index])],
                                color='k', lw=3, zorder=99)
            bob1, = ax_pend.plot(r1_points[index] * np.sin(theta1_points[index]),
                                 -r1_points[index] * np.cos(theta1_points[index]), 'o',
                                 markersize=22, color="m", zorder=100)

            ln2, = ax_pend.plot([r1_points[index] * np.sin(theta1_points[index]),
                                 r1_points[index] * np.sin(theta1_points[index]) +
                                 r2_points[index] * np.sin(theta2_points[index])],
                                [-r1_points[index] * np.cos(theta1_points[index]),
                                 -r1_points[index] * np.cos(theta1_points[index])
                                 - r2_points[index] * np.cos(theta2_points[index])], color='k', linestyle='--', lw=3, zorder=99)
            bob2, = ax_pend.plot(r1_points[index] * np.sin(theta1_points[index]) +
                                 r2_points[index] * np.sin(theta2_points[index]),
                                 -r1_points[index] * np.cos(theta1_points[index])
                                 - r2_points[index] * np.cos(theta2_points[index]), 'o',
                                 markersize=22, color="coral", zorder=100)

            # trail1, = ax_pend.plot(x1_points[:index], y1_points[:index],
            #                        color="lime", lw=0.8, zorder=20)

            trail2, = ax_pend.plot(x2_points[:index], y2_points[:index],
                                   color="cyan", lw=0.8, zorder=20)

            # add pictures to ims list
            ims.append([ln1, bob1, ln2, bob2, trail2])

            # only show every 6 frames
            index += 6

        # save animations
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        # writervideo = animation.FFMpegWriter(fps=60)
        # ani.save('./170_105_elasticDoublePend.mp4', writer=writervideo)
        plt.show()

def dB_to_lin(x):
    return 10**(x/10)

def gen_dataset(
    num_samples: int,
    T: int,
    smnr_db: float,
    dsp_constants: DSPConstants,
    stochastic: bool = False,
) -> tuple[list[DSPState], list[np.ndarray]]:
    decimation_factor = 5
    theta1_initials = [100] * num_samples
    theta2_initials = [120] * num_samples
    l1_initials = [1.2] * num_samples
    l2_initials = [0.8] * num_samples
    all_positions = []
    for the1, the2, l1, l2 in zip(
        theta1_initials, theta2_initials, l1_initials, l2_initials
    ):
        dsp_state = elastic_double_pendulum(
            the1,
            the2,
            l1,
            l2,
            T * decimation_factor,
            dsp_constants,
            stochastic
        )
        all_positions.append(dsp_state.get_tip_point_positions())
    all_positions = [pos[::decimation_factor,:] for pos in all_positions]
    signal_vars = [np.var(pos) for pos in all_positions]
    sigma_w2s = [signal_var / dB_to_lin(smnr_db) for signal_var in signal_vars]
    dim = all_positions[0].shape[-1]
    Cws = [sigma_w2 * np.eye(dim) for sigma_w2 in sigma_w2s]
    ws = np.stack([
        np.random.multivariate_normal(np.zeros(dim), Cw, size=(T,)) for Cw in Cws
    ], axis=0)
    states = np.stack(all_positions, axis=0)
    observations = states + ws
    return states, observations, np.stack(Cws, axis=0)


def main():
    
    dsp_constants = DSPConstants(m1=1.2, m2=0.8, l1=1.2, l2=0.8, k1=500.0, k2=1000.0)
    states, observations, Cws = gen_dataset(10, 100, 20, dsp_constants, stochastic=True)
    x2, y2 = observations[0][:, 0], observations[0][:, 1]
    s1, s2 = states[0][:, 0], states[0][:, 1]
    print(Cws[0])
    print(Cws.shape)
    plt.figure(figsize=(5.2, 5))
    plt.plot(x2, y2, color='blue')
    plt.plot(s1, s2, color='red')
    # plt.xlabel('abscissa', fontsize=32)
    plt.gca().xaxis.set_label_coords(0.5, 1.09)
    # plt.ylabel('ordinate', fontsize=32)
    #plt.xticks([-2, -1, 0, 1, 2], fontsize=26)
    #plt.yticks([-2, -1, 0, 1, 2], fontsize=26)
    plt.grid(True)
    plt.tight_layout(pad=0.2)
    plt.savefig('double_pendelum_with_spring.png')


if __name__ == '__main__':
    main()
