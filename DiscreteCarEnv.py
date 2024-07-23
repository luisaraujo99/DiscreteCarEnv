import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import math
import os
from typing import Callable, Optional, Union
from numpy.typing import NDArray


def R(
    track_matrix: NDArray, out: float, wall: float, tar: float, ice: float
) -> NDArray:
    """
    Build a reward matrix based on the kind of state: each state associated with a
    matrix entry has its own reward value. This is useful if one wants to have a
    a granular control over the expected rewards in specific states (e.g. distance).
    """
    reward_matrix: NDArray = np.zeros(track_matrix.shape)
    reward_matrix[track_matrix == 0] = -1
    reward_matrix[track_matrix == 1] = out
    reward_matrix[track_matrix == 2] = -1
    reward_matrix[track_matrix == 4] = wall
    reward_matrix[track_matrix == 5] = ice
    reward_matrix[track_matrix == 3] = tar

    return reward_matrix


def normalized_r(reward_matrix: NDArray):
    """
    Normalize the reward matrix: M*(1/max(M)).
    """
    return reward_matrix / (np.max(np.abs(reward_matrix)))


def matrix_to_image(map_matrix: NDArray) -> NDArray:
    """
    Map the values to numbers that ensure greater separability for the representation
    of the states as pixel values, and cast them to 8 bit numpy Unsigned Integers.
    """
    matrix_state: NDArray = map_matrix.astype(np.uint8)
    matrix_state[map_matrix == 4] = int(b"00000000", 2)  # wall
    matrix_state[map_matrix == 5] = int(b"00000011", 2)  # ice
    matrix_state[map_matrix == 1] = int(b"00000111", 2)  # out
    matrix_state[map_matrix == 0] = int(b"00011111", 2)  # in
    matrix_state[map_matrix == 2] = int(b"00111111", 2)  # start
    matrix_state[map_matrix == 3] = int(b"11111111", 2)  # target/finish
    return matrix_state.reshape(1, len(map_matrix), len(map_matrix))


class DiscreteCarEnv(gym.Env):
    metadata: dict = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def read_tracks(self, r_params: dict, path: str) -> None:
        """
        Read state track maps from csv files in a specific path. Every csv file in `path` will be
        loaded as a track.
        """
        # Load csv files representing tracks as numpy arrays.
        self.maps: NDArray = np.array(
            [
                np.loadtxt(
                    open(path + track_csv_file, "r"), delimiter=",", dtype=np.uint8
                )
                for track_csv_file in filter(
                    lambda filename: filename.split(".")[-1] == "csv", os.listdir(path)
                )
            ]
        )

        # Number of maps
        self.num_maps: int = len(self.maps)

        # self.probs = np.array(([0.15/13 for i in range(13)]+[0.35/13 for i in range(13)])*2)

        # Array with map ids
        self.inds: NDArray = np.array([i for i in range(self.num_maps)])

        # Width of each map.
        self.size: int = self.maps[0].shape[0]

        # Shape used to normalize the state values.
        self.state_vector_norm: NDArray = np.array([self.size - 1, self.size - 1, 3])

        # Lambda function used to get initial states from a map matrix.
        get_initial_states: Callable[[np.array], np.array] = (
            lambda track_matrix: np.array(
                list(np.argwhere(track_matrix == 2)[0])[::-1] + [1], dtype=np.uint8
            )
        )

        # List of initial states by loaded csv map.
        self.initial_states: NDArray = np.array(
            [get_initial_states(map) for map in self.maps]
        )

        # List of converted maps loaded from csvs to a better representation.
        self.state_matrices: NDArray = np.array(
            [matrix_to_image(map) for map in self.maps]
        )

        # One reward matrix by loaded map.
        # TODO this can be changed to a list of functions for memory reduction
        # when to test generalization.
        self.reward_matrices: NDArray = np.array(
            [normalized_r(R(m, **r_params)) for m in self.maps]
        )

    def __init__(
        self,
        tracks_path: str,
        render_mode=None,
        shuffle=True,
        r_params={"out": -1, "ice": -1, "wall": -1, "tar": 0},
        tabular=False,
        speed=3,
    ):
        """
        :param tracks_path: path that contains csv files representing the maps.
        :param render_mode: gymnasium render mode.
        :param shuffle: set to `True` if at each new episode the selected map is chosen randomly.
            If it is set to `False` the maps are selected sequentially.
        :param r_params: reward values for each state
        :param tabular: whether the environment should be adapted to tabular or functions approx. methods.
        :param speed: max speed that the game allows
        """
        # Pygame window width size.
        self.window_size = 900

        # Load the maps, and reward functions.
        self.read_tracks(r_params, tracks_path)

        # Adapt environment to tabular or non tabular RL methods.
        self.tabular = tabular

        image = spaces.Box(
            low=0, high=255, shape=(1, self.size, self.size), dtype=np.uint8
        )
        if self.tabular:
            state = spaces.Box(
                low=np.array([0, 0, 1]),
                high=np.array([self.size, self.size, speed + 1]),
                shape=(3,),
                dtype=np.uint8,
            )
            self.observation_space = spaces.Dict(state=state)
        else:
            state = spaces.Box(
                low=np.array([0, 0, 0]),
                high=np.array([1, 1, 1]),
                shape=(3,),
                dtype=np.float64,
            )
            self.observation_space = spaces.Dict(image=image, state=state)

        # List of possible game orientations as a radian angle.
        self.orientations = np.array([(i * math.pi) / 4 for i in range(8)])
        # List of actions for ease of access.
        self._index_to_action = np.array(
            [np.array([o, m]) for o in self.orientations for m in [-1, 0, 1][:speed]]
        )
        # Gymnasium Discrete Action Space of size |orientations| * |speeds|.
        self.action_space = spaces.Discrete(len(self._index_to_action))

        # Validate render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Variable used to control the selected-map if ``shuffle=False``.
        self.iter: int = 0
        self.shuffle: bool = shuffle
        # Variable that controls the randomness induced by iced-states.
        self.stochastic_prob = (3 / 4) / 8

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        if render_mode:
            self.clock: Optional[pygame.time.Clock] = None
            # colors for pygame frames
            self.square_colors: dict = {
                5: (0, 255, 255),
                4: (255, 128, 0),
                3: (255, 0, 0),
                2: (0, 0, 0),
                1: (255, 255, 255),
                0: (0, 0, 0),
                "car": (0, 255, 0),
            }

        self.window: Optional[pygame.Surface] = None
        self.state: Optional[NDArray] = None
        self.last_state: Optional[NDArray] = None
        self.angle: float = 0

    @property
    def _get_obs(self) -> dict:
        return (
            {"state": self.state}
            if self.tabular
            else {
                "image": self.matrix_state,
                "state": self.state / self.state_vector_norm,
            }
        )

    def reset(self, seed=None, options=None):
        """
        Terminate on-going episode and return initial state for the next one.
        """
        super().reset(seed=seed)

        self.map_ind = np.random.choice(self.inds) if self.shuffle else self.iter
        self.matrix_state = self.state_matrices[self.map_ind]
        self.track_matrix = self.maps[self.map_ind]
        self.state = self.initial_states[self.map_ind]
        self.r = self.reward_matrices[self.map_ind]
        self.iter = (self.iter + 1) % self.num_maps
        if self.render_mode is not None:
            self._render_frame()

        return self._get_obs, {}

    def _get_info(self):
        return {}

    def is_wall(self, state: NDArray) -> bool:
        """
        Given a state return true if it a finish line state
        """
        return self.track_matrix[state[1], state[0]] == 4

    def is_finish(self, state: NDArray) -> bool:
        """
        Given a state return true if it a finish line state
        """
        return self.track_matrix[state[1], state[0]] == 3

    def is_intrack(self, state: NDArray) -> bool:
        """
        Given a state, return true if it is a track state.
        """
        state_val = self.track_matrix[state[1], state[0]]
        return state_val == 0 or state_val == 2 or state_val == 3

    def is_outtrack(self, state: NDArray) -> bool:
        """
        Given a state, return true if it is out track.
        """
        return self.track_matrix[state[1], state[0]] == 1

    def is_ice(self, state: NDArray) -> bool:
        """
        Given a state return true if it is ice
        """
        return self.track_matrix[state[1], state[0]] == 5

    def move(self, state: NDArray, a: NDArray) -> NDArray:
        """
        Given a map-specific state and an action return the new state.
        """
        o, m = a
        x, y, v = state
        v_prime = np.clip(v + m - self.is_outtrack(state) * 3, 1, 3)
        x_prime = x + v_prime * round(np.cos(o))
        y_prime = y - v_prime * round(np.sin(o))
        return np.array(
            np.clip([x_prime, y_prime, v_prime], 0, self.size - 1), dtype=np.uint8
        )

    def p(self, state: NDArray, a: NDArray) -> tuple[NDArray, float]:
        """
        Transition function (deterministic MDP) p(s,a) -> s',r.
        """
        o = a[0]
        new_state = self.move(state, a)
        v_prime = new_state[2]

        # Check if the car goes towards a wall in its trajectory
        prev_state = state
        for vel in range(1, v_prime + 1):
            xp, yp, _ = prev_state
            sprime = self.move((xp, yp, 1), (o, 0))
            if self.is_wall(sprime):
                return (
                    np.array([xp, yp, vel], dtype=np.uint8),
                    self.r[sprime[1], sprime[0]],
                )
            prev_state = sprime

        return new_state, self.r[new_state[1], new_state[0]]

    def prob_deterministic(
        self,
        new_state: NDArray,
        reward: float,
        state: NDArray,
        action: NDArray,
    ) -> float:
        """
        Transition probability function (deterministic MDP) P(s',r|r,s).
        """
        real_new_state, real_reward = self.p(state, action)
        if [*real_new_state, real_reward] == [*new_state, reward]:
            return 1
        else:
            return 0

    def prob_stochastic(
        self,
        new_state: NDArray,
        reward: float,
        state: NDArray,
        action: NDArray,
        S: NDArray,
    ):
        """
        Transition probability function p(s',r|r,s).
        """
        action = self._index_to_action[action]
        if not self.is_ice(state):
            return self.prob_deterministic(new_state, reward, state, action)
        else:
            if reward != self.r[new_state[1], new_state[0]]:
                return 0
            # calculate probabilities for s_primes
            probs = {s: 0 for s in S}
            for o in self.orientations:
                (x, y, v), r_prime = self.p(state, (o, action[-1]))
                probs[x, y, v] += self.stochastic_prob + 0 if o != action[0] else 1 / 4
            (x, y, v), _ = self.p(state, (action[0], action[-1]))
            return probs[x, y, v]

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Transition to new state given an action and current state.
        """
        if self.is_ice(self.state) and np.random.rand() > 1 / 4:
            o, m = self._index_to_action[action]
            o = np.random.choice(self.orientations)
            new_state, reward = self.p(self.state, (o, m))
        else:
            new_state, reward = self.p(self.state, self._index_to_action[action])

        self.state = new_state
        terminated = bool(self.is_finish(self.state))

        if self.render_mode is not None:
            self.angle = self._index_to_action[action][0] - math.pi / 2
            self._render_frame()

        return self._get_obs, reward, terminated, False, {}

    def render(self) -> None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_bg(self) -> None:
        """
        Draw a given track map as the pygame background.
        """
        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        for line in range(self.size):
            for col in range(self.size):
                if self.track_matrix[line, col] in self.imgs:
                    pos = np.array([col, line]) * self.pix_square_size
                    pygame.draw.rect(
                        self.canvas,
                        (0, 0, 0),
                        [
                            col * self.pix_square_size,
                            line * self.pix_square_size,
                            self.pix_square_size,
                            self.pix_square_size,
                        ],
                    )
                    self.canvas.blit(self.imgs[self.track_matrix[line, col]], pos)
                else:
                    pygame.draw.rect(
                        self.canvas,
                        self.square_colors[self.track_matrix[line, col]],
                        [
                            col * self.pix_square_size,
                            line * self.pix_square_size,
                            self.pix_square_size,
                            self.pix_square_size,
                        ],
                    )

    def _draw_car(self) -> None:
        """
        Draw the the new car position.
        """
        self._draw_bg()
        car_position = (self.state[:2]) * self.pix_square_size
        rotated_image = pygame.transform.rotate(
            self.imgs["car"], math.degrees(self.angle)
        )
        new_rect = rotated_image.get_rect(
            center=self.imgs["car"].get_rect(topleft=car_position).center
        )
        self.canvas.blit(rotated_image, new_rect)

    def save_initial_state_png(self, print_name: str) -> None:
        """
        Print screen the current map.
        """
        self._render_frame()
        pygame.image.save(self.window, print_name)
        self.close()

    def _render_frame(self) -> Optional[NDArray]:
        """
        Render a frame given a track map and a cars position
        """
        self.last_state = self.state
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.imgs: dict[Union[str, int], pygame.Surface] = {}
            car_img = pygame.image.load("./imgs/car.png")
            car_img = pygame.transform.scale(
                car_img,
                (
                    self.window_size * 0.95 / self.size,
                    self.window_size * 0.95 / self.size,
                ),
            )
            car_img.convert()

            wall_img = pygame.image.load("./imgs/wall.png")
            wall_img = pygame.transform.scale(
                wall_img, (self.window_size / self.size, self.window_size / self.size)
            )
            wall_img.convert()

            ice_img = pygame.image.load("./imgs/ice.png")
            ice_img = pygame.transform.scale(
                ice_img, (self.window_size / self.size, self.window_size / self.size)
            )
            ice_img.convert()

            grass_img = pygame.image.load("./imgs/grass.png")
            grass_img = pygame.transform.scale(
                grass_img, (self.window_size / self.size, self.window_size / self.size)
            )
            grass_img.convert()

            start_img = pygame.image.load("./imgs/start.png")
            start_img = pygame.transform.scale(
                start_img, (self.window_size / self.size, self.window_size / self.size)
            )
            start_img.convert()

            finish_img = pygame.image.load("./imgs/finish.png")
            finish_img = pygame.transform.scale(
                finish_img, (self.window_size / self.size, self.window_size / self.size)
            )
            finish_img.convert()

            road_img = pygame.image.load("./imgs/broken-line.png")
            road_img = pygame.transform.scale(
                road_img, (self.window_size / self.size, self.window_size / self.size)
            )
            road_img.convert()

            self.imgs["car"] = car_img
            self.imgs[0] = road_img
            self.imgs[1] = grass_img
            self.imgs[2] = start_img
            self.imgs[3] = finish_img
            self.imgs[4] = wall_img
            self.imgs[5] = ice_img

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self._draw_car()
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        """
        Exit pygame window
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
