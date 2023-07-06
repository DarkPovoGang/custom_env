import gym
from gym import spaces
# import pygame
import numpy as np
import torchvision
import torch
from custom_env.dataset import RefCOCOg
from functools import total_ordering

from enum import Enum

@total_ordering
class Actions(Enum):
  ACT_RT = 0 #Right
  ACT_LT = 1 #Left
  ACT_UP = 2 #Up
  ACT_DN = 3 #Down
  ACT_TA = 4 #Taller
  ACT_FA = 5 #Fatter
  ACT_SR = 6 #Shorter
  ACT_TH = 7 #Thiner
  ACT_TR = 8 #Trigger

  def __lt__(self, other):
    if self.__class__ is other.__class__:
      return self.value < other.value

class VisualGroundingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    CONVERGENCE_THRESHOLD = 0.95
    def __init__(self, split, move_factor=0.2, scale_factor=0.1, render_mode=None):
        # self.window_size = 512  # The size of the PyGame window
        self.move_factor = move_factor
        self.scale_factor = scale_factor
        self.dataset = RefCOCOg('.',split)
        self.idx = 0
        _, _, self.width, self.height = self.dataset[self.idx]
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0, 0, 1, 1]), high=np.array([self.width-1, self.height, self.width, self.height]), dtype=int),
                "target": spaces.Box(low=np.array([0, 0, 1, 1]), high=np.array([self.width-1, self.height, self.width, self.height]), dtype=int),
            }
        )

        # We have 9 actions, corresponding to "right", "up", "left", "down", "v-shrink", "v-stretch", "h-shrink", "h-stretch", "confirm"
        self.action_space = spaces.Discrete(9)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        # TODO: maybe return current history of movement
        return {"iou": self.iou}

    def reset(self, seed=None, options=None):
        self.x1 = 0
        self.y1 = 0

        # We need the following line to seed self.np_random
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Choose the agent's location uniformly
        self._agent_location = torch.tensor([[0, 0, self.width, self.height]], dtype=torch.float32)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        _, bbox, width, height = self.dataset[self.idx]
        print("RESET: processing IMG with bbox: ",bbox )
        self.idx = self.idx + 1
        bbox_x2 = bbox[0]+bbox[2]
        bbox_y2 = bbox[1]+bbox[3]
        self._target_location = torch.tensor([ [bbox[0],bbox[1],bbox_x2,bbox_y2 ] ], dtype=torch.float32)

        self.width = width
        self.height = height
        self.bbox_width = self.width
        self.bbox_height = self.height
        
        self.current_iou = torchvision.ops.box_iou(self._agent_location ,self._target_location)[0].item()
        self.iou = torchvision.ops.box_iou(self._agent_location ,self._target_location)[0].item()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _update_bbox(self, action):
      ALPHA = 0.2
      BETA  = 0.1
      x2 = self.x1 + self.bbox_width
      y2 = self.y1 + self.bbox_height
      assert action >= Actions.ACT_RT.value and action <= Actions.ACT_TR.value
      if not hasattr(self, 'action_history'):
        self.action_history = torch.tensor([action])
      else:
        self.action_history = torch.cat((self.action_history, torch.tensor([action])))

      if action <= Actions.ACT_DN.value:
        delta_w = int(ALPHA * self.bbox_width)
        delta_h = int(ALPHA * self.bbox_height)
      else:
        delta_w = int(BETA * self.bbox_width)
        delta_h = int(BETA * self.bbox_height)

      # PREVENT_STUCK:
      if (delta_h == 0):
        delta_h = 1
      if (delta_w == 0):
        delta_w = 1

      # print(action)

      #Do the corresponding action to the window
      if action == Actions.ACT_RT.value:
        self.x1 += delta_w
        x2 += delta_w
      elif action == Actions.ACT_LT.value:
        self.x1 -= delta_w
        x2 -= delta_w
      elif action == Actions.ACT_UP.value:
        self.y1 -= delta_h
        y2 -= delta_h
      elif action == Actions.ACT_DN.value:
        self.y1 += delta_h
        y2 += delta_h
      elif action == Actions.ACT_TA.value:
        self.y1 -= delta_h
        y2 += delta_h
      elif action == Actions.ACT_FA.value:
        self.x1 -= delta_w
        x2 += delta_w
      elif action == Actions.ACT_SR.value:
        self.y1 += delta_h
        y2 -= delta_h
      elif action == Actions.ACT_TH.value:
        self.x1 += delta_w
        x2 -= delta_w
      elif action == Actions.ACT_TR.value:
        pass
      else:
        raise ValueError('Invalid action')

      # ensure bbox inside image
      if self.x1 < 0:
        self.x1 = 0
      if self.y1 < 0:
        self.y1 = 0
      if x2 >= self.width:
        x2 = self.width - 1
      if y2 >= self.height:
        y2 = self.height - 1
      # ret x,y,w,h
      return torch.as_tensor([[self.x1, self.y1, x2, y2]], dtype=torch.float32)


    def step(self, action):
        
        terminated = False
        reward = 0 
        # COMPUTE REWARD
        # 0 <= x1 < x2 and 0 <= y1 < y2. 
        # print(" agent: ", self._agent_location)
        # print(" target: ", self._target_location)
       
        self.current_iou = torchvision.ops.box_iou(self._agent_location ,self._target_location)[0].item()
        # print("current_iou: ",self.current_iou)
        # print("------")
        # print('iou : ',self.current_iou)

        # if self.current_iou > VisualGroundingEnv.CONVERGENCE_THRESHOLD: 
        #     terminated = True
        
        #Get reward
        if action < Actions.ACT_TR.value:
          if self.current_iou > self.iou:
            reward = self.current_iou
          else:
            reward = -0.05
        else:
          if self.current_iou < 0.5:
            reward = -1.0
          else:
            reward = 1.0
        
        assert reward != 0
        
        self.iou = self.current_iou
        # print("current iou is ",self.current_iou)

        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self._agent_location = self._update_bbox(action)
        # An episode is done iff the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location) #TODO: or quite close
        # print("tensors_shape: ", self._agent_location.shape)  
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
       return None
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     pygame.display.init()
        #     self.window = pygame.display.set_mode((self.window_size, self.window_size))
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()

        # canvas = pygame.Surface((self.window_size, self.window_size))
        # canvas.fill((255, 255, 255))
        # pix_square_size = (
        #     self.window_size / self.size
        # )  # The size of a single grid square in pixels

        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        #)

        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        # if self.render_mode == "human":
        #     # The following line copies our drawings from `canvas` to the visible window
        #     self.window.blit(canvas, canvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()

        #     # We need to ensure that human-rendering occurs at the predefined framerate.
        #     # The following line will automatically add a delay to keep the framerate stable.
        #     self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )

    def close(self):
        pass
        # if self.window is not None:
            # pygame.display.quit()
            # pygame.quit()
