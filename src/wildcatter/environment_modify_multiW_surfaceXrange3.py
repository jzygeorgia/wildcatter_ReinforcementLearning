"""Driller environment module."""


from __future__ import annotations

import random
from typing import Any

import numpy as np
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
from numpy.typing import NDArray


class ModifiedDriller(Env):  # type: ignore
    """Simple driller environment."""

    def __init__(self, env_config: dict[str, Any]) -> None:
        """Initialize environment with config dictionary."""
        self.model = np.loadtxt(
            env_config["model_path"],
            delimiter=env_config["delim"],
        )

        self.nrow, self.ncol = self.model.shape
        
        
        # multiple well drilling; well should not collide with exist well paths   

        if (env_config["exist_well_path"] == 'no') : 
            self.exist_well_path = []
        else:
            self.exist_well_path = (np.loadtxt(
                env_config["exist_well_path"],
                delimiter=env_config["delim"],
                ) ).tolist()

        print( 'Exist_well_path first points:\n', self.exist_well_path[ :1 ]  ) 

        print( 'Exist_well_path last points:\n', self.exist_well_path[-1: ]  ) 
        
        
        self.available_pipe = env_config["available_pipe"]

        print("\navailable piples:", self.available_pipe, '\n' )
        
        self.available_budget = env_config["available_budget"]
        
        print( "\navailable budget:", self.available_budget ) 
        
        
        self.surf_x_start = env_config["surf_x_start"]
        self.surf_x_end = env_config["surf_x_end"]
        
        print("\nuser input, surf_x_start: ", self.surf_x_start )
        
        print("user input, surf_x_end: ", self.surf_x_end )
        
        # x coord range at surface for well 
        self.surf_x_start = max( 0 , self.surf_x_start )
        self.surf_x_end = min( self.surf_x_end  , self.ncol - 1 )

        print("\nwithin earth model, surf_x_start: ", self.surf_x_start )
        
        print("within earth model, surf_x_end: ", self.surf_x_end )


        # production, affects pressure env and thus cost model etc and thus optimal well path    

        self.production = env_config["production"]

        print(f'\n\nself.production : {self.production} ')
        
        self.pipe_used = 0
        
        self.budget_used = 0 
        
        # cost model, related to production, as mentioned above.
        # also for considering economic constraint 
        # here we 'build' cost model as below; can also easily read in complex more realistic cost models provided by users 
        
        if self.production == 0 :   # no production yet in field 
            
            self.cost_model = np.ones( (self.nrow, self.ncol) ) * 18 # higher temp higher pressure more expensive 
            
            shallow = int ( self.nrow / 3 ) 
            mid = int( self.nrow * 2 / 3 ) 
        
            self.cost_model[:shallow,  : ] = 16     # normal pressure normal cost  
            self.cost_model[shallow:mid,  : ] =17  
            
        else:                  # field already has production, with depletion, cost model changes accordingly, thus wellpath
            
            self.cost_model = np.ones( (self.nrow, self.ncol) ) * 16.9
            
            shallow = int ( self.nrow / 3 ) 
            mid = int( self.nrow * 2 / 3 ) 
        
            self.cost_model[:shallow,  : ] = 16       # normal pressure normal cost 
            self.cost_model[shallow:mid,  : ] =16.5 
         
        
        print(f'self.cost_model[:, 2]:  {self.cost_model[:, 2] }' , )
        
        self.trajectory: list[list[int]] = []
        self.bit_location: list[int] = []

        self.action_space = Discrete(4)

        self.well_actions : list[int] = []   # agent's actions history, for avoiding 180 degree turn   
        
        self.observation_space = Box(
            low=0, high=1, shape=(self.nrow, self.ncol), dtype="bool"
        )
        self.reset()

    def step(  # noqa: C901
        self, action: int
    ) -> tuple[NDArray[np.bool_], int, bool, dict[str, Any]]:
        """Take step based on action."""
        done = False
        actions = {
            0: [1, 0],  # down
            1: [0, -1],  # left
            2: [0, 1],  # right
            3: [-1, 0],  # up
        }

        dz_dx = actions[action]
        new_location = [prev + now for prev, now in zip(self.bit_location, dz_dx)]

        self.bit_location = new_location


        # for avoiding 180 degree turn  
        self.well_actions.append( action ) 
        
        self.trajectory.append(new_location)
        newrow, newcol = new_location

        self.pipe_used += 1
        
        inside =  newrow >0 and newrow <self.nrow and newcol>= 0 and newcol <self.ncol 
        if ( inside ):
            self.budget_used += self.cost_model[ newrow, newcol ] # make sure it is inside the range of z and x
              
             

        if newrow < 1 or newrow >= self.nrow:  # should not drill out of earth model 
            done = True
            reward = -100

        elif newcol < 0 or newcol >= self.ncol:  # should not drill out of earth model 
            done = True
            reward = -100
            
        elif (  (3-action) in  self.well_actions[:-1]  ): #! avoid 180 degree turn (0 down vs 3 up; and 1 left vs 2 right) 
            done = True
            reward = -100
                    
        elif  (  self.model[newrow, newcol] == -100 ) :  #shallow gas, high pressure zone, fault to avoid; value -100 in model
            done = True
            reward = -100 
            
        elif ( self.bit_location in self.exist_well_path   ) :  # multi well, avoid collision 
            done = True
            reward = -100
    
        else:
            # reward values are from earth model  
            reward = self.model[newrow, newcol]
            
            self.update_state()

        if self.pipe_used == self.available_pipe:   # limited number of pipes 
            done = True
            reward = 0

        if self.budget_used >= self.available_budget: # economic constraint, limited budget  
            done = True 
            reward = 0 
            
        if self.bit_location in self.trajectory[:-1]:  # it also avoid immediate 180 degree 'wrap around' of path 
            done = True
            reward = -100
            
            
        info: dict[str, Any] = {}

        return self.state, reward, done, info

    def update_state(self) -> None:
        """Update state method."""
        traj_i, traj_j = np.asarray(self.trajectory).T
        self.state[traj_i, traj_j] = 1

    def render(self) -> None:
        """Gym environment rendering."""
        raise NotImplementedError("No renderer implemented yet.")

    def reset(self) -> NDArray[np.bool_]:
        """Reset the status of the environment."""
        #self.surface_hole_location = [1, random.randint(0, self.ncol - 1)]  # noqa: S311
        
        # within surface x coordinates range   
        self.surface_hole_location = [1, random.randint(self.surf_x_start, self.surf_x_end)]  
        
        self.state = np.zeros((self.nrow, self.ncol), dtype=bool)
        self.bit_location = self.surface_hole_location
        
        self.trajectory = [self.surface_hole_location]
        
        self.well_actions  = [ ]   # reset the history of actions which is used for avoiding 180 wrap around of well path 
            
        self.pipe_used = 0
        self.budget_used = 0 

                
        return self.state
