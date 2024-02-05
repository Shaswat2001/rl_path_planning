import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import yaml

class GridEnvironment:
    def __init__(self, min_coords, max_coords, resolution):
        self.min_coords = np.array(min_coords)
        self.max_coords = np.array(max_coords)
        self.resolution = resolution
        
        self.grid_shape = ((self.max_coords - self.min_coords) // resolution) + 1
        self.grid_shape = self.grid_shape.astype(int)
        self.obs_dim = np.array([1,1,1])
        self.grid = np.zeros(self.grid_shape, dtype=np.int)  # Initialize grid
        self.obs_coordinates = []

    def reset(self):
        
        self.grid = np.zeros(self.grid_shape, dtype=np.int)
        self.obs_coordinates = []

    def add_obstacle(self, obstacle_ctr):
        obstacle_min = obstacle_ctr - self.obs_dim/2
        obstacle_max = obstacle_ctr + self.obs_dim/2
        obstacle_min_idx = ((obstacle_min - self.min_coords) // self.resolution).astype(int)
        obstacle_max_idx = ((obstacle_max - self.min_coords) // self.resolution).astype(int)
        self.obs_coordinates.append([obstacle_min, obstacle_max])
        self.grid[obstacle_min_idx[0]:obstacle_max_idx[0]+1, obstacle_min_idx[1]:obstacle_max_idx[1]+1] = 1

    def visualize_grid(self,axis = None):
        
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axis
        # Plot the grid
        grid_size = 7
        ax.set_xlim([-grid_size, grid_size])
        ax.set_ylim([-grid_size, grid_size])
        ax.set_zlim([-grid_size, grid_size])

        # Plot the obstacles
        for obstacle in self.obs_coordinates:
            min_coord, max_coord = obstacle
            vertices = np.array([
                [min_coord[0], min_coord[1], min_coord[2]],
                [min_coord[0], min_coord[1], max_coord[2]],
                [min_coord[0], max_coord[1], max_coord[2]],
                [min_coord[0], max_coord[1], min_coord[2]],
                [max_coord[0], min_coord[1], min_coord[2]],
                [max_coord[0], min_coord[1], max_coord[2]],
                [max_coord[0], max_coord[1], max_coord[2]],
                [max_coord[0], max_coord[1], min_coord[2]]
            ])
            
            # Define the faces of the cuboid
            faces = [[vertices[0], vertices[1], vertices[2], vertices[3]],
                    [vertices[4], vertices[5], vertices[6], vertices[7]], 
                    [vertices[0], vertices[1], vertices[5], vertices[4]], 
                    [vertices[2], vertices[3], vertices[7], vertices[6]], 
                    [vertices[1], vertices[2], vertices[6], vertices[5]], 
                    [vertices[4], vertices[7], vertices[3], vertices[0]]]
            
            # Plot the cuboid
            ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2))

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if axis is None:
            # Show the plot
            plt.show()

    def encode_state(self, agent_position):
        agent_position_idx = ((agent_position - self.min_coords) // self.resolution).astype(int)
        
        min_agent_position_idx = np.clip(agent_position_idx-1,0,self.grid_shape-1)
        max_agent_position_idx = np.clip(agent_position_idx+2,0,self.grid_shape-1)
        # Define the agent's local view, e.g., a 3x3 region centered around the agent
        local_view = self.grid[
            min_agent_position_idx[0]:max_agent_position_idx[0],
            min_agent_position_idx[1]:max_agent_position_idx[1],
            min_agent_position_idx[2]:max_agent_position_idx[2]
        ]
        
        for i,dim in enumerate(local_view.shape):

            difference = 3 - dim
            if difference == 0 or difference < 0:
                continue
            padding_shape = list(local_view.shape)
            padding_shape[i] = difference
            padding = np.zeros(padding_shape)

            local_view = np.concatenate((local_view,padding),axis=i)
            # print(f"region shape : {region.shape}")
            # print(f"the region has ({minX},{minY},{minZ}) and ({maxX},{maxY},{maxZ})")
        
        return local_view.flatten()

if __name__ == "__main__":
    pos_bound = np.array([7,7,7])
    grid = GridEnvironment(min_coords=-pos_bound,max_coords=pos_bound,resolution=0.1)

    with open("/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV/Config/obstacle_1.yaml", "r") as stream:
        try:
            obstacles = yaml.safe_load(stream)["obstacle_coordinates"]
        except yaml.YAMLError as exc:
            print(exc)

    for key,coordinate in obstacles.items():
        ctr_coordinate = coordinate
        print(f"Adding Obstacle : {key}")
        grid.add_obstacle(ctr_coordinate)

    grid.visualize_grid()
