import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata

def read_grid_data(filename):
    """Read grid data from file created by write_grid()"""
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    data = []
    x_coords = []
    y_coords = []
    
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        
        values = line.strip().split()
        if len(values) == 3:
            x, y, u = map(float, values)
            
            # Filter out junk values (-5.0)
            if u > -4.9:
                x_coords.append(x)
                y_coords.append(y)
                data.append(u)
    
    # Find grid dimensions
    unique_x = sorted(list(set(x_coords)))
    unique_y = sorted(list(set(y_coords)))
    nx = len(unique_x)
    ny = len(unique_y)
    
    # Create 2D grid from data
    grid = np.zeros((ny, nx))
    for i in range(len(data)):
        x_idx = unique_x.index(x_coords[i])
        y_idx = unique_y.index(y_coords[i])
        grid[y_idx, x_idx] = data[i]
    
    return grid, unique_x, unique_y

def plot_heatmap(filename, output_file=None, np_count=4):
    """Create a heatmap from grid file"""
    grid, x, y = read_grid_data(filename)
    
    # Create a higher resolution interpolated grid for smoother display
    x_fine = np.linspace(min(x), max(x), 400)
    y_fine = np.linspace(min(y), max(y), 400)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    
    # Convert original grid to meshgrid for interpolation
    X_orig, Y_orig = np.meshgrid(x, y)
    
    # Use bilinear interpolation for smoother visualization
    grid_fine = griddata((np.ravel(X_orig), np.ravel(Y_orig)), np.ravel(grid), 
                         (X_fine, Y_fine), method='cubic', fill_value=0)
    
    plt.figure(figsize=(10, 8))
    
    # Create a smoother heatmap using interpolated data
    im = plt.imshow(grid_fine, extent=[min(x), max(x), min(y), max(y)],
                    origin='lower', aspect='equal', cmap='viridis', interpolation='bicubic')
    
    # Add more tick marks for better readability
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    
    # Ensure axes limits show full domain
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Improved colorbar with more ticks
    cbar = plt.colorbar(im, label='u(x,y)')
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.title(f'Poisson Equation Solution Heatmap\n(31x31 Grid, {np_count} MPI Ranks, 2D Decomposition)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_q4.py <grid_file> [output_image] [np_count]")
        sys.exit(1)
    
    grid_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    np_count = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    plot_heatmap(grid_file, output_file, np_count)