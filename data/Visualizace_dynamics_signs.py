import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from modules.loaders import LoadDynamicSign

def visualize_dynamic_sign(t_total=2, use_dataframe=False, df=None, roi = False):
    """
    Visualizes an animation of the dynamic sign frames with connected points.

    Args:
        dinamic_sign (str or None): Name of the dynamic sign (if not using a dataframe).
        number (str or None): Number of the dynamic sign (if not using a dataframe).
        t_total (float): Total duration of the animation in seconds.
        use_dataframe (bool): If True, uses the dataframe passed as an argument.
        df (pd.DataFrame or None): Dataframe with the dynamic sign data.
    """
    # Load data
    if not use_dataframe:
        dinamic_sign = input("Enter the name of the dynamic sign: ")
        number = input("Enter the number of the dynamic sign: ")
        df = LoadDynamicSign(dinamic_sign, number).load_dynamic_sign()
    elif df is None:
        raise ValueError("If use_dataframe is True, a valid 'df' must be provided.")
        
    # Initial parameters
    n_points = 21  # key points per frame
    n_frames = len(df['cx']) // n_points  # number of frames
    speed = t_total / n_frames * 1000  # animation speed (ms)

    # Data
    if roi:
        x = df['cxROI']
        y = df['cyROI']
    else:
        x = df['cx']
        y = df['cy']

    # Create figure and axes
    fig, ax = plt.subplots()
    sc, = ax.plot([], [], 'ro')  # Red points, no lines initially

    # Configure axis limits
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_title("Dynamic Frame Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_axis_off()
    ax.invert_yaxis()

    # Connections between points
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), 
        (0, 5), (5, 6), (6, 7), (7, 8), 
        (0, 9), (9, 10), (10, 11), (11, 12), 
        (0, 13), (13, 14), (14, 15), (15, 16), 
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    # To store the drawn lines
    lines = []

    def update(frame):
        """Updates the points and lines per frame."""
        nonlocal lines
        start = n_points * frame
        end = n_points * (frame + 1)

        # Clear previous lines
        for line in lines:
            line.remove()
        lines.clear()

        # Update point data
        sc.set_data(x[start:end], y[start:end])

        # Draw lines between connected points
        for (i, j) in connections:
            line, = ax.plot([x[start + i], x[start + j]], [y[start + i], y[start + j]], 'g-')
            lines.append(line)  # Save the line to remove it later
        ax.set_title(f"Frame: {frame + 1}/{n_frames}  Time: {round((frame*speed*10**(-3)),2)}s FPS: {round(1/(speed*10**(-3)),2)}")
        return sc,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=n_frames, interval=speed, blit=False)

    # Show the animation
    plt.show()


if __name__ == "__main__":
    # Example of using with dataframe
    df = pd.DataFrame({
        'cx': np.random.rand(21 * 10),  # 10 frames, 21 keypoints per frame
        'cy': np.random.rand(21 * 10),
        'cxROI': np.random.rand(21 * 10),
        'cyROI': np.random.rand(21 * 10)
    })
    usedataframe = input("Use dataframe? (y/n): ")
    usedataframe = usedataframe.lower()
    if usedataframe == "n":
        visualize_dynamic_sign(use_dataframe=False, df=df,t_total=3,roi=False)
    else:
        visualize_dynamic_sign(use_dataframe=True, df=df)
