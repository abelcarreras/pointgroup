from itertools import permutations
import numpy as np


def get_cubed_sphere_grid_points(delta):
    """
    generate a cubed-grid points grid on the surface of an unitary sphere

    :param delta: max angle between points (radians)
    :return: list of points
    """

    num_points = int(1.0 / delta)

    if num_points < 1:
        return [(1, 0, 0)]

    for i in range(-num_points, num_points+1):
        x = i * delta
        for j in range(-num_points, num_points+1):
            y = j * delta
            for p in permutations([x, y, 1]):
                norm = np.linalg.norm([x, y, 1])
                yield np.array(p)/norm


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    data_0 = get_cubed_sphere_grid_points(np.deg2rad(10))
    data_0.append((0, 0, 0))

    print('{}\n'.format(len(data_0)))
    for x in data_0:
        print('H  {:10.5f} {:10.5f} {:10.5f}'.format(*x))


    def plot_sphere_points(points):
        # Create a 3D plot.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the sphere surface.
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

        # Plot the points on the sphere surface.
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        ax.scatter(xs, ys, zs, s=50, color='red')

        # Set the plot limits and labels.
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Sphere Points')

        # Show the plot.
        plt.show()


    points = []
    for i in range(10):
        # Generate random spherical coordinates.
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(2*np.random.uniform(0, 1) - 1)
        # Convert to cartesian coordinates.
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points.append((x, y, z))

    # Plot the points on the sphere

    plot_sphere_points(data_0)