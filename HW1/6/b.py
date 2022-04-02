import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


cmap = ['dimgrey', 'silver', 'cyan', 'yellow','orange', 'lime', 'darkblue', 'blueviolet', 'red']
my_cmap = matplotlib.colors.ListedColormap(cmap, name='my_colormap_name')

state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [7, 0, 0, 3, 3, 5, 0, 0, 0, 0],
                  [7, 7, 8, 3, 3, 5, 5, 0, 0, 0],
                  [7, 8, 8, 7, 8, 8, 5, 0, 0, 0],
                  [5, 8, 4, 7, 7, 8, 8, 4, 0, 0],
                  [5, 5, 4, 7, 3, 3, 6, 4, 0, 0],
                  [6, 5, 4, 4, 3, 3, 6, 4, 4, 0],
                  [6, 6, 6, 4, 7, 6, 6, 8, 2, 0],
                  [2, 4, 4, 4, 7, 7, 8, 8, 2, 0],
                  [2, 3, 3, 5, 7, 2, 8, 8, 2, 0]])

T = np.array([[7, 7, 7],
              [0, 7, 0],
              [0, 0, 0]])

L = np.array([[0, 6, 0],
              [0, 6, 0],
              [0, 6, 6]])

S = np.array([[0, 5, 5],
              [5, 5, 0],
              [0, 0, 0]])

I = np.array([[0, 2, 0, 0],
              [0, 2, 0, 0],
              [0, 2, 0, 0],
              [0, 2, 0, 0]])

O = np.array([[3, 3],
              [3, 3]])

def rotate(a, n):
    return np.rot90(a, -1 * n)

def show_state(state, piece, dir, x, y):
    if dir == 5:
        state = delete_line(state.copy(),0)
    else:
        state = state.copy()
        piece = rotate(piece, dir)
        state[y : y + piece.shape[0], x : x + piece.shape[1]] = piece + state[y : y + piece.shape[0], x : x + piece.shape[1]];
    fig, ax = plt.subplots()
    output_state = np.ones([22,12])
    output_state[1:21,1:11] = state
    ax.imshow(output_state, cmap=my_cmap)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.51, 21, 1), minor=True)
    ax.grid(which='minor', color='dimgrey', linestyle='-', linewidth=0.7)
    plt.show()
    return fig,state

def delete_line(state, line):
    line = 19 - line
    state[line,:] = 0
    state[:line+1,:] = np.roll(state[:line+1,:],1,axis=0)
    return state

_,state = show_state(state,T,1,4,9)

sequence = [[L, 0, 4, 0],
            [L, 0, 4, 1],
            [L, 0, 3, 2],
            [L, 0, 2, 3],
            [L, 0, 2, 4],
            [L, 0, 1, 5],
            [L, 0, 1, 6],
            [L, 0, 0, 7],
            [L, 0, 0, 8],
            [L, 0, 0, 9]]

if __name__ == "__main__":
    fram = -1
    def make_frame(t):
        global fram
        print(fram)
        fig,_ = show_state(state, sequence[fram][0], sequence[fram][1], sequence[fram][2], sequence[fram][3])
        fram += 1
        return mplfig_to_npimage(fig)


    animation = VideoClip(make_frame, duration = len(sequence)/2)
    animation.ipython_display(fps = 2, loop = True, autoplay = True)