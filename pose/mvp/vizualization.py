import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
import cv2
from matplotlib import gridspec

class CombinedVisualizer:
    def __init__(self, points_array, connections, video_path):
        self.points_array = points_array
        self.connections = connections
        self.video_path = video_path
        self.current_frame = 0
        self.playing = False
        self.slider_updating = False
        
        self.cap = cv2.VideoCapture(video_path)
        self.n_frames = min(len(points_array), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        self.fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        
        self.ax_video = plt.subplot(gs[0])
        self.ax_video.set_xticks([])
        self.ax_video.set_yticks([])
        
        self.ax_skel = plt.subplot(gs[1], projection='3d')
        
        self.z_min = points_array[:, :, 2].min()
        self.z_max = points_array[:, :, 2].max()
        
        self.ani = None
        self.setup_plot()
        self.setup_controls()
        
        ret, frame = self.cap.read()
        if ret:
            self.img_display = self.ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    def setup_plot(self):
        # Enable rotation and zooming for skeleton plot
        self.ax_skel.mouse_init()
        
        # Set specific limits for axes
        self.ax_skel.set_xlim([-500, 500])
        self.ax_skel.set_ylim([-2000, 2000])
        self.ax_skel.set_zlim([self.z_min, self.z_max])
        
        # Set labels
        self.ax_skel.set_xlabel('X (mm)')
        self.ax_skel.set_ylabel('Y (mm)')
        self.ax_skel.set_zlabel('Z (mm)')
        
    def setup_controls(self):
        # Play/Pause button
        play_ax = self.fig.add_axes([0.35, 0.05, 0.1, 0.075])
        self.play_button = Button(play_ax, 'Play')
        self.play_button.on_clicked(self.play_pause)
        
        # Reset button
        reset_ax = self.fig.add_axes([0.46, 0.05, 0.1, 0.075])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        # Frame slider
        slider_ax = self.fig.add_axes([0.35, 0.15, 0.3, 0.03])
        self.frame_slider = Slider(
            slider_ax, 'Frame', 0, self.n_frames-1,
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self.on_slider_changed)
        
    def update_plot(self, frame):
        # Update skeleton plot
        self.ax_skel.clear()
        self.setup_plot()
        
        points = self.points_array[frame]
        
        # Plot points
        self.ax_skel.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=50)
        
        # Draw connections
        for connection in self.connections:
            start = points[connection[0]]
            end = points[connection[1]]
            self.ax_skel.plot([start[0], end[0]], 
                            [start[1], end[1]], 
                            [start[2], end[2]], 
                            'r-', linewidth=2)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, video_frame = self.cap.read()
        if ret:
            self.img_display.set_array(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))
        
        self.ax_skel.set_title(f'Frame {frame}')
        
        if not self.slider_updating:
            self.slider_updating = True
            self.frame_slider.set_val(frame)
            self.slider_updating = False
            
        self.current_frame = frame
        
    def animation_update(self, frame):
        self.update_plot(frame)
        return self.ax_skel, self.img_display
        
    def play_pause(self, event):
        if self.playing:
            if self.ani is not None:
                self.ani.event_source.stop()
            self.play_button.label.set_text('Play')
        else:
            start_frame = self.current_frame
            frames = list(range(start_frame, self.n_frames)) + [0]  
            self.ani = FuncAnimation(
                self.fig, self.animation_update,
                frames=frames,
                interval=50, repeat=True
            )
            self.play_button.label.set_text('Pause')
        self.playing = not self.playing
        plt.draw()
        
    def reset(self, event):
        if self.playing:
            self.play_pause(None)
        self.current_frame = 0
        self.update_plot(0)
        plt.draw()
        
    def on_slider_changed(self, val):
        if not self.slider_updating:
            if self.playing:
                self.play_pause(None)
            self.update_plot(int(val))
            plt.draw()
        
    def show(self):
        self.update_plot(0)
        plt.show()
        
    def __del__(self):
        self.cap.release()


x = np.load('/home/enkwi/Desktop/Sapiens/mvp/data_npy/keypoints_boris_3d.npy', allow_pickle=True)
connections = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], 
               [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], 
               [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], 
               [0, 5], [0, 6]]

visualizer = CombinedVisualizer(x, connections, '/home/enkwi/Documents/Boris.mp4')

visualizer.show()



