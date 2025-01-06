import os.path
import tkinter as tk
from tkinter import filedialog, messagebox, Menu
import chardet
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import zoom
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from sklearn.neighbors import NearestNeighbors
from format_process import is_empty
from match_algorithm import exact_match,cpd,cpd_fully_auto
from marker_handler import read_marker,filter_marker
from apo_handler import read_apo,filter_apo
import json


CONFIG_FILE = 'settings/config.json'
CONFIG_READ_FILE = 'settings/config_read.json'
MATRIX_FILE = 'settings/matrix.json'


class MatchWindow:
    def __init__(self):
        # Create and place widgets
        self.root = tk.Tk()
        self.root.title("匹配程序")

        # Default step sizes
        self.translation_step = 5
        self.rotation_step = 5
        self.scale_step = 1.1

        self.offset = [0, 0]
        self.scale = 1.0
        self.angle = 0.0
        self.read_matrix()

        self.transformation_matrix = np.eye(3)

        self.algorithm = tk.IntVar(value=4)
        self.option_point_to_point = tk.BooleanVar(value=False)

        # Create ui
        self.initialize_ui()

        # Load config
        self.config = self.load_config()

        # Default variables
        self.origin_data = read_table(self.origin_file_var.get())
        self.transformed_data = self.origin_data.copy()
        self.target_data = read_table(self.target_file_var.get())
        self.selected_indices = []
        self.transformed_plot = None
        self.result = None
        self.update_plot()

        # Start the main event loop
        self.root.mainloop()

    def initialize_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.columnconfigure(3, weight=1)
        self.root.columnconfigure(4, weight=1)
        self.root.columnconfigure(5, weight=1)
        self.root.columnconfigure(6, weight=1)
        self.root.columnconfigure(7, weight=1)
        self.root.columnconfigure(8, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=1)
        self.root.rowconfigure(5, weight=1)
        self.root.rowconfigure(6, weight=1)
        self.root.rowconfigure(7, weight=1)
        self.root.rowconfigure(8, weight=1)
        self.root.rowconfigure(9, weight=1)
        self.root.rowconfigure(10, weight=1)

        # file widgets
        tk.Label(self.root, text="选择原坐标:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.origin_file_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.origin_file_var, width=50).grid(row=0, column=1, columnspan=7, padx=10,
                                                                              pady=10,
                                                                              sticky='we')
        tk.Button(self.root, text="Browse...", command=self.select_origin_path).grid(row=0, column=8, padx=10, pady=10,
                                                                                     sticky='w')

        tk.Label(self.root, text="选择目标坐标:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.target_file_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.target_file_var, width=50).grid(row=1, column=1, columnspan=7, padx=10,
                                                                              pady=10,
                                                                              sticky='we')
        tk.Button(self.root, text="Browse...", command=self.select_target_path).grid(row=1, column=8, padx=10, pady=10,
                                                                                     sticky='w')

        tk.Label(self.root, text="选择输出路径:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.output_folder_var = tk.StringVar()
        tk.Entry(self.root, textvariable=self.output_folder_var, width=50).grid(row=2, column=1, columnspan=7, padx=10,
                                                                                pady=10,
                                                                                sticky='we')
        tk.Button(self.root, text="Browse...", command=self.select_output_folder).grid(row=2, column=8, padx=10,
                                                                                       pady=10,
                                                                                       sticky='w')

        tk.Label(self.root, text="选择匹配方式:").grid(row=3, column=0, padx=10, pady=10, sticky='e')
        tk.Radiobutton(self.root, text="rigid", value=0, variable=self.algorithm).grid(row=3, column=1, padx=10, pady=10,
                                                                                  sticky='e')
        tk.Radiobutton(self.root, text="non-rigid", value=1, variable=self.algorithm).grid(row=3, column=2, padx=10, pady=10,
                                                                                   sticky='e')
        tk.Radiobutton(self.root, text="no process", value=2, variable=self.algorithm).grid(row=3, column=3, padx=10, pady=10,
                                                                                        sticky='e')
        tk.Radiobutton(self.root, text="fully auto", value=3, variable=self.algorithm).grid(row=3, column=4, padx=10,
                                                                                            pady=10,
                                                                                            sticky='e')
        tk.Radiobutton(self.root, text="fully auto (save edge points)", value=4, variable=self.algorithm).grid(row=3, column=5, padx=10,
                                                                                            pady=10,
                                                                                            sticky='e')

        tk.Checkbutton(self.root, text="point to point", variable=self.option_point_to_point).grid(row=3, column=6,
                                                                                                     sticky='e',
                                                                                                     padx=10)

        # visual widgets
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=4, column=0, rowspan=13, columnspan=6, sticky='we')
        self.scatter_selector = RectangleSelector(self.ax, self.on_select,
                                                  useblit=True,
                                                  button=[1], minspanx=5, minspany=5,
                                                  spancoords='pixels', interactive=True)
        self.canvas.get_tk_widget().bind("<Button-3>", self.show_context_menu)
        self.context_menu = Menu(self.root, tearoff=0)
        self.context_menu.add_command(label="delete outlier in origin", command=self.extract_outlier_in_origin)
        self.context_menu.add_command(label="delete outlier in target", command=self.extract_outlier_in_target)
        self.context_menu.add_command(label="random downsample half points", command=self.random_downsample)

        # control widgets
        auto_center_button = tk.Button(self.root, text="Auto Center", command=self.auto_center)
        auto_center_button.grid(row=4, column=6, columnspan=3, padx=50, sticky='we')

        horizontal_flip_button = tk.Button(self.root, text="Horizontal Flip", command=self.horizontal_flip)
        horizontal_flip_button.grid(row=5, column=6, columnspan=3, padx=50, sticky='we')

        vertical_flip_button = tk.Button(self.root, text="Vertical Flip", command=self.vertical_flip)
        vertical_flip_button.grid(row=6, column=6, columnspan=3, padx=50, sticky='we')

        tk.Label(self.root, text="Translation Step:").grid(row=7, column=6, padx=10, pady=10, sticky='e')
        self.translation_step_entry = tk.Entry(self.root)
        self.translation_step_entry.insert(0, str(self.translation_step))
        self.translation_step_entry.grid(row=7, column=7, columnspan=2, padx=10, pady=10, sticky='we')

        tk.Label(self.root, text="Rotation Step:").grid(row=8, column=6, padx=10, pady=10, sticky='e')
        self.rotation_step_entry = tk.Entry(self.root)
        self.rotation_step_entry.insert(0, str(self.rotation_step))
        self.rotation_step_entry.grid(row=8, column=7, columnspan=2, padx=10, pady=10, sticky='we')

        tk.Label(self.root, text="Scale Step:").grid(row=9, column=6, padx=10, pady=10, sticky='e')
        self.scale_step_entry = tk.Entry(self.root)
        self.scale_step_entry.insert(0, str(self.scale_step))
        self.scale_step_entry.grid(row=9, column=7, columnspan=2, padx=10, pady=10, sticky='we')

        output_button = tk.Button(self.root, text="Output Matrix", command=self.output_matrix)
        output_button.grid(row=10, column=6, columnspan=3, padx=50, sticky='we')

        run_match_button = tk.Button(self.root, text="Run Match", command=self.run_match)
        run_match_button.grid(row=11, column=6, columnspan=3, padx=50, sticky='we')

        continue_button = tk.Button(self.root, text="continue", command=self.restore_image)
        continue_button.grid(row=12, column=6, columnspan=3, padx=50, sticky='we')

        tk.Label(self.root, text="use 'W' 'A' 'S' 'D' to translate, '←' '→' to rotate, '↑' '↓' to zoom")\
            .grid(row=13, column=6, columnspan=3, padx=20, pady=20, sticky='we')

        delete_point_button = tk.Button(self.root, text="delete", command=self.delete_selected)
        delete_point_button.grid(row=14,column=6,padx=20,pady=20, sticky='we')
        save_point_button = tk.Button(self.root, text="save", command=self.save_point)
        save_point_button.grid(row=14, column=7, padx=10, pady=10, sticky='we')
        reset_select_button = tk.Button(self.root, text="reset select", command=self.reset_select)
        reset_select_button.grid(row=14, column=8, padx=10, pady=10, sticky='we')

        self.canvas.get_tk_widget().bind("<Key>", self.on_key)

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def select_origin_path(self):
        file_selected = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"),("Excel files", "*.xlsx"),
                                                              ("Marker files", "*.marker"),("Matlab files", ".mat"),
                                                              ("Polygon files", "*.ply"),("APO files",".apo")])
        if file_selected:
            self.update_config('origin_path', file_selected)
            self.origin_file_var.set(file_selected)
            self.origin_data = read_table(self.origin_file_var.get())
            self.transformed_data = self.origin_data.copy()
            self.update_plot()

    def select_target_path(self):
        file_selected = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"),("Excel files", "*.xlsx"),
                                                              ("Marker files", "*.marker"),("Matlab files", ".mat"),
                                                              ("Polygon files", "*.ply"),("APO files",".apo")])
        if file_selected:
            self.update_config('target_path', file_selected)
            self.target_file_var.set(file_selected)
            self.target_data = read_table(self.target_file_var.get())
            self.update_plot()

    def select_output_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.update_config('output_folder', folder_selected)
            self.output_folder_var.set(folder_selected)

    def update_config(self, key, value):
        self.config = self.load_config()
        self.config[key] = value
        self.save_config()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as config_file:
                config = json.load(config_file)
                self.origin_file_var.set(config.get('origin_path', ''))
                self.target_file_var.set(config.get('target_path', ''))
                self.output_folder_var.set(config.get('output_folder', ''))
                return config
        return {}

    def save_config(self):
        with open(CONFIG_FILE, 'w') as config_file:
            json.dump(self.config, config_file, indent=4)

    # point control function
    """
    针对所有点的操作会被记录缓存
    针对局部点的操作不会被记录
    删除操作会直接对原始数据进行
    """
    def auto_center(self):
        x, y = self.transformed_data['X'], self.transformed_data['Y']
        center_x, center_y = np.mean(x), np.mean(y)
        x_target, y_target = self.target_data['X'], self.target_data['Y']
        target_center_x, target_center_y = np.mean(x_target), np.mean(y_target)

        add_x, add_y = target_center_x - center_x, target_center_y - center_y

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        x_target_range = x_target.max() - x_target.min()
        y_target_range = y_target.max() - y_target.min()

        add_scale = 1
        if x_range > 0.1 and y_range > 0.1:
            add_scale = max(x_target_range/x_range, y_target_range/y_range)

        self.update_transformed_data(0, add_scale, add_x, add_y)

    def horizontal_flip(self):
        self.update_transformed_data(0, -1)

    def vertical_flip(self):
        self.update_transformed_data(180, -1)

    def read_matrix(self):
        if os.path.exists(MATRIX_FILE):
            with open(MATRIX_FILE) as config_read_file:
                config_read = json.load(config_read_file)
                self.offset = config_read.get('offset', [0, 0])
                self.scale = config_read.get('scale', 1.0)
                self.angle = config_read.get('angle', 0.0)

    def output_matrix(self):
        print("Transformation Matrix:")
        print(self.transformation_matrix)
        with open(MATRIX_FILE, 'w') as config_file:
            config = {'offset': self.offset, 'scale': self.scale, 'angle': self.angle}
            json.dump(config, config_file, indent=4)

    def run_match(self):
        # file name process
        target_file_name = os.path.basename(self.target_file_var.get())  # 不包含路径
        target_file = os.path.splitext(target_file_name)[0]  # 不包含后缀
        output_folder = self.output_folder_var.get()
        output_file_name = target_file + '_result.csv'
        output_file_path = os.path.join(output_folder, output_file_name)
        validate_image_name = target_file + '_validate.png'
        validate_image_path = os.path.join(output_folder, validate_image_name)

        method = self.algorithm.get()
        if method == 0:
            self.transformed_data = cpd(self.transformed_data,self.target_data)
        elif method == 1:
            self.transformed_data = cpd(self.transformed_data, self.target_data,non_rigid=True)
        elif method == 3:
            self.auto_center()
            self.extract_outlier_in_origin()
            self.extract_outlier_in_origin()
            self.extract_outlier_in_target()
            self.extract_outlier_in_target()
            self.transformed_data = cpd_fully_auto(self.transformed_data, self.target_data)
        elif method == 4:
            self.auto_center()
            self.transformed_data = cpd_fully_auto(self.transformed_data, self.target_data)

        self.update_plot()

        # 是否精确到点对点
        is_p2p = self.option_point_to_point.get()
        if is_p2p:
            self.result = exact_match(self.transformed_data, self.target_data)
            # 这里还需要把result中的origin坐标替换下 但暂时没必要
            self.result.to_csv(output_file_path)
            self.save_validate_image(validate_image_path)

    def restore_image(self):
        self.update_plot()

    def on_key(self, event):
        try:
            self.translation_step = int(self.translation_step_entry.get())
            self.rotation_step = int(self.rotation_step_entry.get())
            self.scale_step = float(self.scale_step_entry.get())
        except ValueError:
            print("Invalid input for step size. Using default values.")
            self.translation_step = 5
            self.rotation_step = 5
            self.scale_step = 1.1
        add_angle, add_scale, add_x, add_y = 0,1,0,0
        if event.char == 'w' or event.char == 'W':
            add_y = self.translation_step
        elif event.char == 's' or event.char == 'S':
            add_y = -self.translation_step
        elif event.char == 'a' or event.char == 'A':
            add_x = -self.translation_step
        elif event.char == 'd' or event.char == 'D':
            add_x = self.translation_step
        elif event.keysym == 'Up':
            add_scale = self.scale_step
        elif event.keysym == 'Down':
            add_scale = (1/self.scale_step) if (abs(self.scale_step)>1E-3) else 1
        elif event.keysym == 'Left':
            add_angle = -self.rotation_step * np.sign(self.scale)
        elif event.keysym == 'Right':
            add_angle = self.rotation_step * np.sign(self.scale)
        else:
            return
        self.update_transformed_data(add_angle,add_scale,add_x,add_y)

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        mask = (self.transformed_data['X'] >= min(x1, x2)) & (self.transformed_data['X'] <= max(x1, x2)) & \
               (self.transformed_data['Y'] >= min(y1, y2)) & (self.transformed_data['Y'] <= max(y1, y2))
        selected = np.where(mask)[0]
        self.selected_indices.extend(selected)
        self.selected_indices = list(set(self.selected_indices))
        colors = ['red' if i in self.selected_indices else 'blue' for i in range(len(self.transformed_data))]
        if self.transformed_plot is not None:
            self.transformed_plot.set_color(colors)
            self.canvas.draw()

    def delete_selected(self):
        if self.selected_indices:
            self.origin_data = self.origin_data.drop(index=self.selected_indices).reset_index(drop=True)
            self.transformed_data = self.transformed_data.drop(index=self.selected_indices).reset_index(drop=True)
            self.selected_indices = []
            self.update_plot()
        else:
            messagebox.showinfo("Info", "No coordinates selected")

    def save_point(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            self.origin_data.to_csv(file_path, index=False)
            self.update_config('origin_path', file_path)
            self.origin_file_var.set(file_path)
            messagebox.showinfo("Info", "Coordinates saved successfully")

    def reset_select(self):
        self.selected_indices = []
        if self.transformed_plot is not None:
            self.transformed_plot.set_color('blue')
            self.canvas.draw()

    def update_transformed_data(self, add_angle=0, add_scale=1, add_x=0, add_y=0):
        if self.selected_indices is None or not self.selected_indices:
            x, y, z = self.transformed_data['X'], self.transformed_data['Y'], self.transformed_data['Z']
            center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
            x, y = x - center_x, y - center_y

            x_rot = x.copy()
            y_rot = y.copy()
            add_angle *= -np.sign(self.scale*add_scale)
            if add_angle != 0:
                self.angle += add_angle
                theta = np.radians(add_angle)
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                x_rot = cos_theta * x - sin_theta * y
                y_rot = sin_theta * x + cos_theta * y

            self.scale *= add_scale
            x_rot *= add_scale
            y_rot *= abs(add_scale)
            z_rot = z
            if add_scale < 0:
                z_rot = 2 * center_z - z

            self.offset[0] += add_x
            self.offset[1] += add_y
            x_rot += center_x + add_x
            y_rot += center_y + add_y

            self.transformed_data['X'] = x_rot
            self.transformed_data['Y'] = y_rot
            self.transformed_data['Z'] = z_rot
        else:
            # local transform
            selected_data = self.transformed_data.iloc[self.selected_indices]
            x, y, z = selected_data['X'], selected_data['Y'], selected_data['Z']
            center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)
            x, y = x - center_x, y - center_y

            x_rot = x.copy()
            y_rot = y.copy()
            if add_angle != 0:
                self.angle += add_angle
                theta = np.radians(add_angle)
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                x_rot = cos_theta * x - sin_theta * y
                y_rot = sin_theta * x + cos_theta * y

            self.scale *= add_scale
            x_rot *= add_scale
            y_rot *= abs(add_scale)
            z_rot = z
            if add_scale < 0:
                z_rot = 2 * center_z - z

            self.offset[0] += add_x
            self.offset[1] += add_y
            x_rot += center_x + add_x
            y_rot += center_y + add_y
            self.transformed_data.loc[self.selected_indices, 'X'] = x_rot
            self.transformed_data.loc[self.selected_indices, 'Y'] = y_rot
            self.transformed_data.loc[self.selected_indices, 'Z'] = z_rot

        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        if self.target_data is not None:
            self.ax.scatter(self.target_data['X'], self.target_data['Y'], c='gray', s=10)
        if self.transformed_data is not None:
            self.transformed_plot = self.ax.scatter(self.transformed_data['X'], self.transformed_data['Y'], c='blue', s=8)
        self.canvas.draw()

    def save_validate_image(self, save_path):
        if self.result is not None:
            for i in self.result.index:
                id1,x1,y1,id2,x2,y2 = self.result.loc[i,['Id_origin','X_origin','Y_origin','Id_target','X_target','Y_target']]
                if id1 and id2:
                    plt.arrow(x1, y1, x2 - x1, y2 - y1, color='green', head_width=1, length_includes_head=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # extra function in menu
    def statistical_outlier_filtering(self, df, k=5, std_dev_multiplier=2.0):
        # 提取 X, Y, Z 坐标
        points = df[['X', 'Y', 'Z']].values

        # 使用最近邻计算每个点的 k 个邻居距离
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)

        # 计算每个点的平均邻居距离（去掉第一个点本身）
        mean_distances = distances[:, 1:].mean(axis=1)

        # 计算平均距离的全局均值和标准差
        mean_mean_distance = np.mean(mean_distances)
        std_mean_distance = np.std(mean_distances)

        # 离群点判断条件：平均距离超过 mean + std_dev_multiplier * std
        threshold = mean_mean_distance + std_dev_multiplier * std_mean_distance
        mask_outliers = mean_distances > threshold

        # 分离内点和离群点
        # df_inliers = df[~mask_outliers].reset_index(drop=True)
        # df_outliers = df[mask_outliers].reset_index(drop=True)

        return mask_outliers

    def extract_outlier_in_origin(self):
        mask_origin = self.statistical_outlier_filtering(self.transformed_data, k=5, std_dev_multiplier=2.0)
        self.origin_data = self.origin_data[~mask_origin].reset_index(drop=True)
        self.transformed_data = self.transformed_data[~mask_origin].reset_index(drop=True)
        self.update_plot()

    def extract_outlier_in_target(self):
        mask_target = self.statistical_outlier_filtering(self.target_data, k=5, std_dev_multiplier=2.0)
        self.target_data = self.target_data[~mask_target].reset_index(drop=True)
        self.update_plot()

    def randomly_choose_half(self,df):
        num_rows = len(df)
        num_to_remove = num_rows // 2
        rows_to_remove = np.random.choice(df.index, size=num_to_remove, replace=False)
        return rows_to_remove

    def random_downsample(self):
        mask = self.randomly_choose_half(self.origin_data)
        self.origin_data = self.origin_data.drop(index=mask)
        self.transformed_data = self.transformed_data.drop(index=mask)
        print("remove",len(mask),"points in origin data")
        mask = self.randomly_choose_half(self.target_data)
        self.target_data = self.target_data.drop(index=mask)
        print("remove",len(mask),"points in target data")
        self.update_plot()

def load_read_config():
    if os.path.exists(CONFIG_READ_FILE):
        with open(CONFIG_READ_FILE, 'r') as config_read_file:
            config_read = json.load(config_read_file)
            saved_status = config_read.get('saved_status', ["Perfused", "Init", "Perfusing"])
            dropped_intensities = config_read.get('dropped_intensities', [-1, -2, -3])
            return saved_status, dropped_intensities
    return ["Perfused", "Init", "Perfusing"], [-1, -2, -3]

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
    return result['encoding']


def read_csv_with_detected_encoding(file_path):
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")
    try:
        df = pd.read_csv(file_path, encoding=encoding, index_col=None)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ANSI', index_col=None)
    return df

def filter_csv_data(data):
    global saved_status, dropped_intensities
    saved_lines = []
    if 'Status' in data.columns:
        if 'Intensity' in data.columns:
            for i in data.index:
                if (data.loc[i, 'Status'] in saved_status
                        and not data.loc[i, 'Intensity'] in dropped_intensities):
                    saved_lines.append(i)
        elif 'current_intensity' in data.columns:
            for i in data.index:
                if (data.loc[i, 'Status'] in saved_status
                        and not data.loc[i, 'current_intensity'] in dropped_intensities):
                    saved_lines.append(i)
        else:
            for i in data.index:
                if data.loc[i, 'Status'] in saved_status:
                    saved_lines.append(i)
    else:
        if 'Intensity' in data.columns:
            for i in data.index:
                if not data.loc[i, 'Intensity'] in dropped_intensities:
                    saved_lines.append(i)
        elif 'current_intensity' in data.columns:
            for i in data.index:
                if not data.loc[i, 'current_intensity'] in dropped_intensities:
                    saved_lines.append(i)
        else:
            return data
    if 'dye_name' in data.columns:
        data = data.loc[saved_lines, ['Id', 'X', 'Y', 'Z', 'dye_name']]
    else:
        data = data.loc[saved_lines, ['Id', 'X', 'Y', 'Z']]
    return data

def filter_excel_data(data : pd.DataFrame):
    if 'cell_id' not in data.columns or 'soma_x' not in data.columns:
        raise Warning("The format of input excel file is wrong.")
        return data
    result = pd.DataFrame(columns=['Id','X','Y','Z'])
    for i in data.index:
        line = data.loc[i]
        cid = 'C'+str(line['cell_id'])
        pid = line['patient_number']
        tid = line['tissue_block_number']
        rid = line['small_number']
        sid = line['slice_number']
        idList = []
        for id_ in (pid,tid,rid,sid,cid):
            if not is_empty(id_): idList.append(id_)
        ids = '-'.join(idList)
        x = line['soma_x']
        y = line['soma_y']
        z = line['soma_z']
        if not is_empty(x):
            result = result._append({'Id':ids,'X':x,'Y':y,'Z':z},ignore_index=True)
    return result

def read_csv(file_path):
    data = read_csv_with_detected_encoding(file_path)
    data = filter_csv_data(data)
    return data

def read_excel(file_path):
    data = pd.read_excel(file_path)
    data = filter_excel_data(data)
    return data

def read_mat(file_path):
    mat = loadmat(file_path)
    data = pd.DataFrame(columns=['Id','X','Y','Z'])
    for key in mat:
        if key[0] == '_' or key[-1] == '_':
            continue
        matkey = mat[key]
        if isinstance(matkey,np.ndarray):
            for i in range(len(matkey)):
                data = data._append({'Id':i,'X':matkey[i][0],'Y':matkey[i][1],'Z':matkey[i][2]},ignore_index=True)
            data.reset_index(drop=True)
        else:
            continue
    return data


def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = pcd.points
    df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    df.insert(0, 'Id', range(len(df)))
    return df

def read_table(file_path):
    file_name = os.path.basename(file_path)
    file,extension = os.path.splitext(file_name)
    try:
        if extension == '.csv':
            data = read_csv(file_path)
        elif extension == '.xlsx':
            data = read_excel(file_path)
        elif extension == '.marker':
            data = read_marker(file_path)
            data = filter_marker(data)
        elif extension == '.mat':
            data = read_mat(file_path)
        elif extension == '.ply':
            data = read_ply(file_path)
        elif extension == '.apo':
            data = read_apo(file_path)
            data = filter_apo(data)
        else:
            raise Exception("Input unsupported table file.")
        data = data.reset_index(drop=True)
        return data
    except Exception as e:
        print(e)
        return pd.DataFrame(columns=['Id','X','Y','Z'])


if __name__ == '__main__':
    saved_status, dropped_intensities = load_read_config()
    window = MatchWindow()