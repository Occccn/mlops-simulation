# ///////////////////////////////////////////
# ///////// 2次元熱拡散方程式を解くツール ///////
# //////////////////////////////////////////

# Ref
# URL:https://watlab-blog.com/2022/10/08/2d-diffusion/

# --- Import Libraries ---
## Standard
import argparse
import os
import glob
import math
import shutil
## Third Party
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import yaml

# --- Define Function ---
class Fusecalc:
    def __init__(self, config):
        self.job_name = config["JOB_NAME"]
        os.makedirs(os.path.join("./", self.job_name), exist_ok=True)
        self.is_save = config["CALC_OPTION"]["is_save"]

        # 計算パラメータ
        self.dt          = config["CALC_OPTION"]["dt"]
        self.calctime    = config["CALC_OPTION"]["calctime"]
        self.dx          = config["CALC_OPTION"]["dx"]
        self.dy          = config["CALC_OPTION"]["dy"]
        self.x_max       = config["CALC_OPTION"]["x_max"]
        self.y_max       = config["CALC_OPTION"]["y_max"]
        self.alpha       = config["CALC_OPTION"]["alpha"]
        self.sigma11     = config["CALC_OPTION"]["sigma11"]
        self.sigma22     = config["CALC_OPTION"]["sigma22"]
        self.sigma12     = config["CALC_OPTION"]["sigma12"]
        self.sigma21     = config["CALC_OPTION"]["sigma21"]
        self.mean_1      = config["CALC_OPTION"]["mean_1"]
        self.mean_2      = config["CALC_OPTION"]["mean_2"]
        # 計算結果関連
        self.result           = None
        self.position_x_list  = None
        self.position_y_list  = None
        self.time_list        = None
        self.temperature_list = None
        self.random_seed      = config["CALC_OPTION"]["random_seed"]

    def gaussian_2d(self,sigma11, sigma12, sigma21, sigma22, mean_1, mean_2):
        sigma = np.array([[sigma11, sigma12],
                        [sigma21, sigma22]])
        mu = np.array([mean_1, mean_2])
        
        return sigma, mu

    def initial_field(self, x_max, y_max, dx, dy):
        ''' 初期場を用意する '''

        # 初期場(x方向をj, y方向をkとする行列を作成→2D画像のデータ構造を模擬)
        x = np.linspace(0, x_max, int(x_max / dx))
        y = np.linspace(0, y_max, int(y_max / dy))
        z = np.zeros((len(y), len(x)))

        # 2D Gaussian(sigma:分散共分散行列, mu:平均ベクトル)
        sigma, mu = self.gaussian_2d(self.sigma11, self.sigma12, self.sigma21, self.sigma22, self.mean_1, self.mean_2)
        det_sigma = np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)

        # 式のj,kの順番と同じにするため転置させて計算する
        z = z.T
        for j in range(len(z)):
            for k in range(len(z.T)):
                X = np.array([x[j], y[k]])
                z[j, k] = (1 / np.sqrt(2 * np.pi * det_sigma)) * np.exp((-1 / 2) * (X - mu).T @ inv_sigma @ (X - mu))
        z = z.T

        return x, y, z

    def boundary_condition(self, z):
        ''' 分布に境界条件を設定する '''

        # 境界条件(左右上下)
        z[:, 0] = 0
        z[:, -1] = 0
        z[0, :] = 0
        z[-1, :] = 0

        return z

    def sol_2d_diffusion(self, x, y, q, dt, dx, dy, a, calctime):
        ''' 2次元拡散方程式を計算する '''
        step = math.ceil(calctime / dt)

        # 漸化式を反復計算
        q = q.T
        for i in range(step):
            q0 = q.copy()
            for j in range(1, len(q) - 1):
                for k in range(1, len(q.T) - 1):
                    r = a * (dt / dx ** 2)
                    s = a * (dt / dy ** 2)
                    q[j, k] = q0[j, k] + r * (q0[j+1, k] - 2 * q0[j, k] + q0[j-1, k]) + \
                            s * (q0[j, k+1] - 2 * q0[j, k] + q0[j, k-1])
            # 境界条件を設定
            q = q.T
            q = self.boundary_condition(q)
            q = q.T

            # # 指定した間隔で画像保存
            # if i % result_interval == 0:
            #     print('Iteration=', i)
            #     q = q.T
            #     plot(x, y, q, i, dir, 1)
            #     q = q.T

        return q

    def sol_2d_diffusion_and_get_onepos_temp(self, x, y, q, dt, dx, dy, a, calctime, index_x, index_y):
        ''' 2次元拡散方程式を計算する '''
        step = math.ceil(calctime / dt)

        # 漸化式を反復計算
        q = q.T
        self.position_x_list  = [index_x*self.dx for _ in range(step)]
        self.position_y_list  = [index_y*self.dy for _ in range(step)]
        self.time_list        = []
        self.temperature_list = []
        for i in range(step):
            q0 = q.copy()
            for j in range(1, len(q) - 1):
                for k in range(1, len(q.T) - 1):
                    r = a * (dt / dx ** 2)
                    s = a * (dt / dy ** 2)
                    q[j, k] = q0[j, k] + r * (q0[j+1, k] - 2 * q0[j, k] + q0[j-1, k]) + \
                            s * (q0[j, k+1] - 2 * q0[j, k] + q0[j, k-1])
            # 結果の取得
            time        = step * dt
            temperature = q[index_x, index_y]
            self.time_list.append(time)
            self.temperature_list.append(temperature) 
            # 境界条件を設定
            q = q.T
            q = self.boundary_condition(q)
            q = q.T

            # # 指定した間隔で画像保存
            # if i % result_interval == 0:
            #     print('Iteration=', i)
            #     q = q.T
            #     plot(x, y, q, i, dir, 1)
            #     q = q.T
        return q


    def plot(self, x, y, z, index_x, index_y):
        ''' 関数をプロットする '''

        # フォントの種類とサイズを設定する。
        plt.rcParams['font.size'] = 14
        plt.rcParams['font.family'] = 'Times New Roman'

        # 目盛を内側にする。
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        # グラフの入れ物を用意して上下左右に目盛線を付ける。
        x_size = 8
        y_size = int(0.8 * x_size * (np.max(y) / np.max(x)))
        fig = plt.figure(figsize=(x_size, y_size))
        ax1 = fig.add_subplot(111)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')


        # 軸のラベルを設定する。
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # データをプロットする。
        im = ax1.imshow(z,
                        vmin=0, vmax=1,
                        extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                        aspect='auto',
                        cmap='jet')
        ax1.scatter(index_x*self.dx, index_y*self.dy, color="w")
        # カラーバーを設定する。
        cbar = fig.colorbar(im)
        cbar.set_label('q')

        # 画像を保存
        save_filepath = os.path.join("./", self.job_name, "result.png")
        plt.savefig(save_filepath)
        
    def get_index(self):
        if self.random_seed < 0:
            # 乱数の設定をしないときは、中心の値をとってくる
            index_x = int(self.mean_1 / self.dx)
            index_y = int(self.mean_2 / self.dy)
        else:
            # 乱数の設定をするときは、中心近傍の値をランダムにとってくる(分散の1/2)
            x = np.random.normal(self.mean_1, self.sigma11 / 2, 1)[0]
            y = np.random.normal(self.mean_2, self.sigma22 / 2, 1)[0]
            index_x = int(x / self.dx)
            index_y = int(y / self.dy)
        
        return index_x, index_y
    
    def run_get_laststep_value(self):
        # 初期場の用意
        x, y, q = self.initial_field(self.x_max, self.y_max, self.dx, self.dy)
        # 境界条件の設定
        q = self.boundary_condition(q)
        # 計算
        self.result = self.sol_2d_diffusion(x, y, q, self.dt, self.dx, self.dy, self.alpha, self.calctime)
        # 結果の取得
        index_x, index_y = self.get_index()
        self.result_value = self.result[index_x, index_y]
        if self.is_save:
            self.plot(x, y, self.result.T, index_x, index_y)
    
    def run(self):
        # 初期場の用意
        x, y, q = self.initial_field(self.x_max, self.y_max, self.dx, self.dy)
        # 境界条件の設定
        q = self.boundary_condition(q)
        # 計算
        index_x, index_y = self.get_index()
        result = self.sol_2d_diffusion_and_get_onepos_temp(x, y, q, self.dt, self.dx, self.dy, self.alpha, self.calctime, index_x, index_y)
        if self.is_save:
            self.plot(x, y, result.T, index_x, index_y)
    
    
    def get_result_value(self):
        return self.result_value

    def get_result(self):
        # 位置、時間、温度のリストを返すメソッド
        return self.time_list, self.position_x_list, self.position_y_list, self.temperature_list

if __name__ == '__main__':
    ''' 条件設定を行いシミュレーションを実行、流れのGIF画像を作成する '''
    # 初期化
    parser = argparse.ArgumentParser(
        usage="main.py job_config.yml",
        description="2次元の熱拡散方程式を求めるプログラム"
    )
    parser.add_argument("job_config", help="計算の設定ファイル")
    args = parser.parse_args()
    with open(args.job_config, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 計算実行
    fusecalc = Fusecalc(config)
    fusecalc.run()
    # ポスト処理
    time_list, position_x_list, position_y_list, temperature_list = fusecalc.get_result()
    shutil.copy2(args.job_config, os.path.join("./", config["JOB_NAME"]))
    with open(os.path.join("./", config["JOB_NAME"], "result.txt"), mode="w") as f:
        f.write("time, position_x, position_y, temperature\n")
        for t in range(len(time_list)):
            position_x  = position_x_list[t]
            position_y  = position_y_list[t]
            time        = time_list[t]
            temperature = temperature_list[t]
            f.write(f"{time}, {position_x}, {position_y}, {temperature}\n")
