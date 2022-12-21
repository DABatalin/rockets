import matplotlib.pyplot as plt
from matplotlib.widgets import Slider   
from scipy.integrate import solve_ivp
import numpy as np
from math import pi

# Общие константы
R_Earth = 6371000 # Радиус Земли, м
G_M_Earth = 6.674*5.9722*10**13 # Гравитационная постоянная * масса Земли
geostationary_orbit_h = 35786000 # Высота геостационарной орбиты над экватором, м
M = 0.029 # Молярная масса воздуха, кг/моль
R = 8.314 # Универсальная газовая постоянная

# Погодные условия на уровне моря
T0 = 19 # Температура , град Цельсия
P0 = 760 # Давление, мм ртутного столба

# Общие параметры ракеты
# Зависимость коэффициента лобового аэродинамического сопротивления от числа Маха
Cx = [[0, 0.165], [0.5, 0.149], [0.7, 0.175], [0.9, 0.255], [1, 0.304], [1.1, 0.36], [1.3, 0.484], [1.5, 0.5], [2, 0.51], [2.5, 0.502], [3, 0.5], [3.5, 0.485], [4, 0.463], [4.5, 0.458], [5, 0.447]]
S = 172 # Площадь наибольшего поперечного сечения (миделево сечения), м^2

# Время работы каждого этапа полета, c
T1 = 123 # I ступень
T2 = 218 # II ступень
T3 = 242 # III ступень
T4 = 500 # Автономный полет
T5 = 270 # 1-ый запуск РБ "Бриз-М"
T6 = 3000 # Автономный полет
T7 = 1080 # 2-ый запуск РБ "Бриз-М"
T8 = 20000 # Автономный полет
T9 = 1200 # 3-ый запуск РБ "Бриз-М"
T10 = 84000 # Автономный полет

# Массы ступеней вместе с топливом, кг
M1 = 458.9*1000 # I ступень
M2 = 168.3*1000 # II ступень
M3 = 46.562*1000 # III ступень
M4_1 = 6.565*1000 # РБ "Бриз-М" (стартовая)
M4_2 = 5.871*1000 # РБ "Бриз-М" (2-ой этап)
M4_3 = 3.095*1000 # РБ "Бриз-М" (3-ий этап)
M5 = 2210 # Спутник "Экспресс-80"

def DrawEarth(ax):
    x = np.concatenate((np.arange(-R_Earth, R_Earth, 1000), np.array([R_Earth])))
    y1 = np.array([np.sqrt(R_Earth**2-int(x_i)**2) for x_i in x])
    y2 = np.array([-np.sqrt(R_Earth**2-int(x_i)**2) for x_i in x])

    ax.plot(x, y1, linewidth=2, color='green')
    ax.plot(x, y2, linewidth=2, color='green')

def DrawGeostationaryOrbit(ax):
    x = np.concatenate((np.arange(-(R_Earth+geostationary_orbit_h), R_Earth+geostationary_orbit_h, 1000), np.array([R_Earth+geostationary_orbit_h])))
    y1 = np.array([np.sqrt((R_Earth+geostationary_orbit_h)**2-int(x_i)**2) for x_i in x])
    y2 = np.array([-np.sqrt((R_Earth+geostationary_orbit_h)**2-int(x_i)**2) for x_i in x])

    ax.plot(x, y1, linewidth=2, color='grey', linestyle='--')
    ax.plot(x, y2, linewidth=2, color='grey', linestyle='--')

# Метод для получения значения из таблицы коэффициентов лобового аэродинамического сопротивления
def GetCx(M):
    for i in range(len(Cx)-1):
        if M == Cx[i][0]:
            return Cx[i][1]
        elif M > Cx[i][0] and M < Cx[i+1][0]:
            return (Cx[i][1]+Cx[i+1][1])/2
    return Cx[len(Cx)-1][1]

# Зависимость температуры от высоты (T в град Цельсия)
def Temperature(h, T0):
    temp = h*(-0.0065) + T0
    if temp < 4-273.15:
        temp = 4-273.15
    return temp

# Зависимость давления от высоты
def Pressure(h, P0):
    return (P0*133.32)*np.exp(-(M*9.81*h)/(R*(Temperature(h, T0)+273.15)))

# Зависимость плотности воздуха от высоты
def Density(h):
    if h >= 50000:
        return 0
    T = Temperature(h, T0)+273.15
    P = Pressure(h, P0)
    return (P*M)/(R*T)

# Зависимость скорости звука от высоты (T в Кельвинах)
def SoundSpeed(T):
    if T < 150:
        return 250
    return np.sqrt(1.4*R*T/M)

# Сила сопротивления воздуха
def ResistanceForce(r, phi, r_dot, phi_dot):
    return GetCx(((r_dot**2+(r*phi_dot)**2)**0.5)/(SoundSpeed(Temperature(r-R_Earth, T0)+273.15)))*Density(r-R_Earth)*(r_dot**2+(r_dot*phi_dot)**2)*S/2

'''
T - время работы ступени, с
F - суммарная тяга двигателей на старте, Н
sigma - изменение тяги двигателя за 1 секунду
M - масса всей ракеты на данном этапе
k - расход топлива за 1 секунду
beta_start - начальный угол поворота ракеты, град
beta_end - Конечный угол поворота ракеты, град
'''

# Разгонный этап с учетом атмосферы
def UpperStageAtmosphere(initial_conditions, T, F, sigma, M, k, beta_start, beta_end):
    beta_incr = (beta_end-beta_start) / T # Изменение угла в секунду
    # Переводим в радианы
    beta_start *= pi/180
    beta_incr *= pi/180

    def f(t, y):
        y1, y2, y3, y4 = y
        return [
            y2,
            y1*(y4**2)-G_M_Earth/(y1**2)+(np.cos(beta_start+beta_incr*t)/(M-k*t))*((F+sigma*t)-ResistanceForce(y1, y3, y2 ,y4)),
            y4,
            (4000000*np.sin(beta_start+beta_incr*t)*((F+sigma*t)-ResistanceForce(y1, y3, y2 ,y4))/(M-k*t)-2*y2*y1*y4)/(y1**2)
            ] 

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(f, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution

# Разгонный этап без учета атмосферы
def UpperStage(initial_conditions, T, F, M, k, beta_start, beta_end):
    beta_incr = (beta_end-beta_start) / T # Изменение угла в секунду
    # Переводим в радианы
    beta_start *= pi/180
    beta_incr *= pi/180 
         
    def f(t, y):
         y1, y2, y3, y4 = y
         return [
                y2,
                y1*(y4**2)-G_M_Earth/(y1**2)+(np.cos(beta_start+beta_incr*t)/(M-k*t))*F,
                y4,
                (4000000*np.sin(beta_start+beta_incr*t)*F/(M-k*t)-2*y2*y1*y4)/(y1**2)
                ] 

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(f, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution

# Автономный полет (двигатели выключены)
def AutonomousFlight(initial_conditions, T):
    def f(t, y):
         y1, y2, y3, y4 = y
         return [
                y2,
                y1*(y4**2)-G_M_Earth/(y1**2),
                y4,
                -2*((y4*y2)/y1)
                ] 

    t = np.array([i for i in range(0, T, 1)])
    solver = solve_ivp(f, [0, T], initial_conditions, method='RK45', dense_output=True)
    num_solution = solver.sol(t)
    return num_solution

# Подсчет траектории на всех этапах
def CalculateAllStages(start_pos):
    path = []
    path.append(UpperStageAtmosphere(start_pos, T1, 10026*1000, 7983.5, M1+M2+M3+M4_1+M5, 3622, 0, 60))
    path.append(UpperStageAtmosphere(path[-1][:,-1], T2, 2400*1000, 0, M2+M3+M4_1+M5, 731.63, 60, 60))
    path.append(UpperStage(path[-1][:,-1], T3, 583*1000, M3+M4_1+M5, 180, 60, 60))
    path.append(AutonomousFlight(path[-1][:,-1], T4))
    path.append(UpperStage(path[-1][:,-1], T5, 150*1000, M4_1+M5, 2.57, 60, 80))
    path.append(AutonomousFlight(path[-1][:,-1], T6))
    path.append(UpperStage(path[-1][:,-1], T7, 32.2*1000, M4_2+M5, 2.57, 90, 90))
    path.append(AutonomousFlight(path[-1][:,-1], T8))
    path.append(UpperStage(path[-1][:,-1], T9, 39.7*1000, M4_3+M5, 2.57, 89.6, 89.6))
    path.append(AutonomousFlight(path[-1][:,-1], T10))
    return path

# Траектрия пути
def ShowPath(ax, stages):
    line, = ax.plot(stages[0]*np.cos(stages[2]), stages[0]*np.sin(stages[2]), linewidth=1, color='blue')
    return line


def ConcatenateStages(path):
    t = np.array([i for i in range(0, T1+T2+T3+T4+T5+T6+T7+T8+T9+T10, 1)])
    r = np.concatenate([stage[0,:] for stage in path])
    r_dot = np.concatenate([stage[1,:] for stage in path])
    phi = np.concatenate([stage[2,:] for stage in path])
    phi_dot = np.concatenate([stage[3,:] for stage in path])
    return t, np.array([r, r_dot, phi, phi_dot])

# Получение точек, где начинаются новые этапы
def GetPoints(path, t):
    if t == 0:
        return np.array([])
    T = np.array([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10])
    points = []
    for i in range(len(path)-1):
        if np.sum(T[:i]) < t:
            points.append([path[i][0][-1], path[i][2][-1]])
    return np.array(points)

# График высоты
def ShowHeight(ax, path):
    t, stages = ConcatenateStages(path)
    a = np.min(path[-1][0,:])
    b = np.max(path[-1][0,:])
    print("Апогей геостационарной орбиты:", b, "м")
    print("Перигей геостационарной орбиты:", a, "м")
    print("Эксцентриситет геостационарной орбиты:", (1-(a**2)/(b**2))**0.5)
    h = stages[0, :]-R_Earth
    ax.plot(t[:28000], h[:28000])
    ax.grid()

# График скорости
def ShowSpeed(ax, path):
    t, stages = ConcatenateStages(path)
    v = np.sqrt(stages[1, :]**2 + (stages[0, :]*stages[3, :])**2)
    print("Скорость на геостационарной орбите:", v[-1], "м/с")
    ax.plot(t[:28000], v[:28000])
    ax.grid()

# График угловой скорости
def ShowAngularVelocity(ax, path):
    t, stages = ConcatenateStages(path)
    print("Угловая скорость на геостационарной орбите:", stages[3][-1], "рад/с")
    ax.plot(t[:28000], stages[3,:][:28000])
    ax.grid()


def main():
    fig, ax = plt.subplots(figsize=(8.2, 7))
    fig.subplots_adjust(left=0.25)
    axtime= fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    time_slider = Slider(
        ax=axtime,
        label='время, с',
        valmin=1,
        valmax=T1+T2+T3+T4+T5+T6+T7+T8+T9+T10,
        valinit=T1+T2+T3+T4+T5+T6+T7+T8+T9+T10,
        valstep=1,
        orientation="vertical",
        color="blue",
    )

    DrawEarth(ax)
    DrawGeostationaryOrbit(ax)
    path = CalculateAllStages([R_Earth, 0, 0, 0])
    _, stages = ConcatenateStages(path)
    line, = ax.plot(stages[0]*np.cos(stages[2]), stages[0]*np.sin(stages[2]), linewidth=1, color='blue')
    points = GetPoints(path, T1+T2+T3+T4+T5+T6+T7+T8+T9+T10)
    sc = ax.scatter(points[:, 0]*np.cos(points[:, 1]), points[:, 0]*np.sin(points[:, 1]), color='blue', s=8, alpha=0.7)

    def update_time(val):
        line.set_xdata(stages[0][:time_slider.val]*np.cos(stages[2][:time_slider.val]))
        line.set_ydata(stages[0][:time_slider.val]*np.sin(stages[2][:time_slider.val]))
        points = GetPoints(path, time_slider.val)
        sc.set_array(points[:, 0]*np.cos(points[:, 1]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)
    plt.axis('off')
    plt.show()

    fig, axs = plt.subplots(nrows=3, figsize = (8, 8))
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel('h, м')
    axs[1].set_xlabel('t, с')
    axs[1].set_ylabel('V, м/с')
    axs[2].set_xlabel('t, с')
    axs[2].set_ylabel('omega, рад/c')

    ShowHeight(axs[0], path)
    ShowSpeed(axs[1], path)
    ShowAngularVelocity(axs[2], path)
    plt.show()


if __name__ == "__main__":
    main()