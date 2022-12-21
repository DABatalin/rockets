import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider   
import numpy as np
from math import pi
from json import load
plt.rcParams['figure.figsize'] = [10, 10]

# Общие константы
R_Earth = 600000 # Радиус Кербина, м
G_M_Earth = 3.5316*10**12 # Гравитационная постоянная * масса Кербина
geostationary_orbit_h = 2863330 # Высота геостационарной орбиты над экватором, м
M = 0.029 # Молярная масса воздуха, кг/моль
R = 8.314 # Универсальная газовая постоянная

# Погодные условия на уровне моря
T0 = 15 # Температура , град Цельсия
P0 = 760 # Давление, мм ртутного столба

# Общие параметры ракеты
# Зависимость коэффициента лобового аэродинамического сопротивления от числа Маха
Cx = [[0, 0.165], [0.5, 0.149], [0.7, 0.175], [0.9, 0.255], [1, 0.304], [1.1, 0.36], [1.3, 0.484], [1.5, 0.5], [2, 0.51], [2.5, 0.502], [3, 0.5], [3.5, 0.485], [4, 0.463], [4.5, 0.458], [5, 0.447]]
S = 63.6 # Площадь наибольшего поперечного сечения (миделево сечения), м^2

# Время работы каждого этапа полета, c
T1 = 90 # I ступень
T2 = 55 # II ступень
T3 = 8 # III ступень (1 запуск)
T4 = 380 # Автономный полет
T5 = 50 # III ступень (2 запуск)
T6 = 4800 # Автономный полет
T7 = 80 # РБ "Бриз-М"
T8 = 21000 # Автономный полет

# Массы ступеней вместе с топливом, кг
M1 = 83.56*1000 # I ступень
M2 = 10.71*1000 # II ступень
M3 = 5.341*1000 # III ступень
M4 = 1.435*1000 # РБ "Бриз-М"
M5 = 1748 # Спутник

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
                (220000*np.sin(beta_start+beta_incr*t)*((F+sigma*t)-ResistanceForce(y1, y3, y2 ,y4))/(M-k*t)-2*y2*y1*y4)/(y1**2)
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
                (220000*np.sin(beta_start+beta_incr*t)*F/(M-k*t)-2*y2*y1*y4)/(y1**2)
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
    path.append(UpperStageAtmosphere(start_pos, T1, 3000*1000, 1550, M1+M2+M3+M4+M5, 582.2, 0, 70))
    path.append(UpperStage(path[-1][:,-1], T2, 430*1000, M2+M3+M4+M5, 30, 70, 80))
    path.append(UpperStage(path[-1][:,-1], T3, 255*1000, M3+M4+M5, 48.5, 90, 90))
    path.append(AutonomousFlight(path[-1][:,-1], T4))
    path.append(UpperStage(path[-1][:,-1], T5, 1025*1000, (M3-388)+M4+M5, 48.5, 90, 90))
    path.append(AutonomousFlight(path[-1][:,-1], T6))
    path.append(UpperStage(path[-1][:,-1], T7, 179*1000, M4+M5, 11, 91.37, 91.37))
    path.append(AutonomousFlight(path[-1][:,-1], T8))
    return path

# Траектрия пути
def ShowPath(ax, path):
    for stage in path:
        ax.plot(stage[0]*np.cos(stage[2]), stage[0]*np.sin(stage[2]), linewidth=1, color='blue')
        #ax.scatter([stage[0,-1]*np.cos(stage[2,-1]), stage[0,-1]*np.sin(stage[2,-1])], color='blue')

def ConcatenateStages(path):
    t = np.array([i for i in range(0, T1+T2+T3+T4+T5+T6+T7+T8, 1)])
    r = np.concatenate([stage[0,:] for stage in path])
    r_dot = np.concatenate([stage[1,:] for stage in path])
    phi = np.concatenate([stage[2,:] for stage in path])
    phi_dot = np.concatenate([stage[3,:] for stage in path])
    return t, np.array([r, r_dot, phi, phi_dot])

# График высоты
def ShowHeight(ax, path):
    t, stages = ConcatenateStages(path)
    a = np.min(path[-1][0,:])
    b = np.max(path[-1][0,:])
    print("Апогей геостационарной орбиты:", b)
    print("Перигей геостационарной орбиты:", a)
    print("Эксцентриситет геостационарной орбиты:", (1-(a**2)/(b**2))**0.5)
    h = stages[0, :]-R_Earth
    ax.plot(t[:10000], h[:10000])
    ax.grid()

# График скорости
def ShowSpeed(ax, path):
    t, stages = ConcatenateStages(path)
    v = np.sqrt(stages[1, :]**2 + (stages[0, :]*stages[3, :])**2)
    print("Скорость на геостационарной орбите по расчетам:", v[-1], "м/с")
    print("Скорость на геостационарной орбите в KSP: 1009 м/с")
    print(f"Погрешность составляет {v[-1] / (1009)}%")
    ax.plot(t[:25000], v[:25000])
    ax.grid()

# График скорости из ksp
def ShowSpeedAlternate(ax):
    with open("logs", encoding = "utf-8") as file:
        data = load(file)
        time = [int(i) for i in data.keys()] [:45]
        speed = list(data.values()) [:45]
    ax.plot(time, speed)
    ax.grid()

# График угловой скорости
def ShowAngularVelocity(ax, path):
    t, stages = ConcatenateStages(path)
    print("Угловая скорость на геостационарной орбите:", stages[3][-1])
    ax.plot(t[:25000], stages[3,:][:25000])
    ax.grid()

def GetPoints(path, t):
    if t == 0:
        return np.array([])
    T = np.array([T1, T2, T3, T4, T5, T6, T7, T8])
    points = []
    for i in range(len(path)-1):
        if np.sum(T[:i]) < t:
            points.append([path[i][0][-1], path[i][2][-1]])
    return np.array(points)

def main():
    fig, ax = plt.subplots(figsize=(8.2, 7))
    fig.subplots_adjust(left=0.25)
    axtime= fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    time_slider = Slider(
        ax=axtime,
        label='время, с',
        valmin=1,
        valmax=T1+T2+T3+T4+T5+T6+T7+T8,
        valinit=T1+T2+T3+T4+T5+T6+T7+T8,
        valstep=1,
        orientation="vertical",
        color="blue",
    )

    DrawEarth(ax)
    DrawGeostationaryOrbit(ax)
    path = CalculateAllStages([R_Earth, 0, 0, 0])
    _, stages = ConcatenateStages(path)
    line, = ax.plot(stages[0]*np.cos(stages[2]), stages[0]*np.sin(stages[2]), linewidth=1, color='blue')
    points = GetPoints(path, T1+T2+T3+T4+T5+T6+T7+T8)
    sc = ax.scatter(points[:, 0]*np.cos(points[:, 1]), points[:, 0]*np.sin(points[:, 1]), color='blue', s=8, alpha=0.7)

    def update_time(val):
        line.set_xdata(stages[0][:time_slider.val]*np.cos(stages[2][:time_slider.val]))
        line.set_ydata(stages[0][:time_slider.val]*np.sin(stages[2][:time_slider.val]))
        points = GetPoints(path, time_slider.val)
        sc.set_array(points[:, 0]*np.cos(points[:, 1]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)
    plt.axis('off')
    # plt.show()

    fig, axs = plt.subplots(nrows=2, figsize = (8, 8))
    fig.tight_layout(h_pad = 10)
    plt.subplots_adjust(left = 0.1, bottom = 0.1)
    axs[0].set_xlabel('t, с')
    axs[0].set_ylabel('V, м/с')
    axs[0].set_title("Расчетная скорость")
    axs[1].set_xlabel('t, с')
    axs[1].set_ylabel('V, м/с')
    axs[1].set_title("Реальная скорость из KSP")

    # ShowHeight(axs[0], path)
    ShowSpeed(axs[0], path)
    ShowSpeedAlternate(axs[1])
    # ShowAngularVelocity(axs[2], path)
    plt.show()


if __name__ == "__main__":
    main()