import matplotlib.pyplot as plt
import matplotlib as cm
import math

from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import csv
import sommelier_p as sm
import pandas as pd



# === PLOT FUNCITONS  ===============================

def plot_scatter_matrix(wine_data, save_plot=False):
    headers = list(wine_data.columns.values)
    fig, axs = plt.subplots(len(headers) - 2, len(headers) - 2, figsize=(15, 15))
    k = 0
    for i in headers[0:-2]:
      l = 0
      for j in headers[0:-2]:
        if i == j:
          axs[k, l].text(0.5, 0.5, j.replace(' ', '\n'), va='center', ha='center')
        else:
          axs[k, l].scatter(wine_data[j], wine_data[i], s=2, c=wine_data['GoodBad'], cmap='PiYG')
        axs[k, l].tick_params(bottom=0, labelbottom=0, labelleft=0, left=0)
        l += 1
      k += 1
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, wspace=0, hspace=0)
    #---auto-download only works in Chrome
    if save_plot:
        plt.savefig("Wines_Plot.png")
        #files.download("Wines_Plot.png")
    else:
        plt.show()

def plot_scatter_trained(neuron, wd_x, wd_y):
    plt.figure(figsize=(15, 5))

    #---plot errors
    plt.subplot(1, 2, 1)
    plt.plot([neuron.log[i][0] for i in range(len(neuron.log))], \
             [neuron.log[i][1] for i in range(len(neuron.log))])
    plt.xlabel('epoch')
    plt.ylabel('classification errors')
    plt.title('Errors as a function of epoch')

    #---scatter plot
    plt.subplot(1, 2, 2)
    scat_1 = ('green', 'good wines (>' + str(gt) + ' score)', 1)
    scat_2 = ('magenta', 'bad wines (<' + str(bt) + ' score)', -1)
    for color in (scat_1, scat_2):
        plt.scatter([wd_x[i][1] for i in range(len(wd_y)) if wd_y[i] == color[2]], \
                    [wd_x[i][0] for i in range(len(wd_y)) if wd_y[i] == color[2]], \
                    s=8, c=color[0], label=color[1])

    #---getting window coordinates for clipping
    xmin, xmax, ymin, ymax = plt.axis()

    #---plotting decision boundary. calculating k,b in y = kx + b and drawing the
    #---line in clipped window + filling sides with colors.
    x1 = (-1 * neuron.b) / neuron.w[1] # y1 = 0
    y2 = (-1 * neuron.b) / neuron.w[0] # x2 = 0
    k = (y2 - 0)/(0 - x1)
    plt.plot([xmin, xmax], [k * xmin + y2, k * xmax + y2], \
            linestyle='--', color='black', linewidth=1, label='Decision Boundary')
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymax, color='magenta', alpha=.2)
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymin, color='Green', alpha=.2)

    #---plot characteristics:
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('alchogol')
    plt.ylabel('pH')
    plt.title('Decision Boundary on epoch: ' + str(neuron.log[-1][0]))
    plt.legend(loc='best', framealpha=0.4, prop={'size': 6})
    plt.show()

def plot_scatter_validation(neuron, wd_x, wd_y, wd_x_a, wd_y_a, errors):
    plt.figure(figsize=(15, 5))
    #---scatter plot training
    plt.subplot(1, 2, 1)
    scat_1 = ('green', 'good wines (>' + str(gt) + ' score)', 1)
    scat_2 = ('magenta', 'bad wines (<' + str(bt) + ' score)', -1)
    for color in (scat_1, scat_2):
        plt.scatter([wd_x[i][1] for i in range(len(wd_y)) if wd_y[i] == color[2]], \
                    [wd_x[i][0] for i in range(len(wd_y)) if wd_y[i] == color[2]], \
                    s=8, c=color[0], label=color[1])
    #---getting window coordinates for clipping
    xmin, xmax, ymin, ymax = plt.axis()
    #---plotting decision boundary. calculating k,b in y = kx + b and drawing the
    #---line in clipped window + filling sides with colors.
    x1 = (-1 * neuron.b) / neuron.w[1] # y1 = 0
    y2 = (-1 * neuron.b) / neuron.w[0] # x2 = 0
    k = (y2 - 0)/(0 - x1)
    plt.plot([xmin, xmax], [k * xmin + y2, k * xmax + y2], \
            linestyle='--', color='black', linewidth=1, label='Decision Boundary')
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymax, color='magenta', alpha=.2)
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymin, color='Green', alpha=.2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('alchogol')
    plt.ylabel('pH')
    plt.title('Decision Boundary on epoch: ' + str(neuron.log[-1][0]))
    plt.legend(loc='best', framealpha=0.4, prop={'size': 6})

    #---scatter plot validation:
    plt.subplot(1, 2, 2)
    for color in (scat_1, scat_2):
        plt.scatter([wd_x_a[i][1] for i in range(len(wd_y_a)) if wd_y_a[i] == color[2]], \
                    [wd_x_a[i][0] for i in range(len(wd_y_a)) if wd_y_a[i] == color[2]], \
                    s=8, c=color[0], label=color[1], marker='x')
    #---plotting decision boundary. calculating k,b in y = kx + b and drawing the
    #---line in clipped window + filling sides with colors.
    plt.plot([xmin, xmax], [k * xmin + y2, k * xmax + y2], \
            linestyle='--', color='black', linewidth=1, label='Decision Boundary')
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymax, color='magenta', alpha=.2)
    plt.fill_between([xmin, xmax], [k * xmin + y2, k * xmax + y2], ymin, color='Green', alpha=.2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel('alchogol')
    plt.ylabel('pH')
    plt.title('Decision boundary validation\n Errors: ' + str(errors))
    plt.legend(loc='best', framealpha=0.4, prop={'size': 6})
    plt.show()

def plot_3d_scatter(data):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = [1 - abs(x[0] / x[1]) for x in pan_gal.values]
    #print(x, y, z, sep="\n")

    ax.scatter(x, y, z, s=8, c=data['GoodBad'])
    plt.show()

def plot_3d(neuron):
    x = [neuron.log[i][2][0] for i in range(len(neuron.log))]
    y = [neuron.log[i][2][1] for i in range(len(neuron.log))]
    z = [neuron.log[i][1] for i in range(len(neuron.log))]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    urf = ax.plot(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False)
    #urf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=False)
    #ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
    plt.show()

# === DATA FUNCITONS  ===============================

def happy_panda(df, bt, gt, trim_by='quality', desired_columns=0):
    '''
    Parameters
    -----------
    params : list
      List of column names to be returned.
      (shoud be in your data)
    gt, bt : int
      Good and bad tresholds to trim (>= gt and <= bt)
    sep : string
      Separator of your data
      (default value is ';')
    trim_by : string
      Name of the column by which your data will be trimmed
    '''
    treshold_data = df[~df[trim_by].isin(range(bt, gt + 1))]
    bools = pd.Series(treshold_data[trim_by].apply(lambda x: 1 if x >= gt else -1), name='GoodBad')
    data = treshold_data if not desired_columns else treshold_data[desired_columns]
    concatenated = pd.concat([data, bools], axis=1)
    return(concatenated)

def norm_data(wd):
    '''
    this function allows to normolize different NxM data matrixes
    x' = (x - avg)/(xmax - xmin)
    '''
    wd_sc = wd.copy()
    ranges = []
    means = []
    for i in range(len(wd.iloc[0])):
        ranges.append(max(wd.iloc[:, i] - min(wd.iloc[:, i])))
        means.append(sum(wd.iloc[:, i]) / len(wd.iloc[:, i]))
    for i in range(len(wd.iloc[0])):
        for j in range(len(wd)):
            wd_sc.iloc[j][i] = (wd.iloc[j][i] - means[i]) / ranges[i]
    return wd_sc

def validation_holdout(percent, data, shuffle=False):
    if shuffle:
        data.sample(frac=1)
    boundary = int(len(df) * percent)
    return (data.iloc[0:boundary], data.iloc[boundary:-1])

def proportion(percent, data, shuffle=False):
    if shuffle:
        data = data.sample(frac=1)
    boundary = int(len(data) * percent)
    return (data.iloc[boundary:-1], data.iloc[0:boundary])

def k_fold(num_folds, data):
    step = int(len(data) / num_folds)
    folds = ()
    result = []
    for x, y in zip(range(0, len(data), step), range(step, len(data) + step, step)):
        folds += (tuple([data.iloc[x:y]]))
    for i in range(num_folds):
        result.append((pd.concat([folds[x] for x in range(len(folds)) if not x == i], axis = 0), folds[i]))
    return result


# === MAIN ====================================

#---getting a data file
url = "https://raw.githubusercontent.com/katolikyan/sommelier/master/resources/winequality-red.csv"
try:
    df = pd.read_csv(url, ';')
except FileNotFoundError:
    print("[-] Link doesn't contain .csv file")

gt = 7 #good trashold
bt = 4 #bad trashold
#---plotting the data matrix
wine_data = happy_panda(df, bt, gt)
plot_scatter_matrix(wine_data)

#---preparing new slice of data and training perceptron
gt = 7 #good trashold
bt = 4 #bad trashold
wine_data = happy_panda(df, bt, gt, desired_columns=['pH', 'alcohol', 'quality'])
neuron = sm.perceptron(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
neuron.learning_simple(wine_data.loc[:, 'pH':'alcohol'].values, \
                       wine_data.loc[:, 'GoodBad'].values, 20000, 0.3)

#---plotting the training results
plot_scatter_trained(neuron, wine_data.loc[:, 'pH':'alcohol'].values, \
                     wine_data.loc[:, 'GoodBad'].values)

#---reorganizing the data for better learning
wd_sc = norm_data(wine_data.iloc[:, 0:2])
#---train another neuron with normalized dataset and print it
neuron_2 = sm.perceptron(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
neuron_2.learning_simple(wd_sc.loc[:, 'pH':'alcohol'].values, \
                         wine_data.loc[:, 'GoodBad'].values, 300, 0.3)
plot_scatter_trained(neuron_2, wd_sc.loc[:, 'pH':'alcohol'].values, \
                     wine_data.loc[:, 'GoodBad'].values)

#---create and train adaline
adaline = sm.adaline(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
adaline.gdl(wd_sc.loc[:, 'pH':'alcohol'].values, \
            wine_data.loc[:, 'GoodBad'].values, 300, 0.001, 2)
plot_scatter_trained(adaline, wd_sc.loc[:, 'pH':'alcohol'].values, \
                     wine_data.loc[:, 'GoodBad'].values)

#---create and train adaline online
adaline = sm.adaline(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
adaline.gdl_online(wd_sc.loc[:, 'pH':'alcohol'].values, \
            wine_data.loc[:, 'GoodBad'].values, 300, 0.001, 2)
plot_scatter_trained(adaline, wd_sc.loc[:, 'pH':'alcohol'].values, \
                     wine_data.loc[:, 'GoodBad'].values)

#---attemt to print 3d:
#plot_3d(adaline)

#---finding an optimal learning rate:
for i in [0.1, 0.01, 0.001]:
    adaline.__init__(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
    adaline.gdl(wd_sc.loc[:, 'pH':'alcohol'].values, \
                wine_data.loc[:, 'GoodBad'].values, 300, i, 2)
    plot_scatter_trained(adaline, wd_sc.loc[:, 'pH':'alcohol'].values, \
                         wine_data.loc[:, 'GoodBad'].values)


wd_sc = pd.concat([wd_sc, wine_data.loc[:, 'GoodBad']], axis=1)
#---HoldOut data validation set
df_ho = proportion(0.3, wd_sc)
adaline.__init__(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
adaline.gdl(df_ho[0].loc[:, 'pH':'alcohol'].values, \
            df_ho[0].loc[:, 'GoodBad'].values, 300, 0.01, 2)
answ = adaline.predict(df_ho[1].loc[:, 'pH':'alcohol'].values)
predict_err = sum([True for i in range(len(answ)) \
                if df_ho[1].loc[:, 'GoodBad'].values[i] != answ[i]])
plot_scatter_validation(adaline, df_ho[0].loc[:, 'pH':'alcohol'].values, \
                        df_ho[0].loc[:, 'GoodBad'].values, \
                        df_ho[1].loc[:, 'pH':'alcohol'].values, \
                        df_ho[1].loc[:, 'GoodBad'].values, predict_err)
print("holdout: ", 1 - predict_err/len(answ))

#---k-fold data validation set
df_kf = k_fold(4, wd_sc)
for fold in df_kf:
    print(len(fold[0]), len(fold[1]))
    adaline.__init__(len(wine_data.loc[:, 'pH':'alcohol'].values[0] - 1))
    adaline.gdl(fold[0].loc[:, 'pH':'alcohol'].values, \
                fold[0].loc[:, 'GoodBad'].values, 300, 0.01, 2)
    answ = adaline.predict(fold[1].loc[:, 'pH':'alcohol'].values)
    predict_err = sum([True for i in range(len(answ)) \
                       if fold[1].loc[:, 'GoodBad'].values[i] != answ[i]])
    plot_scatter_validation(adaline, fold[0].loc[:, 'pH':'alcohol'].values, \
                         fold[0].loc[:, 'GoodBad'].values, \
                         fold[1].loc[:, 'pH':'alcohol'].values, \
                         fold[1].loc[:, 'GoodBad'].values, predict_err)
    print("k_folds: ", 1 - predict_err/len(answ))

#---calculating % of good unswers with more then 2 parameters, 3 params:
wine_data = happy_panda(df, bt, gt, desired_columns=\
                        ['pH', 'alcohol', 'volatile acidity', 'quality'])
wd_sc = pd.concat([norm_data(wine_data.iloc[:, 0:3]),\
                             wine_data.loc[:, 'GoodBad']], axis=1)
df_ho = proportion(0.3, wd_sc)

adaline.__init__(len(wd_sc.loc[:, 'pH':'volatile acidity'].values[0] - 1))
adaline.gdl(df_ho[0].loc[:, 'pH':'volatile acidity'].values, \
            df_ho[0].loc[:, 'GoodBad'].values, 1000, 0.01, 2)
answ = adaline.predict(df_ho[1].loc[:, 'pH':'volatile acidity'].values)
predict_err = sum([True for i in range(len(answ)) \
                if df_ho[1].loc[:, 'GoodBad'].values[i] != answ[i]])
print("Percent of good predictions: ", 1 - predict_err/len(answ))

#---calculating % of good unswers with more then 2 parameters, 4 params:
wine_data = happy_panda(df, bt, gt, desired_columns=\
            ['pH', 'alcohol', 'volatile acidity', 'fixed acidity', 'quality'])
wd_sc = pd.concat([norm_data(wine_data.iloc[:, 0:4]),\
                             wine_data.loc[:, 'GoodBad']], axis=1)
df_ho = proportion(0.3, wd_sc)

adaline.__init__(len(wd_sc.loc[:, 'pH':'fixed acidity'].values[0] - 1))
adaline.gdl(df_ho[0].loc[:, 'pH':'fixed acidity'].values, \
            df_ho[0].loc[:, 'GoodBad'].values, 1000, 0.01, 2)
answ = adaline.predict(df_ho[1].loc[:, 'pH':'fixed acidity'].values)
predict_err = sum([True for i in range(len(answ)) \
                if df_ho[1].loc[:, 'GoodBad'].values[i] != answ[i]])
print("Percent of good predictions: ", 1 - predict_err/len(answ))

# --- V6 ---------------------------

url = "https://raw.githubusercontent.com/katolikyan/sommelier/master/resources/Pan%20Galactic%20Gargle%20Blaster.csv"
try:
    df = pd.read_csv(url, ';')
except FileNotFoundError:
    print("[-] Link doesn't contain .csv file")

bt = 2
gt = 8
pan_gal = happy_panda(df, bt, gt)
pan_normed = norm_data(pan_gal.iloc[:, 0:2])
pan_final = pd.concat([pan_normed.iloc[:], pan_gal.iloc[:, 2:]], axis=1)
plot_scatter_matrix(pan_final, save_plot=False)
r = pd.Series([math.sqrt(x[0]*x[0] + x[1]*x[1]) for x in pan_final.values], name='r').to_frame()
phi = pd.Series([math.atan(x[1] / x[0]) for x in pan_final.values], name='phi').to_frame()
pan_final['wonderflonium'] = r.values
pan_final['fallian marsh gas'] = phi.values
plot_scatter_matrix(pan_final, save_plot=False)

#---training perceptron:
percep = sm.perceptron(2)
percep.learning_simple(pan_final.iloc[:, 0:2].values, \
                         pan_final.loc[:, 'GoodBad'].values, 300, 0.3)
plot_scatter_trained(percep, pan_final.iloc[:, 0:2].values, \
                     pan_final.loc[:, 'GoodBad'].values)


