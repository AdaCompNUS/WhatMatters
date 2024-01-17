import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score

if __name__ == "__main__":

    ### Analyze the results that training and validating in Dense Summit Dataset ###
    # methods = ["ConstVelo", "ConstAccel", "KNN", "lstm" ,"s-lstm", "mapPrior-lstm", "LaneGCN", "HiVT"]
    # result, plot = dict(), dict()
    # result['FDE1'] = np.array([[6.693,7.145,6.505,5.006,5.136,6.037,4.215,3.525],
    #                  [6.053,6.174,6.721,3.948,3.954,5.966,3.02,3.547]])
    # result['ADE1'] = np.array([[3.147,3.402,3.1,2.372,2.469,3.4,1.954,1.692],
    #                  [2.715,2.777,2.99,1.728,1.741,2.66,1.367,1.609]])
    # result['MR1'] = np.array([[0.739,0.737,0.654,0.665,0.707,0.68,0.475,0.462],
    #                 [0.742,0.748,0.816,0.658,0.656,0.7,0.499,0.599]])
    # result['FDE6'] = np.array([[3.738,3.881,2.742,4.871,4.906,4.957,1.495,1.065],
    #                  [3.897,4.011,2.579,3.86,3.818,3.914,1.081,0.969]])
    # result['ADE6'] = np.array([[1.867,1.947,1.668,2.312,2.369,2.877,1.009,0.808],
    #                  [1.727,1.773,1.412,1.694,1.688,1.911,0.711,0.661]])
    # result['MR6'] = np.array([[0.443,0.443,0.395,0.653,0.676,0.658,0.233,0.164],
    #                 [0.559,0.555,0.428,0.643,0.634,0.627,0.103,0.092]])
    

    # fig = plt.figure(figsize=(60, 40))
    # plt.rcParams.update({'font.size': 22})
    # count = [1,3,5,2,4,6]
    # for i,key in enumerate(result):
    #     ax = plt.subplot(3, 2, int(count[i]))
    #     plt.xlabel("Summit")
    #     plt.ylabel("Argoverse")
    #     x,y = result[key][0,:].reshape(-1,1),result[key][1,:].reshape(-1,1)
    #     plt.scatter(x, y)
    #     for j in range(len(x)):
    #         bias = 0.05 if key[:2] != 'MR' else 0.01
    #         plt.annotate(methods[j], xy = (x[j], y[j]), xytext = (x[j]+bias, y[j]+bias))
    #     x = sm.add_constant(x)
    #     model = sm.OLS(y,x).fit()
    #     # print(model.summary())
    #     std=np.std(model.predict(x))
    #     std_z = 1.96 # from z-table for 95%
    #     confidence_interval = std * std_z
    #     plt.plot(x[:,1], model.predict(x), label="Linear Regression")
    #     plt.plot(x[:,1], model.predict(x) - confidence_interval,label="95%-")
    #     plt.plot(x[:,1], model.predict(x) + confidence_interval,label="95%+")
    #     [xmin,xmax] = ax.xaxis.get_view_interval()
    #     [ymin,ymax] = ax.yaxis.get_view_interval()
    #     # params (y=ax+b), R2 and p-value
    #     a,b = model.params[1],model.params[0]
    #     r2 = round(model.rsquared, 3)
    #     p_value = round(model.summary2().tables[1]['P>|t|']['x1'], 4)
    #     plt.text(0.95*xmin+0.05*xmax, 0.9*ymax+0.1*ymin, '$y$ = {:.2f} + {:.2f}$x$'.format(b,a))
    #     plt.text(0.95*xmin+0.05*xmax, 0.85*ymax+0.15*ymin, f'$R$$^{2}$ = {r2}')
    #     plt.text(0.95*xmin+0.05*xmax, 0.8*ymax+0.2*ymin, f'$p-value$ = {p_value}')
    #     plt.legend(loc=4)
    # plt.savefig('Argoverse VS Summit.png')

    ######  Plot bar pictures ----- Not very clear  #####
    # count = [1,3,5,2,4,6]
    # plt.figure(figsize=(60, 40))
    # plt.rcParams.update({'font.size': 22})
    # for i,key in enumerate(result):
    #     new_key1 = key + "_ratio"
    #     new_key2 = key + "_avg"
    #     plot[new_key1] = result[key][1,:]/result[key][0,:]
    #     plot[new_key2] = np.mean(plot[new_key1])
    #     plt.subplot(3, 2, int(count[i]))
    #     plt.ylabel("Ratio")
    #     plt.bar(methods,plot[new_key2], label = "Average Ratio", color='orange')
    #     plt.bar(methods,plot[new_key1]-plot[new_key2], bottom=plot[new_key2], label='Bias to The Average Ratio', color='blue')
    #     plt.legend()
    #     plt.title(key+ "(Argoverse}/" + key + "(Summit)", )

    # plt.savefig('Argoverse VS Summit.png')


    ## Analyze the results that training in Dense Summit Dataset but validating in Sparse Dataset ###
    # methods = ["ConstVelo", "ConstAccel", "lstm" ,"s-lstm", "mapPrior-lstm", "LaneGCN", "HiVT"]
    # result, plot = dict(), dict()
    # result['FDE1'] = np.array([[5.547,5.639,4.795,5.502,5.209,3.359,2.509],
    #                  [6.053,6.174,3.948,3.954,5.966,3.02,3.547]])
    # result['ADE1'] = np.array([[2.513,2.564,2.059,2.493,3.05,1.376,1.03],
    #                  [2.715,2.777,1.728,1.741,2.66,1.367,1.609]])
    # result['MR1'] = np.array([[0.457,0.464,0.647,0.721,0.587,0.35,0.328],
    #                 [0.742,0.748,0.658,0.656,0.7,0.499,0.599]])
    # result['FDE6'] = np.array([[3.738,3.881,4.871,4.906,4.957,1.495,1.065],
    #                  [3.897,4.011,3.86,3.818,3.914,1.081,0.969]])
    # result['ADE6'] = np.array([[1.867,1.947,2.312,2.369,2.877,1.009,0.808],
    #                  [1.727,1.773,1.694,1.688,1.911,0.711,0.661]])
    # result['MR6'] = np.array([[0.443,0.443,0.653,0.676,0.658,0.233,0.164],
    #                 [0.559,0.555,0.643,0.634,0.627,0.103,0.092]])
    

    # fig = plt.figure(figsize=(60, 40))
    # plt.rcParams.update({'font.size': 22})
    # count = [1,3,5,2,4,6]
    # for i,key in enumerate(result):
    #     ax = plt.subplot(3, 2, int(count[i]))
    #     plt.xlabel("Summit")
    #     plt.ylabel("Argoverse")
    #     x,y = result[key][0,:].reshape(-1,1),result[key][1,:].reshape(-1,1)
    #     plt.scatter(x, y)
    #     for j in range(len(x)):
    #         bias = 0.05 if key[:2] != 'MR' else 0.01
    #         plt.annotate(methods[j], xy = (x[j], y[j]), xytext = (x[j]+bias, y[j]+bias))
    #     x = sm.add_constant(x)
    #     model = sm.OLS(y,x).fit()
    #     std=np.std(model.predict(x))
    #     std_z = 1.96 # from z-table for 95%
    #     confidence_interval = std * std_z
    #     plt.plot(x[:,1], model.predict(x), label="Linear Regression")
    #     plt.plot(x[:,1], model.predict(x) - confidence_interval,label="95%-")
    #     plt.plot(x[:,1], model.predict(x) + confidence_interval,label="95%+")
    #     [xmin,xmax] = ax.xaxis.get_view_interval()
    #     [ymin,ymax] = ax.yaxis.get_view_interval()
    #     #params (y=ax+b), R2 and p-value
    #     a,b = model.params[1],model.params[0]
    #     r2 = round(model.rsquared, 3)
    #     p_value = round(model.summary2().tables[1]['P>|t|']['x1'], 4)
    #     plt.text(0.95*xmin+0.05*xmax, 0.9*ymax+0.1*ymin, '$y$ = {:.2f} + {:.2f}$x$'.format(b,a))
    #     plt.text(0.95*xmin+0.05*xmax, 0.85*ymax+0.15*ymin, f'$R$$^{2}$ = {r2}')
    #     plt.text(0.95*xmin+0.05*xmax, 0.8*ymax+0.2*ymin, f'$p-value$ = {p_value}')
    #     plt.legend(loc=4)
    # plt.savefig('Argoverse VS Summit Sparse.png')

    ## Analyze the results that training & validating in Sparse Dataset ###
    methods = ["ConstVelo", "ConstAccel", "KNN", "lstm" ,"s-lstm", "LaneGCN", "HiVT"]
    result, plot = dict(), dict()
    result['FDE1'] = np.array([[5.547,5.639,5.319,4.065,4.437,3.1497,2.15],
                     [6.053,6.174,6.721,3.948,3.954,3.02,3.547]])
    result['ADE1'] = np.array([[2.513,2.564,2.263,1.712,1.937,1.291,0.85],
                     [2.715,2.777,2.99,1.728,1.741,1.367,1.609]])
    result['MR1'] = np.array([[0.457,0.464,0.526,0.561,0.582,0.348,0.294],
                    [0.742,0.748,0.816,0.658,0.656,0.499,0.599]])
    result['FDE6'] = np.array([[3.860,3.884,2.239,3.879,4.215,1.270,0.734],
                     [3.897,4.011,2.579,3.86,3.818,1.081,0.969]])
    result['ADE6'] = np.array([[1.655,1.697,1.147,1.635,1.840,0.701,0.423],
                     [1.727,1.773,1.412,1.694,1.688,0.711,0.661]])
    result['MR6'] = np.array([[0.360,0.355,0.279,0.530,0.551,0.183,0.101],
                    [0.559,0.555,0.428,0.643,0.634,0.103,0.092]])
    

    fig = plt.figure(figsize=(60, 40))
    plt.rcParams.update({'font.size': 22})
    count = [1,3,5,2,4,6]
    for i,key in enumerate(result):
        ax = plt.subplot(3, 2, int(count[i]))
        plt.xlabel("Summit")
        plt.ylabel("Argoverse")
        x,y = result[key][0,:].reshape(-1,1),result[key][1,:].reshape(-1,1)
        plt.scatter(x, y)
        for j in range(len(x)):
            bias = 0.05 if key[:2] != 'MR' else 0.01
            plt.annotate(methods[j], xy = (x[j], y[j]), xytext = (x[j]+bias, y[j]+bias))
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit()
        std=np.std(model.predict(x))
        std_z = 1.96 # from z-table for 95%
        confidence_interval = std * std_z
        plt.plot(x[:,1], model.predict(x), label="Linear Regression")
        plt.plot(x[:,1], model.predict(x) - confidence_interval,label="95%-")
        plt.plot(x[:,1], model.predict(x) + confidence_interval,label="95%+")
        [xmin,xmax] = ax.xaxis.get_view_interval()
        [ymin,ymax] = ax.yaxis.get_view_interval()
        #params (y=ax+b), R2 and p-value
        a,b = model.params[1],model.params[0]
        r2 = round(model.rsquared, 3)
        p_value = round(model.summary2().tables[1]['P>|t|']['x1'], 4)
        plt.text(0.95*xmin+0.05*xmax, 0.9*ymax+0.1*ymin, '$y$ = {:.2f} + {:.2f}$x$'.format(b,a))
        plt.text(0.95*xmin+0.05*xmax, 0.85*ymax+0.15*ymin, f'$R$$^{2}$ = {r2}')
        plt.text(0.95*xmin+0.05*xmax, 0.8*ymax+0.2*ymin, f'$p-value$ = {p_value}')
        plt.legend(loc=4)
    plt.savefig('Argoverse VS Summit Sparse.png')



