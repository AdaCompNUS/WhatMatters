import matplotlib
matplotlib.use('pdf') # or 'eps' or 'svg'

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from decimal import Decimal

import random
import matplotlib.colors as mcolors
# Results in Argoverse Validation Dataset
# Model	    minFDE(K=6)    minADE(K=6)   FDE(K=1)	    ADE(K=1)
# CV	        3.799	       1.687	     6.053	        2.715
# CA 	        3.727	       1.674	     6.174	        2.777
# KNN	        2.579	       1.412	     6.721	        2.998
# s-KNN       2.814          1.534         7.338          3.322
# lstm	    3.829	       1.682	     3.923	        1.722
# s-lstm	    3.818	       1.688	     3.954	        1.741
# map-lstm	3.914	       1.911	     5.966	        2.660
# LaneGCN	    1.081	       0.711	     3.020	        1.367
# HiVT	    0.969	       0.661	     3.547	        1.609
# DSP         1.015          0.727         3.050          1.415
# HOME        2.653          1.323         6.338          2.763 

# Results in Summit_10Hz Dataset
# CV	        1.189	       0.662	     2.012	        1.020
# CA	        1.229	       0.675	     1.984	        1.005
# KNN	        1.018	       0.674	     2.565	        1.291
# s-KNN	    1.033	       0.680	     2.594	        1.309
# lstm	    1.660	       0.839	     1.706	        0.859
# s-lstm		1.846	       0.937	     1.905	        0.963
# map-lstm	2.339	       1.300	     2.619	        1.459
# LaneGCN	    0.513	       0.370	     1.276          0.637
# HiVT        0.401	       0.350	     1.356	        0.692
# DSP         0.486          0.391         1.418          0.717
# HOME        1.461          0.813         2.594          1.282

# # Results in Summit Validation Dataset 14K
# # CV	        3.962	       1.965	     5.950	        2.938
# # CA	        3.986	       2.003	     6.040	        2.989
# # KNN	        2.742	       1.667	     6.505	        3.099
# # s-KNN	    2.853	       1.740	     6.633	        3.196
# # lstm	    4.924	       2.348	     5.061	        2.410
# # s-lstm		4.954	       2.396	     5.152	        2.480
# # map-lstm	4.819	       2.695	     5.934	        3.301
# # LaneGCN	    1.503	       1.016	     4.189          1.944
# # HiVT        1.064	       0.807	     3.524	        1.692
# # DSP         1.260          0.983         3.602          1.728
# # HOME        7.226          3.417         15.30          6.761   (I train) 

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']


def read_txt(inputpath):
    with open(inputpath, 'r', encoding='utf-8') as infile:
        result = {}
        keys = []
        for line in infile:
            data_line = line.strip("\n").split()
            if line == "\n":
                continue
            elif "Dataset" in data_line:
                dataset = data_line[2]
                dataset_count = 0 if dataset == "Argoverse" else 1
                print("Obtaining Prediction metrics from %s dataset" % dataset)
            elif "Model" in data_line:
                keys = data_line[1:]
                for key in keys:
                    result[key] = [[],[]]
                print("Prediction Metrics:", keys)
            else:
                for i, key in enumerate(keys):
                    result[key][dataset_count].append(float(data_line[i+1]))
        
        for key in keys:
            result[key] = np.array(result[key])
        
        return result

if __name__ == "__main__":

    ## Analyze the results that training and validating in Dense Summit Dataset ###

    methods = ["CV", "CA", "KNN", "S-KNN", "LSTM" ,"S-LSTM", "LaneGCN", "HiVT", "DSP","HOME"]
    plot = {}
    result = read_txt('pred_metrics.txt')

    #fig = plt.figure(figsize=(60, 40))
    plt.rcParams.update({'font.size': 12})
    for i, key in enumerate(result):
        fig = plt.figure(figsize=(4, 4))
        #fig.set_size_inches(15,15)
        plt.xlabel("Argoverse", fontsize=16)
        plt.ylabel("SUMMIT", fontsize=16)
        plt.title(f"{key}", fontsize=16)
        
        x = result[key][0,:].reshape(-1,1)
        y = result[key][1,:].reshape(-1,1)
        # plt.xlim(0,max(x))
        # plt.ylim(0,max(x))
        #plt.scatter(x, y)

        for j in range(len(x)):
            bias_x = -random.uniform(0.05, 0.1)  # Randomize bias to avoid overlapping
            bias_y = random.uniform(0.03, 0.05)
            color = colors[j % len(colors)]  # Assign a color from the predefined set
            plt.scatter(x[j], y[j], color='black', s=5)  # Set the scatter point color
            plt.annotate(methods[j], xy=(x[j], y[j]), xytext=(x[j]+bias_x, y[j]+bias_y),
                 color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=0), fontsize=8)

        
        x = sm.add_constant(x)
        model = sm.OLS(y,x).fit()
        std=np.std(model.predict(x))
        std_z = 1.96 # from z-table for 95%
        confidence_interval = std * std_z
        plt.plot(x[:,1], model.predict(x), label="Best-fit Line", color='black')
        #plt.plot(x[:,1], model.predict(x) - confidence_interval,label="Lower 95% CI")
        #plt.plot(x[:,1], model.predict(x) + confidence_interval,label="Upper 95% CI")
        x_sorted = np.stack(sorted(x, key = lambda x: x[1]), axis=0)
        print(x_sorted)
        plt.fill_between(x_sorted[:, 1], model.predict(x_sorted) - confidence_interval, model.predict(x_sorted) + confidence_interval, 
                         color='blue', alpha=0.1, label="95% CI")

        ymax_confidence_interval = max(model.predict(x) + confidence_interval)
        ymin_confidence_interval = min(model.predict(x) - confidence_interval)
        
        [xmin,xmax] = plt.xlim()
        [ymin,ymax] = plt.ylim()
        # params (y=ax+b), R2 and p-value
        a,b = model.params[1],model.params[0]
        r2 = round(model.rsquared, 3)
        p_value = '%.2E' % Decimal(model.summary2().tables[1]['P>|t|']['x1'])
        print(f"r2 = {r2}, p-value = {p_value}")
        #plt.text(0.95*xmin+0.05*xmax, ymax_confidence_interval, '$y$ = {:.2f} + {:.2f}$x$'.format(b, a))
        #plt.text(0.95*xmin+0.05*xmax, ymax_confidence_interval - 0.1 * (y.max() - y.min()), f'$R$$^{2}$ = {r2}')
        #plt.text(0.95*xmin+0.05*xmax, ymax_confidence_interval - 0.2 * (y.max() - y.min()), f'$p-value$ = {p_value}')
        plt.xlim(right=xmax*1.06)
        #plt.legend(loc=4)
        plt.tight_layout() 
        plt.grid()
        plt.grid(True, linewidth=0.5, alpha=0.7)
        plt.savefig(f'Argoverse_VS_Summit_{key}.pdf', dpi=300)
        plt.close(fig)

