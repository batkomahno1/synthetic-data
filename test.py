#!/usr/bin/env python3

# import numpy as np
# import time
# start = time.time()
# import pandas as pd
# data=pd.read_csv('car.data', header=None)

# data.drop(data[data.eq(' ?').any(1)].index, inplace=True)
# data.reset_index(inplace=True, drop=True)
# data_discrete=data

# drops=[]
# nans=[]

# for i in range(2):   
# #     IPython.kernel.manager.KernelManager.restart_kernel()
#     import jtree_refactored_procedural as jt
#     try:
#         result_table, tree_marginals = jt.JTree(data_discrete, verbose=False).synthesize()
#     #     print(f'Percent NaN in synthetic data:\n{result_table.isna().sum()/result_table.shape[0]}')
#         temp=result_table.shape[0]
#         numpoints = result_table.shape[0]*result_table.shape[1]
#         nans.append(result_table.isna().values.sum())
#         result_table = result_table.dropna()
#         dropped=(temp-result_table.shape[0])/temp
#         print(f'Dropped {dropped:.2f}')
#         print(f'NaNs : {nans[-1]/numpoints:.2f}')
#         drops.append(dropped)
#     finally:
#         del jt
        
        

# print(f'\navg blind row drop {np.array(drops).mean():.2f}')
# print(f'med blind row drop {np.median(drops):.2f}')
# print(f'avg nans points {np.array(nans).mean()/numpoints:.2f}')
# print(f'med nans points {np.median(nans)/numpoints:.2f}')
# print(f'Time : {round(time.time()-start)} s')

# import matplotlib.pyplot as plt

# plt.scatter(range(len(nans)),nans)
# plt.title('Percent NaNs in sets')
# plt.show()

###############################
import subprocess
try:
#    subprocess.Popen(["ipcluster", "start"])
#    print('Loading engines...')

#    import time
#    # wait 10s for engines to start
#    time.sleep(10)
    
    
    import time
    start = time.time()
    import pandas as pd


    from sklearn import datasets
    import numpy as np
    from sklearn import preprocessing

    var=datasets.fetch_covtype()
    X = np.hstack([var.data, var.target[:,np.newaxis]])
    data = pd.DataFrame(X)
    print(f'Dataset Size: {data.shape}')

    from sklearn.preprocessing import KBinsDiscretizer  
    disc = KBinsDiscretizer(strategy='uniform')
    data_discrete=pd.DataFrame(disc.fit_transform(data).toarray())
    print(data_discrete.shape)

    from sklearn.model_selection import train_test_split
    X, y = train_test_split(data_discrete)


    print(f'Training\n{X.sample(5)}')
    print(f'Testing\n{y.sample(5)}')

    res=[]
    for i in range(5):    
        print('-'*25,i,'-'*25)

        import jtree_refactored_procedural as jt

        try:
            result_table, tree_marginals = jt.JTree(X, verbose=True).synthesize()

            res.append((result_table, tree_marginals))
        except Exception as ex:
    #         raise
            print(ex)
            print('*'*40,'FAILED','*'*40)
            res.append("FAILED")
        finally:
            pass

    tables=[mat[0] for mat in res if not isinstance(mat[0],str)]
    nans = [mat.isna().values.sum()/(mat.shape[0]*mat.shape[1]) for mat in tables]

    import numpy as np
    import matplotlib.pyplot as plt

    plt.scatter(range(len(nans)),nans)
    plt.title('Percent NaNs in sets')
    plt.show()

    print(f'Mean NaN is {np.array(nans).mean()}')

    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(max_iter=500)

    # from sklearn.model_selection import cross_val_score
    # acc_raw = cross_val_score(model, X, y, cv=10).mean()

    accs=[runModelSimple(y, table, model, 'MLP', 'adult') for table in tables]
    print(f'Mean acc for {len(accs)} runs is {np.array(accs).mean()}')

    X = pd.get_dummies(data_discrete.iloc[:,:-1])#data.iloc[:,:-1].copy()
    y = data_discrete.iloc[:,-1].copy()

    print(f'Time : {round(time.time()-start)}')
    
finally:
    subprocess.Popen(["ipcluster", "stop"])
    time.sleep(5)
