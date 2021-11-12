import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


##----- Define model names --
name = ['PI3NN_OOD','PI3NN_No_OOD','QD','PIVEN','SQR','DER']


##############################################################
###=========== plot PDF of ID and OOD =============
##############################################################
plt.figure(figsize=(12, 2.2), dpi=200)
fontsize = 9

for i in range(len(name)):

    print('----------- %s ----------'%name[i])

    #---------- load the data -----------
    train = np.loadtxt('%s_MPIW_train.dat'%name[i])
    test = np.loadtxt('%s_MPIW_test.dat'%name[i])
    # print('min and max of train',train.min(), train.max())
    # print('min and max of test',test.min(), test.max())

    train_kde = stats.gaussian_kde(train)
    test_kde = stats.gaussian_kde(test)

    d1 = train.max() - train.min()
    a1 = train.min()-0.1*d1; b1 = train.max()+0.1*d1
    train_x = np.linspace(a1, b1, 1000)
    train_p = train_kde(train_x)

    d2 = test.max() - test.min()
    a2 = test.min()-0.1*d2; b2 = test.max()+0.1*d2
    test_x = np.linspace(a2, b2, 1000)
    test_p = test_kde(test_x)

    
    ###------- Option I: calculate confidence interval
    # train_score = train_kde(train)/train_p.max()
    # test_score = train_kde(test)/train_p.max()
    
    train_score = np.clip(train.mean()/train, 0.0, 1.0)
    test_score = np.clip(train.mean()/test, 0.0, 1.0)

    print('--------- Confidence score of %s ------------'%name[i])
    print('train_mean=%6.4f and train_std=%6.4f'%(np.mean(train_score), np.std(train_score)))
    print('test_mean=%6.4f and test_std=%6.4f'%(np.mean(test_score), np.std(test_score)))

    #---- Plot PDF of train and test data
    plt.subplot(1, len(name), i+1)
    plt.plot(train_x, train_p, 'r');
    plt.plot(test_x, test_p, 'b');
    plt.fill_between(train_x, train_p, facecolor='r', alpha=0.3, label='Train (InD)')
    plt.fill_between(test_x, test_p, facecolor='b', alpha=0.3, label='Test (OOD)')
    plt.xlabel('PI width',fontsize=fontsize)
    plt.ylabel('Density',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlim([np.floor(min(a1,a2)), np.ceil(max(b1,b2))])
    plt.title('%s'%name[i], fontsize=fontsize)

    if i==5:
        plt.xlim([1,8.5])
    elif i==0:
        plt.title('PI3NN (With OOD)', fontsize=fontsize)
        # plt.text(4.5,0.24,'OOD samples are identified. \n ID and OOD samples are \n well separated.', fontsize=6)
    elif i==1:
        plt.title('PI3NN (No OOD)', fontsize=fontsize)
    plt.ylim([0,1.3*np.max(np.concatenate((train_p,test_p)))])

    plt.legend(loc='upper right',ncol=1, fontsize=7,frameon=False)

# plt.subplots_adjust(bottom=0.2, wspace=0.3)
plt.tight_layout()
plt.savefig('OOD.png', pad_inches=0.1, dpi=400)
plt.show()



