import numpy as np
import matplotlib.pyplot as plt



n = [0,10,20,40,80,160,320,640]
error_rate_knn = [0.352873876328,0.387737617135,0.453305017255,0.458524980174,0.474808757584,0.470144927536,0.473847774559,0.467094872065]
error_rate_svm = [0.357722691365,0.355341365462,0.355402176799,0.352894528152,0.352941176471,0.352621870883,0.352541480116,0.352532378646]

knn_error = list(reversed(error_rate_knn))
svm_error = list(reversed(error_rate_svm))

logLik_knn = [20105,20781,21577,21718,21918,21897,21944,21890]
logLik_svm = [20150,20479,20655,20719,20763,20782,20792,20802]

knn_log = list(reversed(logLik_knn))
svm_log = list(reversed(logLik_svm))

svm_supervised = np.repeat(0.436641221374,8)
knn_supervised = np.repeat(0.386943932614,8)

plt.figure(1)
plt.subplot(211)
plt.plot(n,knn_error, '-')
plt.plot(n, knn_supervised, '-')
plt.ylabel('Error Rate')
plt.title('Error Rate for Semi supervised KNN')

plt.subplot(212)
plt.plot(n,svm_error, '-')
plt.plot(n, svm_supervised, '-')
plt.xlabel('Unlabelled Data')
plt.ylabel('Error Rate')
plt.title('Error Rate for CPLE SVM')



plt.figure(2)
plt.subplot(211)
plt.plot(n,knn_log, '-')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood for Semi supervised KNN')

plt.subplot(212)
plt.plot(n,svm_log, '-')
plt.xlabel('Unlabelled Data')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood for CPLE SVM')
plt.show()
