from sklearn.metrics import precision_recall_fscore_support as score

predicted = [1,2,3,4,5,1,2,1,1,4,5]
y_test = [1,2,3,4,5,1,2,1,1,4,1]

precision, recall, fscore, support = score(y_test, predicted)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

# scipy has an instalaltion issue on my Mac and cannot be used.