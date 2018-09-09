import numpy as np

def get_performance(y_actual, y_hat):

  # assert y_actual.shape[0] == y_pred.shape[0]
  # assert y_actual.shape[1] == y_pred.shape[1]
  # y_actual = np.asarray(y_actual)
  # y_pred = np.asarray(y_pred)

  # TP = 0
  # FP = 0
  # TN = 0
  # FN = 0

  # for i in range(len(y_pred)):
  #   if y_actual[i,1] == y_pred[i,1] == 1:
  #     TP += 1
  #   if y_pred[i,1] == 1 and y_actual[i,1] != y_pred[i,1]:
  #     FP += 1
  #   if y_actual[i,0] == y_pred[i,0] == 1:
  #     TN += 1
  #   if y_pred[i,0] == 1 and y_actual[i,0] != y_pred[i,0]:
  #     FN += 1

  # return (TP, FP, TN, FN)

  assert len(y_actual) == len(y_hat)

  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i in range(len(y_hat)):
    if y_actual[i] == y_hat[i] == 1:
     TP += 1
    if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
     FP += 1
    if y_actual[i] == y_hat[i] == 0:
     TN += 1
    if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
     FN += 1

  return (TP, FP, TN, FN)


def get_performance_on_mask_pred(mask, pred):
  mask_reshaped = np.squeeze(np.reshape(mask, (1, -1)))
  pred_reshaped = np.squeeze(np.reshape(pred, (1, -1)))
  TP, FP, TN, FN = get_performance(mask_reshaped, pred_reshaped)
  return (TP, FP, TN, FN)

