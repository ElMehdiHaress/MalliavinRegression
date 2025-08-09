import matplotlib.pyplot as plt
from tqdm import tqdm
import malliavin_regression as mr

# Set parameters for the experiment
n_train = 50
noise_variance = 0.2
model_name = "sinus"
n_test = 500

# params for H1 and H2 gradient descents
step1 = 0.1
step2 = 1
n_iter = 5000

# params for gradient-free descent
n_iter_gfree = 10000
gamma = 1
step_gfree = 0.01
n_brownians = 100

t = np.linspace(0,1,1000)
plt.plot(t,mr.payoff1D(t, model_name))
plt.title("Payoff function")
plt.show()

X_train, Y_train, X_test = mr.generate_data(n_train, n_test, noise_var=noise_variance, name_model=model_name, show=True)
y_test = mr.payoff1D(X_test, model_name)

F_train_h1 = mr.gradient_descent(X_train, Y_train, f0, 1, n_iter, step1)
y_pred_test_h1 = mr.f(X_test, F_train_h1, f0, X_train, Y_train, 1, step1)

F_train_h2 = mr.gradient_descent(X_train, Y_train, f0, 2, n_iter, step2)
y_pred_test_h2 = mr.f(X_test, F_train_h2, f0, X_train, Y_train, 2, step2)

B = []
for i in tqdm(range(n_iter_gfree)):
    B.append(mr.brownian(n_brownians))
    
F_train_gfree = mr.gradient_free_descent(X_train, Y_train, f0, gamma, n_iter_gfree, step_gfree)
y_pred_test_gfree = mr.gradient_free_f(X_test, F_train_gfree, f0, X_train, Y_train, gamma, step_gfree)

knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
knn_pred_y_test = knn.fit(X_train.reshape((-1, 1)), Y_train).predict(X_test.reshape((-1, 1)))

gpr = GaussianProcessRegressor().fit(X_train.reshape((-1, 1)), Y_train)
gpr_pred_y_test = gpr.predict(X_test.reshape((-1, 1)))

krr = KernelRidge(alpha=.0001, kernel=RBF()).fit(X_train.reshape((-1, 1)), Y_train)
krr_pred_y_test = krr.predict(X_test.reshape((-1, 1)))

test_order = np.argsort(X_test)
plt.title("n_iter = {}, n_train = {}, noise variance = {}".format(n_iter,n_train, noise_variance))

sorted_X_test = np.sort(X_test)
plt.plot(sorted_X_test, knn_pred_y_test[test_order], label='KNN', color='y')
plt.plot(sorted_X_test, krr_pred_y_test[test_order], label='KRR', color='orange')
plt.plot(sorted_X_test, gpr_pred_y_test[test_order], label='GP', color='grey')
plt.plot(sorted_X_test, y_pred_test_h1[test_order], label='GD_H1', color='b')
plt.plot(sorted_X_test, y_pred_test_h2[test_order], label='GD_H2', color='r')
plt.plot(sorted_X_test, y_pred_test_gfree[test_order], label='GD_free', color='black')
plt.plot(sorted_X_test, y_test[test_order], label='Ground truth', color='g')
plt.legend()
plt.show()


y_pred_deriv_h1 = np.array([derivative(f, x, dx = 1e-6, args=(F_train_h1, f0, X_train, Y_train,1, step1, True, 
                                                              False)) 
                            for x in tqdm(X_test)])
y_pred_deriv_h2 = np.array([derivative(f, x, dx = 1e-6, args=(F_train_h2, f0, X_train, Y_train, 2, step2, True, 
                                                              False)) 
                            for x in tqdm(X_test)])
# y_pred_deriv_gfree = [derivative(gradient_free_f, x, dx = 1e-6, args=(F_train_gfree, f0, X_train, Y_train, gamma, 
#                                                                       step_gfree, True, False)) 
#                       for x in tqdm(X_test)]
true_deriv = np.array([derivative(partial(payoff1D, **{"name": model_name}), x, dx = 1e-6) for x in sorted_X_test])

plt.plot(sorted_X_test, y_pred_deriv_h1[test_order], label='GD-deriv_H1', color='b')
plt.plot(sorted_X_test, y_pred_deriv_h2[test_order], label='GD-deriv_H2', color='r')
# plt.plot(sorted_X_test, y_pred_deriv_gfree[test_order], label='GD-gfree',color='black')
plt.plot(sorted_X_test, true_deriv, label='Ground truth-deriv', color='g')
plt.legend()
plt.show()

print("Test L2 loss for the H1 gradient descent: {:.3f}".format(np.linalg.norm(y_test - y_pred_test_h1)))
print("Test L2 loss for the H2 gradient descent: {:.3f}".format(np.linalg.norm(y_test - y_pred_test_h2)))
print("Test L2 loss for the gradient-free descent: {:.3f}".format(np.linalg.norm(y_test - y_pred_test_gfree)))
print("Test L2 loss for the KNN regressor: {:.3f}".format(np.linalg.norm(y_test - knn_pred_y_test)))
print("Test L2 loss for the KRR: {:.3f}".format(np.linalg.norm(y_test - krr_pred_y_test)))
print("Test L2 loss for the GP regressor: {:.3f}".format(np.linalg.norm(y_test - gpr_pred_y_test)))

#Comparison of the losses on the test sample
loss_train_h1 = np.mean((F_train_h1 - Y_train.reshape((1, -1)))**2, axis=1)
loss_train_h2 = np.mean((F_train_h2 - Y_train.reshape((1, -1)))**2, axis=1)
loss_train_gfree = np.mean((F_train_gfree - Y_train.reshape((1, -1)))**2, axis=1)

plt.figure()
plt.plot(loss_train_h1, label="GD_H1", color='blue')
plt.plot(loss_train_h2, label="GD_H2", color='red')
plt.plot(loss_train_gfree, label="GD_free", color='black')
plt.legend()
plt.show()

