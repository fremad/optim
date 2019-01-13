import matplotlib.pyplot as plt
import numpy as np
import sympy as sym



# a0 = 1
# b0 = 2
# ro = 0.382
#
#
# def func(x):
#     return x**2+4*np.cos(x)
#
# x = np.arange(a0,b0,0.01,np.float64)
#
# plt.plot(x,func(x))
#
#
#
# a1 = a0+(b0-a0)*ro
# b1 = b0-(b0-a0)*ro
#
# print(a1,b1,func(a1),func(b1))
#
# # plt.plot(a1,func(a1),'ro')
# # plt.plot(b1,func(b1),'go')
#
# a2 = a1+(b0-a1)*ro
# b2 = b0-(b0-a1)*ro
#
# print(a2,b2,func(a2),func(b2))
# # plt.plot(a2,func(a2),'ro')
# # plt.plot(b2,func(b2),'go')
#
# a3 = b2
# b3 = b0-(b0-a2)*ro
# print(a3,b3,func(a3),func(b3))
# # plt.plot(a3,func(a3),'ro')
# # plt.plot(b3,func(b3),'go')
#
# a4 = b3
# b4 = b0-(b0-a3)*ro
# print(a4,b4,func(a4),func(b4))
# plt.plot(a4,func(a4),'ro')
# plt.plot(b4,func(b4),'go')
#
# plt.show()
X = np.array([[0,0],[1,1],[-1,0],[0.7,-0.2],[-0.2,1.5]])

c1 = np.array([[-1,0],[0,-1],[-0.5,-0.5],[-1.5,-1.5],[-2,0],[0,-2],[-1,-1.3]])
c2 = np.array([[1,1],[1.3,0.7],[0.7,1.3],[2.5,1],[0,1]])



#
# def p_x_c(X,c1,c2):
#
#     def prop_form(X,mean):
#         #Preallocate
#
#         # tmp = np.array([np.exp(-1 * np.linalg.norm(X - mean[0])) \
#         # / (np.exp(-1 * np.linalg.norm(X - mean[0])) +
#         #    np.exp(-1 * np.linalg.norm(X - mean[1]))),
#         #           np.exp(-1 * np.linalg.norm(X - mean[1])) \
#         #           / (np.exp(-1 * np.linalg.norm(X - mean[0])) +
#         #              np.exp(-1 * np.linalg.norm(X - mean[1])))
#         #           ])
#
#         #Covariance matrix
#         # Sigma = np.array([[1,0],[0,2]])
#         # Sigma = np.invert(Sigma)
#         #
#         # tmp = np.array([np.exp(-1*np.transpose(X-mean[0]).dot(Sigma.dot(np.transpose(X-mean[0]))))
#         #                 /(np.exp(-1*np.transpose(X-mean[1]).dot(Sigma.dot(np.transpose(X-mean[1]))))+
#         #                   np.exp(-1 * np.transpose(X - mean[0]).dot(Sigma.dot(np.transpose(X - mean[0]))))),
#         #                 np.exp(-1*np.transpose(X-mean[1]).dot(Sigma.dot(np.transpose(X-mean[1]))))
#         #                 /(np.exp(-1*np.transpose(X-mean[1]).dot(Sigma.dot(np.transpose(X-mean[1]))))+
#         #                   np.exp(-1 * np.transpose(X - mean[0]).dot(Sigma.dot(np.transpose(X - mean[0])))))])
#
#         return tmp
#
#     #Calculate mean of vectors
#     mean = np.zeros((2,2))
#     mean[0] = np.mean(c1, axis=0)
#     mean[1] = np.mean(c2, axis=0)
#
#
#
#     res = np.zeros((X.shape[0],X.shape[1]))
#
#     #Assign propabilities p(x|c) from formulae prop_form
#     for i in range(X.shape[0]):
#         res[i] = prop_form(X[i],mean)
#
#     #Count entries and multiply
#     P_c1 = c1.shape[0]/(c1.shape[0]+c2.shape[0])
#     P_c2 = c2.shape[0]/(c1.shape[0]+c2.shape[0])
#
#
#     for i in range(X.shape[0]):
#         res[i][0] = res[i][0]*P_c1
#         res[i][1] = res[i][1]*P_c2
#
#
#     # #Define risktable COMMENT IF NOT NEEDED!!!
#     # risk = np.array([[0.3, 0.7], [0.8, 0.2]])
#     #
#     # risk = np.transpose(risk)
#     # for i in range(X.shape[0]):
#     #     tmp1 = res[i].dot(risk[0])
#     #     tmp2 = res[i].dot(risk[1])
#     #     res[i][1] = tmp2
#     #     res[i][0] = tmp1
#
#
#
#     return res



#sol = p_x_c(X,c1,c2)
#print(sol)

Sigma = np.array([[1,0],[0,2]])
Sigma = np.invert(Sigma)

a =np.exp(-np.transpose(1*np.array([0,0])-np.array([-0.8571,-0.9])).dot(Sigma.dot(np.array([0,0])-np.array([-0.8571,-0.9]))))

b = np.exp(-np.transpose(1*np.array([0,0])-np.array([1.1,1])).dot(Sigma.dot(np.array([0,0])-np.array([1.1,1]))))

print(a/(a+b))