import numpy as np
from math import fsum, pow, comb


def P_A11B1(i, k):
    return 0 if i == k else fsum([(m - u) * comb(u - 1, i - k - 1) * fsum([(u - v) * 
                         comb(m - u - v, k - 1) for v in range(1, int(m - u - k + 1))])
                                for u in range(int(i - k), int(m - k + 1))]) / (
                                      (i - k + 1) * pow(m, k) * comb(m + 1, i - k + 2))


def P_A12B1(i, k):
    return 0 if i == k else fsum(
        [(m - u) * comb(u - 1, i - k - 1) * fsum([(u + v) * (u + v + (k - 1) * 
            (m + 1)) * comb(m - u - v, k - 1) for v in range(1, int(m - u - k + 1))])
                for u in range(int(i - k), int(m - k + 1))]) / (
                    k * (i - k + 1) * pow(m, k + 1) * comb(m + 1, i - k + 2))


def P_A21B2(i, k):
    return 0 if i == k else fsum([u * comb(u - 1, i - k - 1) * comb(m - u, k)
                                  for u in range(int(i - k), int(m - k + 1))]) / (
                                        (i - k) * pow(m, k) * comb(m + 1, i - k + 1))


def P_A22B2(i, k):
    return 0 if i == k else fsum([u * (u + (m + 1) * k) * comb(u - 1, i - k - 1) * 
                      comb(m - u, k) for u in range(int(i - k), int(m - k + 1))]) / (
                                    (k + 1) * (i - k) * pow(m, k + 1) * 
                                                               comb(m + 1, i - k + 1))




def delta_1(i):
    if i == 0:
        return 1
    elif i >= m:
        return 0
    return 1 / (i + 1)


def delta_2(i, j, k):
    if i < 0 or i > m or j < 0 or j > n:
        return 0
    elif k == 0:
        return 1
    elif j == 0:
        return P_A11B1(i, k)
    return P_A12B1(i, k)


def delta_3(i, j, k):
    if i < 0 or i > m or j < 0 or j > n:
        return 0
    elif k == 0:
        return 1
    elif j == 0:
        return P_A21B2(i, k)
    return P_A22B2(i, k)


Z = lambda z: format(abs(z), '.3g')


def matrices_construct(n, m, a, b, w):
    A, B = [], []
    C = np.array(np.zeros((m + 1, n + 1), dtype=np.float64))

    C[0, 0] = -1
    C[1, 0] = a / b
    A.append(C.flatten().tolist())
    B.append(0)

    for j in range(1, n + 1):
        C = np.array(np.zeros((m + 1, n + 1), dtype=np.float64))
        C[0, j] = -1
        A.append(C.flatten().tolist())
        B.append(0)
    
    for i in range(1, m + 1):
        C = np.array(np.zeros((m + 1, n + 1), dtype=np.float64))
        C[i, 0] = -1
        C[i - 1, 0] = a * delta_1(i - 1) / (a + i * b)
        K = 1 if i == m else 0
        k1 = min(i, n)
        for k in range(K, k1 + 1):
            C[i - k + 1, k] += b * (i - k + 1) * delta_2(i, 0, k) / (a + i * b)
        k2 = min(i - 1, n - 1)
        for k in range(0, k2 + 1):
            C[i - k, k + 1] += w * delta_3(i, 0, k) / (a + i * b)
        A.append(C.flatten().tolist())
        B.append(0)

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            C = np.array(np.zeros((m + 1, n + 1), dtype=np.float64))
            C[i, j] = -1
            C[i, j - 1] = a * (1 - delta_1(i - 1)) / (a + i * b + j * w)
            K = 1 if i == m else 0
            k1 = min(i, n - j)
            for k in range(K, k1 + 1):
                C[i - k + 1, k + j] += b * (i - k + 1) *
                                               delta_2(i, j, k) / (a + i * b + j * w)
            if j != n:
                k2 = min(i - 1, n - j - 1)
                for k in range(0, k2 + 1):
                    C[i - k, k + j + 1] += w * delta_3(i, j, k) / (a + i * b + j * w)
                C[i, j + 1] += w * j / (a + i * b + j * w)
            A.append(C.flatten().tolist())
            B.append(0)
    
    C = np.array(np.ones((m + 1, n + 1), dtype=np.float64))
    A[(m + 1) * (n + 1) - 1] = C.flatten().tolist()
    B[(m + 1) * (n + 1) - 1] = 1
    return np.array(A), np.array(B)

    
def stead_state_probabilities(n, m, a, b, w):
    A, b = matrices_construct(n, m, a, b, w)
    x = np.linalg.solve(A, b)
    return x

    
def system_performance_measures(n, m, a, b, w, file_name='output.txt'):
    x = stead_state_probabilities(n, m, a, b, w)
    Pr, Pf = 0, 0
    i, j = 0, 0 
    
    with open(file_name, 'w') as f:
        for xx in x:
            if j == n:
                Pr += a * xx / (a + i * b + j * w)
            Pf += j * w * xx / (a + i * b + j * w)
            f.writelines(f'P_[{i}, {j}] = ' + format(abs(xx), '.3g') + '\n')
            j += 1
            if j == n + 1:
                j = 0
                i += 1
                
        Pa = (1 - Pr) * (1 - Pf)
        lambda_eff = a * Pa
        rho = lambda_eff * (m + 1) / (2 * m * b)

        f.writelines(f'P_r = {Z(Pr)}\n')
        f.writelines(f'P_f = {Z(Pf)}\n')
        f.writelines(f'P_a = {Z(Pa)}\n')
        f.writelines(f'\u03BB = {Z(lambda_eff)}\n')
        f.writelines(f'\u03C1 = {Z(rho)}\n')
    return Pr, Pf, rho


def  optimal_queue_length(m, a, b, w):
    opt_n, n = 1, 1
    Pr, Pf, opt_rho =  system_performance_measures(m, n, a, b, w)
    M1 = Pr + (1 - Pr) * Pf
    print(f'Min1 = {F(M1)}')
    n += 1
    Pr, Pf, rho =  system_performance_measures(m, n, a, b, w)
    M2 = Pr + (1 - Pr) * Pf
    
    while M1 > M2:
        opt_n = n
        n += 1
        M1 = M2
        opt_rho = rho
        Pr, Pf, rho =  system_performance_measures(m, n, a, b, w)
        M2 = Pr + (1 - Pr) * Pf
        
    if opt_rho >= 1:
        print("These parameters are not good for the system's effective performance!")
    else:
        print(f"The optimal queue length is {opt_n} and P_l = {Z(M1)}")
        return opt_n