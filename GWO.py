# Get the GWO Pseudocode from --> https://www.researchgate.net/figure/Pseudo-code-of-improved-Grey-Wolf-Optimization-IGWO-algorithm_fig3_350143760
import numpy as np

def GWO(population,objective_function, search_space, population_size, iterations):
    search_space,D = population.shape[0],population.shape[1]
    alpha, beta, delta = np.inf, np.inf, np.inf
    best_sub= np.zeros((iterations, D))
    convergence = np.zeros((iterations,best_solution))
    fitness_alpha, fitness_beta, fitness_delta = float('inf'), float('inf'), float('inf')
    ct = time.time()
    for iteration in range(iterations):
        for i in range(population_size):
            fitness_i = objective_function(population[i, :])
            minf = min(fitness_i); iminf = np.where(min(fitness_i) == minf)
            best_solution[iteration, :] = population[iminf,:]

            if fitness_i < fitness_alpha:
                alpha = population[i, :].copy()
                fitness_alpha = fitness_i
            elif fitness_i < fitness_beta:
                beta = population[i, :].copy()
                fitness_beta = fitness_i
            elif fitness_i < fitness_delta:
                delta = population[i, :].copy()
                fitness_delta = fitness_i

        a = 2 - 2 * (iteration / iterations)  # Alpha decreases linearly from 2 to 0

        for i in range(population_size):
            r1, r2 = np.random.random(), np.random.random()
            A1, A2, A3 = 2 * a * r1 - a, 2 * a * r2 - a, 2 * r1

            C1 = 2 * np.random.random()
            D_alpha = abs(C1 * alpha - population[i, :])
            X1 = alpha - A1 * D_alpha

            C2 = 2 * np.random.random()
            D_beta = abs(C2 * beta - population[i, :])
            X2 = beta - A2 * D_beta

            C3 = 2 * np.random.random()
            D_delta = abs(C3 * delta - population[i, :])
            X3 = delta - A3 * D_delta

            population[i, :] = (X1 + X2 + X3) / 3
        convergence[iteration] = best_fitness
    ct = time.time()-ct
    return best_solution,convergence, best_fitness,ct


