import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Configurações dos Access Points
APs = {
    "A": {"loc": [0, 0], "cap": 64},
    "B": {"loc": [80, 0], "cap": 64},
    "C": {"loc": [0, 80], "cap": 128},
    "D": {"loc": [80, 80], "cap": 128},
}


# Função para calcular a distância entre dois pontos
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


# Carregar posições dos clientes do arquivo CSV
def load_clients(file_path):
    clients = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file , delimiter=";")
        next(reader)  # Pular cabeçalho
        for row in reader:
            print( row[1] , " - " , row[2] )
            clients.append([int(row[1]), int(row[2])])
    return clients

def load_result(file_path):

    df = pd.read_csv(file_path)
    return df


# Inicializar população aleatória
## Recebe o número de clientes no ambiente e gera uma lista de tamanho pop_size na qual cada item da lista são os possíveis acessos de cada cliente
## Retorna uma lista , onde cada item dessa lista é justamente uma possível solução ["A" , "A" , "C" , ... ]
def initialize_population(num_clients, pop_size):
    return [
        [random.choice(list(APs.keys())) for _ in range(num_clients)] # Aleatoriamente atribui uma AP para cada Cliente possível
        for _ in range(pop_size)
    ]


# Avaliação da fitness (distância total e capacidade dos APs)
## Basciamente ela vai usar a distancia dos pontos até o AP e uma penalização para quando um AC excender o limite
## Para cada possível solução dada pela população , calculamos a distância do ponto até o AP dele
## Para cada possível solução dada pela população , verficamos se o APa excedeu o limite e penalizamos
def evaluate_fitness(solution, clients):
    total_distance = 0
    capacity_violation = 0
    ap_loads = {ap: 0 for ap in APs.keys()}

    for i, ap in enumerate(solution):
        # print(f"Cliente {i} está em {ap} , calculando a distância de {clients[i]} até {APs[ap]["loc"]}")
        ap_loads[ap] += 1
        total_distance += calculate_distance(clients[i], APs[ap]["loc"])

    # Penalidade por violação de capacidade
    for ap, load in ap_loads.items():
        if load > APs[ap]["cap"]:
            # print(f"Load: {load} e Capacidade: {APs[ap]["cap"]}")
            capacity_violation += (load - APs[ap]["cap"]) * 110

    # print(f"Distancia total: {total_distance} e Penalização por capacidade: {capacity_violation}")
    return total_distance + capacity_violation


# Seleção por torneio
## Selecionamos aleatoriamente 5 indivíduos da população e verificamos o que , entre eles
## tem o menos valor de fitness e retornamos ele
def tournament_selection(population, fitnesses, k=5):
    selected = random.sample(list(zip(population, fitnesses)), k)
    return min(selected, key=lambda x: x[1])[0]


# Crossover de um ponto
## Realiza o crossover entre duas soluções possíveis
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1) # ponto de divisão (aleatório)
    child1 = parent1[:point] + parent2[point:] # pega do index 0 até o ponto de divisão de um pai e junta com o ponto de divisão até o fim da mãe
    child2 = parent2[:point] + parent1[point:] # pega do index 0 até o ponto de divisão de uma mãe e junta com o ponto de divisão até o fim do pai
    return child1, child2


# Mutação
## Pega uma solução possível e calcula um possível mutação
def mutate(solution, mutation_rate=0.1):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = random.choice(list(APs.keys())) # se cair na probabilidade mutation_rate , seleciona um novo valor aleatório de AP
    return solution


# Algoritmo genético
## Calcula inicialmente uma população
## Usamos uma geração de 200 (200 filhos) e para cada um deles:
### Calculamos o fitness da população da geração atual
### Fazemos os cruzamentos de metade da população
### Realizamos a mutação dos genes
### Atualizamos a população para a população atual (passamos entre gerações)
## No fim pegamos a geração mais recente , calculamos o fitness de toda a geração e escolhemos aquele com o menor fitness
def genetic_algorithm(clients, pop_size=100, generations=200, mutation_rate=0.1):
    population = initialize_population(len(clients), pop_size)

    #for i in population:
        #print(i)

    for generation in range(generations):
        fitnesses = [evaluate_fitness(individual, clients) for individual in population]
        next_population = []

        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))

        population = next_population

    # Melhor solução
    fitnesses = [evaluate_fitness(individual, clients) for individual in population]
    best_solution = population[np.argmin(fitnesses)]
    return best_solution, min(fitnesses)


# Salvar resultados em um arquivo CSV
def save_results_to_csv(clients, solution, output_path):
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Cliente", "AP Conectado", "X", "Y"])
        for i, ap in enumerate(solution):
            writer.writerow([f"{i + 1}", ap, clients[i][0], clients[i][1]])



input_file_path = "clientes.csv"
output_file_path = "alocacao_clientes.csv"

clients = load_clients(input_file_path)
best_solution, best_fitness = genetic_algorithm(clients)

print("Melhor solução encontrada:", best_solution)
print("Fitness da solução:", best_fitness)

save_results_to_csv(clients, best_solution, output_file_path)
print(f"Resultados salvos no arquivo: {output_file_path}")

# make the data
result = load_result(output_file_path)
x = result["X"].to_numpy()
y = result["Y"].to_numpy()
colors = result["AP Conectado"].to_numpy()
# size and color:
#sizes = 
colors_dict = {
    "A" : "green",
    "B" : "yellow",
    "C" : "blue",
    "D" : "red"
}

new_colors = [colors_dict[char] for char in colors]

unique, counts = np.unique(colors, return_counts=True)
print(dict(zip(unique,counts)))
# plot
fig, ax = plt.subplots()
green_patch = mpatches.Patch(color = "green", label = "AP A")
yellow_patch = mpatches.Patch(color = "yellow", label = "AP B")
blue_patch = mpatches.Patch(color = "blue", label = "AP C")
red_patch = mpatches.Patch(color = "red", label = "AP D")


ax.scatter(x, y, s = 50, c = new_colors)

ax.set(xlim=(0, 80), ylim=(0, 80))


fig.legend(loc = "outside lower left", handles=[green_patch])
fig.legend(loc = "outside lower right", handles=[yellow_patch])
fig.legend(loc = "outside upper left", handles=[blue_patch])
fig.legend(loc = "outside upper right", handles=[red_patch])

plt.show()