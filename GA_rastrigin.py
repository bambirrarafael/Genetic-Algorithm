# GA solução de Rastrigin
# Rafael Bambirra Pereira

from numpy.random import randint as randint
from numpy.random import uniform as randfloat
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as deep_copy

def rafael_bambirra_rastrigin(nvar, ncal):

    class Individuo:
        def __init__(self):
            self.fitness = 0
            self.variaveis = []
            self.n_var = 0
            self.tamanho_fenotipo_por_var = 1
            self.tipo_funcao = Funcao()
            self.limite_inf = []
            self.limite_sup = []

        def calc_fitness(self):
            self.fitness = self.tipo_funcao.calc_resultado(self.variaveis)

        def inicializar_parametros(self, n_var, tipo_funcao):
            self.n_var = n_var                  # gambs
            self.tipo_funcao = tipo_funcao      # gambs
            self.variaveis = [None]*n_var
            self.limite_inf = tipo_funcao.limite_inf
            self.limite_sup = tipo_funcao.limite_sup


    class IndividuoBinario(Individuo):
        def __init__(self):
            Individuo.__init__(self)
            self.codigo_binario = []

        def criar_individuo(self):
            for i in range(self.n_var):
                for j in range(self.tamanho_fenotipo_por_var):
                    self.codigo_binario.append(str(randint(0, 2)))

        def decriptar(self):
            for i in range(0, self.n_var):
                variavel = self.codigo_binario[i:i + self.tamanho_fenotipo_por_var]
                variavel = int(''.join(variavel), 2) / (2 ** self.tamanho_fenotipo_por_var - 1)
                variavel = variavel*(self.limite_sup[i]-self.limite_inf[i])+self.limite_inf[i]
                self.variaveis[i] = variavel

        def calc_fitness(self):
            self.decriptar()
            Individuo.calc_fitness(self)

        def inicializar_parametros(self, n_var, tipo_funcao, n_bits_por_var=10):
            Individuo.inicializar_parametros(self, n_var, tipo_funcao)
            self.tamanho_fenotipo_por_var = n_bits_por_var

        def mutacao(self, nbit):
            self.codigo_binario[nbit] = str((int(self.codigo_binario[nbit])-1)**2)

        def cruzamento(self, indiviuo2):
            posicoes = []
            for i in range(self.n_var):
                posicoes.append(randint(0, self.tamanho_fenotipo_por_var) + i * self.tamanho_fenotipo_por_var)
            flag = False
            for i in range(len(self.codigo_binario)):
                if i == posicoes[i // self.tamanho_fenotipo_por_var]:
                    flag = not flag
                if flag:
                    aux1 = self.codigo_binario[i]
                    aux2 = indiviuo2.codigo_binario[i]
                    self.codigo_binario[i] = aux2
                    indiviuo2.codigo_binario[i] = aux1
            return posicoes


    class AlgoritmoGenetico:

        def __init__(self, nvar, ncal):
            # Atributos
            self.n_var = nvar
            self.n_cal = ncal
            self.pc = 0.7
            self.pm = 0.005
            self.n_pop = int(np.sqrt(self.n_cal))
            if self.n_pop > 200:
                self.n_pop = 200
            if self.n_pop % 2 == 1:
                self.n_pop = self.n_pop - 1
            self.n_iter = self.n_cal//self.n_pop
            self.populacao = []
            self.pais = []
            self.filhos = []
            self.problema = Rastringin(self.n_var)
            self.fitness = []
            self.variaveis = []             # gambs para plotar
            # Auxiliares
            self.media_fitness_pop = []
            self.fitness_plot = []
            self.geracoes = []
            # Inicialização a população
            for i in range(self.n_pop):
                ind = self.problema.tipo_individuo()
                ind.inicializar_parametros(n_var=self.n_var, tipo_funcao=self.problema)
                ind.criar_individuo()
                ind.calc_fitness()
                self.variaveis.append(ind.variaveis)
                self.fitness.append(ind.fitness)
                self.populacao.append(ind)
            # Definir Best em geração 0
            self.best = deep_copy(self.populacao[0])
            self.salvar_elite()
            # salvar valores para plotar
            self.media_fitness_pop.append(np.mean(self.fitness))
            self.fitness_plot.append(self.best.fitness)
            self.geracoes.append(0)
            # Evolução
            self.evoluir()

        def salvar_elite(self):
            for i in range(self.n_pop):
                if self.populacao[i].fitness < self.best.fitness:
                    self.best = deep_copy(self.populacao[i])

        def evoluir(self):
            for i in range(self.n_iter-1):
                self.pais = []
                self.filhos = []
                self.selecao()
                self.cruzamento()
                self.mutacao()
                self.calcular_fitness()
                self.salvar_elite()
                self.criar_nova_pop()
                #print(i)    # TODO RETIRAR PRINT PARA ENVIAR
                #print('fitness best: ' + str(self.best.fitness) + ' | fitness pop: ' + str(self.media_fitness_pop[i]))
                # salvar valores para plotar
                self.media_fitness_pop.append(np.mean(self.fitness))
                self.fitness_plot.append(self.best.fitness)
                self.geracoes.append(i)
                # terminar se chegar no 0
                if self.best.fitness == 0:
                    break

        def selecao(self):
            self.roleta_aux()
            for i in range(self.n_pop):
                if randfloat() > 0.5:
                    self.torneio()
                else:
                    self.roleta()

        def torneio(self):
            candidato_1 = self.populacao[randint(0, self.n_pop)]
            candidato_2 = self.populacao[randint(0, self.n_pop)]
            if candidato_1.fitness <= candidato_2.fitness:
                self.pais.append(deep_copy(candidato_1))
            else:
                self.pais.append(deep_copy(candidato_2))

        def roleta_aux(self):
            self.soma_roleta = 0
            for i in range(self.n_pop):
                self.soma_roleta += 1 / self.populacao[i].fitness

        def roleta(self):
            giro_da_roleta = randfloat(0, self.soma_roleta)
            aux = 0
            i = -1
            while aux < giro_da_roleta:
                aux += 1/self.populacao[i].fitness
                i += 1
            self.pais.append(deep_copy(self.populacao[i]))

        def cruzamento(self):
            for i in range(0, self.n_pop, 2):
                if randfloat() <= self.pc:
                    filho_1 = deep_copy(self.pais[i])
                    filho_2 = deep_copy(self.pais[i+1])
                    filho_1.cruzamento(filho_2)
                    self.filhos.append(filho_1)
                    self.filhos.append(filho_2)
                else:
                    self.filhos.append(deep_copy(self.pais[i]))
                    self.filhos.append(deep_copy(self.pais[i+1]))

        def mutacao(self):
            for i in range(self.n_pop):
                for j in range(self.populacao[0].tamanho_fenotipo_por_var*self.n_var):
                    if randfloat() < self.pm:
                        self.filhos[i].mutacao(j)

        def calcular_fitness(self):
            for i in range(self.n_pop):
                self.filhos[i].calc_fitness()
                self.fitness[i] = self.filhos[i].fitness

        def criar_nova_pop(self):
            self.populacao = deep_copy(self.filhos)
            self.populacao[np.argmax(self.fitness)] = self.best


    class Funcao:
        def __init__(self):
            self.resultado = ''
            self.variaveis = []

        def calc_resultado(self, variaveis):
            pass


    class Rastringin(Funcao):

        def __init__(self, n_var):
            Funcao.__init__(self)
            self.limite_inf = np.ones(n_var)*-5.12
            self.limite_sup = np.ones(n_var)*5.12

        def tipo_individuo(cls):
            ind = IndividuoBinario()
            return ind

        def calc_resultado(self, variaveis):                  # x é o vetor com todas as variáveis
            self.variaveis = variaveis
            n = len(variaveis)
            soma = 0.0
            for i in range(n):
                soma = soma + variaveis[i]**2 - 10*np.cos(2*np.pi*variaveis[i])
            self.resultado = 10*n + soma
            return self.resultado


    populacao = AlgoritmoGenetico(nvar, ncal)
    x = populacao.best.variaveis
    f = populacao.best.fitness
    return x, f

x, f = rafael_bambirra_rastrigin(nvar=10, ncal=10000)
'''print(x)
print(f)
plt.scatter(x=populacao.geracoes, y=populacao.media_fitness_pop, label = 'média da população')
plt.scatter(x=populacao.geracoes, y=populacao.fitness_plot, label = 'melhor indivíduo')
plt.xlabel('iteração')
plt.ylabel('aptidão')
plt.legend()
plt.show()'''
