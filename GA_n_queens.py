# Algoritmo genetico para solução do problema das N-Rainhas
# Rafael Bambirra Pereira

from numpy.random import randint as randint
from numpy.random import uniform as randfloat
import numpy as np
import pandas as pd
from scipy import stats
from scipy import special
import statistics as stc
import csv
import itertools
import matplotlib.pyplot as plt
from copy import deepcopy as deep_copy

def rafael_bambirra_rainhas(nvar, ncal):

    class AlgoritmoGenetico():

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
            self.n_iter = (self.n_cal//self.n_pop)-1
            self.populacao = []
            self.pais = []
            self.filhos = []
            self.fitness = []
            self.variaveis = []
            self.problema = N_rainhas(self.n_var)
            # Auxiliares
            self.media_fitness_pop = []
            self.fitness_plot = []
            self.geracoes = []
            self.permut  = list(itertools.permutations(range(self.n_var)))  # gambs para criar pop com permutação
            # Inicialização a população
            for i in range(self.n_pop):
                ind = self.problema.tipo_individuo()
                ind.inicializar_parametros(n_var=self.n_var, tipo_funcao=self.problema)
                ind.criar_individuo(permut = self.permut)   # comentar gambs para pop aleatória
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
                # print('fitness best: ' + str(self.best.fitness) + ' | fitness pop: ' + str(self.media_fitness_pop[i]))
                # salvar valores para plotar
                self.media_fitness_pop.append(np.mean(self.fitness))
                self.fitness_plot.append(self.best.fitness)
                self.geracoes.append(i)
                # terminar se chegar no 0
                if self.best.fitness == 0:
                    break

        def selecao(self):
            for i in range(self.n_pop):
                self.torneio()

        def torneio(self):
            candidato_1 = self.populacao[randint(0, self.n_pop)]
            candidato_2 = self.populacao[randint(0, self.n_pop)]
            if candidato_1.fitness <= candidato_2.fitness:
                self.pais.append(deep_copy(candidato_1))
            else:
                self.pais.append(deep_copy(candidato_2))

        def cruzamento(self):  # Ok
            for i in range(0, self.n_pop, 2):
                if randfloat() <= self.pc:
                    filho_1 = deep_copy(self.pais[i])
                    filho_2 = deep_copy(self.pais[i + 1])
                    filho_1.cruzamento(filho_2)
                    self.filhos.append(filho_1)
                    self.filhos.append(filho_2)
                else:
                    self.filhos.append(deep_copy(self.pais[i]))
                    self.filhos.append(deep_copy(self.pais[i + 1]))

        def mutacao(self):  # Ok
            for i in range(self.n_pop):
                for j in range(self.populacao[0].tamanho_fenotipo_por_var * self.n_var):
                    if randfloat() < self.pm:
                        self.filhos[i].mutacao(j)       # definir mutação no individuo

        def calcular_fitness(self):
            for i in range(self.n_pop):
                self.filhos[i].calc_fitness()

        def criar_nova_pop(self):
            self.populacao = deep_copy(self.filhos)
            self.populacao[np.argmax(self.fitness)] = self.best     # adiciona best em população
            n_eliminados = 2
            for i in range(n_eliminados):                   # cria novos individuos aleatórios
                self.populacao.append(self.populacao[randint(self.n_pop)])
            for i in range(n_eliminados):      # remover piores
                marca = np.argmax(self.fitness)
                self.populacao.pop(marca)


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

        def inicializar_parametros(self, n_var, tipo_funcao, n_bits_por_var=8):
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


    class IndividuoReal(Individuo):
        def __init__(self):
            Individuo.__init__(self)
            self.codigo_real = []

        def criar_individuo(self, permut):          # Com permutação de 8 - Ok
            numero = randint(0, len(permut))
            self.codigo_real = list(permut[numero])

        '''def criar_individuo(self):          # Aleatório - ok
            for i in range(self.n_var):
                self.codigo_real.append(randint(0, self.n_var))'''

        def inicializar_parametros(self, n_var, tipo_funcao, n_bits_por_var=1):
            Individuo.inicializar_parametros(self, n_var, tipo_funcao)
            self.tamanho_fenotipo_por_var = n_bits_por_var

        def calc_fitness(self):
            self.decriptar()
            Individuo.calc_fitness(self)

        def decriptar(self):
            self.variaveis = self.codigo_real       # gambs para aproveitar o código do rastringin

        '''def cruzamento(self, individuo2):  # PMX - deu errado
            selecao_1 = []
            selecao_2 = []
            fi_1 = list(np.zeros(self.n_var)-1)  # filho em mineires
            fi_2 = list(np.zeros(self.n_var)-1)
            pos_1 = randint(self.n_var)
            pos_2 = randint(self.n_var)
            for i in range(self.n_var):  # definir secão de corte
                flag = False
                if min(pos_1, pos_2) <= i <= max(pos_1, pos_2):
                    flag = not flag
                if flag:
                    selecao_1.append(self.codigo_real[i])
                    selecao_2.append(individuo2.codigo_real[i])
            for i in range(len(selecao_1)):  # Adicionar seleção nos filhos
                fi_1[min(pos_1, pos_2)+i] = selecao_1[i]
                fi_2[min(pos_1, pos_2)+i] = selecao_2[i]
            for i in range(len(selecao_2)):  # vasculhar na outra seleção os valores que ainda não estão no filho
                flag = False
                tem = selecao_2[i] in selecao_1
                if tem == False:  # aloca os valores que ainda não tem nas possições correspondentes
                    k = i
                    while flag == False:
                        marca = individuo2.codigo_real.index(selecao_1[k])
                        k = marca
                        if marca < min(pos_1, pos_2) or marca > max(pos_1, pos_2):    # (se a posição é valida, então ...)
                            fi_1[marca] = selecao_2[i]
                            flag = True
    #                    if len(selecao_1) < marca:
    #                        marca = individuo2.codigo_real.index(self.codigo_real[k + min(pos_2, pos_1)])
    #                        fi_1[marca] = selecao_2[i]
    #                        flag = True
            for i in range(len(selecao_1)):  # tudo de novo para o filho 2 - vasculhar na outra seleção os valores que...
                flag = False
                tem = selecao_1[i] in selecao_2
                if tem == False:  # aloca os valores que ainda não tem nas possições correspondentes
                    k = i
                    while flag == False:
                        marca = self.codigo_real.index(selecao_2[k])
                        k = marca
                        if marca < min(pos_1, pos_2) or marca > max(pos_1, pos_2):   # não pode escolher dentro da seleção
                            fi_2[marca] = selecao_1[i]
                            flag = True
     #                   if len(selecao_2) < marca:
     #                       marca = self.codigo_real.index(individuo2.codigo_real[k + min(pos_2, pos_1)])
     #                       fi_2[marca] = selecao_1[i]
     #                       flag = True
            for i in range(self.n_var):  # completar as variaveis não preenchidas nos filhos
                if fi_1[i] < 0:  # primeiro pro filho 1
                    flag = False
                    k = i
                    while flag == False:
                        tem = individuo2.codigo_real[k] in fi_1
                        if tem == False:
                            fi_1[i] = individuo2.codigo_real[k]
                            flag = True
                        else:
                            k = k + 1
                            if k > self.n_var:
                                k = 0
                if fi_2[i] < 0:  # depois pro 2
                    flag = False
                    k = i
                    while flag == False:
                        tem = self.codigo_real[k] in fi_2
                        if tem == False:
                            fi_2[i] = self.codigo_real[k]
                            flag = True
                        else:
                            k = k + 1
                            if k > self.n_var:
                                k = 0'''

        def cruzamento(self, individuo2):  # codigo cut_and_fill
            posit = randint(self.n_var)
            fi_1 = []  # filho em mineires
            fi_2 = []
            for i in range(posit):
                fi_1.append(self.codigo_real[i])
                fi_2.append(individuo2.codigo_real[i])
            flag = False
            for i in range(self.n_var):
                for j in range(len(fi_1)):
                    if individuo2.codigo_real[i] == fi_1[j]:
                        flag = True
                if flag == False:
                    fi_1.append(individuo2.codigo_real[i])
                flag = False
            flag = False
            for i in range(self.n_var):
                for j in range(len(fi_2)):
                    if self.codigo_real[i] == fi_2[j]:
                        flag = True
                if flag == False:
                    fi_2.append(self.codigo_real[i])
                flag = False
            erro = len(self.codigo_real) - len(fi_1)
            if erro > 0:   # (ajuste para população aleatória)
                for i in range(erro):
                    fi_1.append(individuo2.codigo_real[self.n_var - erro + i])
            if erro < 0:
                for i in range(self.n_var - 1 - erro, self.n_var - 1, -1):
                    fi_1.pop(i)
            erro = len(individuo2.codigo_real) - len(fi_2)
            if erro > 0:
                for i in range(erro):
                    fi_2.append(self.codigo_real[self.n_var - erro + i])
            if erro < 0:
                for i in range(self.n_var - 1 - erro, self.n_var - 1, -1):
                    fi_2.pop(i)
            self.codigo_real = fi_1
            individuo2.codigo_real = fi_2

        def mutacao(self, nbit):
            # trocar de posições
            troca = randint(0, self.n_var)
            if troca == nbit:
                while troca == nbit:
                    troca = randint(0, self.n_var)
            aux = self.codigo_real[troca]
            self.codigo_real[troca] = self.codigo_real[nbit]
            self.codigo_real[nbit] = aux


    class Funcao:
        def __init__(self):
            self.f = ''
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
            self.f = 10 * n + soma
            return self.f


    class N_rainhas(Funcao):

        def __init__(self, n_var):
            Funcao.__init__(self)
            self.limite_inf = 0
            self.limite_sup = n_var

        def tipo_individuo(cls):
            ind = IndividuoReal()
            return ind

        def calc_resultado(self, variaveis): #TODO
            self.variaveis = variaveis
            f = 0
            n = len(self.variaveis)
            for i in range(len(self.variaveis)):
                for j in range(len(self.variaveis)):
                    if abs(i-j) == abs(float(self.variaveis[i])-float(self.variaveis[j])) and i != j:
                        f = f + 1
                    if self.variaveis[i] == self.variaveis[j] and i != j:
                        f = f + 1
            f = f / 2
            return f

    populacao = AlgoritmoGenetico(nvar, ncal)
    x = populacao.best.variaveis
    f = populacao.best.fitness
    return x, f

x,f = rafael_bambirra_rainhas(nvar=8, ncal=10000)
#n_ger = []
#for i in range(2000):

    #print(x)    # TODO RETIRAR PRINT PARA ENVIAR
    #print(f)    # TODO RETIRAR PRINT PARA ENVIAR
    #n_ger.append(max(populacao.geracoes))
    #print(i)
#plt.scatter(x=populacao.geracoes, y=populacao.media_fitness_pop)
#plt.scatter(x=populacao.geracoes, y=populacao.fitness_plot)
#plt.show()
'''plt.hist(n_ger, bins= 100)
plt.xlabel('número de gerações - criação permutação')
plt.ylabel('frequencia')
plt.show()
print('média = ' + str(stc.mean(n_ger)))
print('mediana = ' + str(stc.median(n_ger)))
print('desvio padrão = ' + str(stc.pstdev(n_ger)))
print('moda = ' + str(stc.mode(n_ger)))'''
#csvFile = open('dados_permutacao.csv','w')
#csvFile.write(str(n_ger))
#csvFile.close()

# teste de hipoteses
# geração aleatória:
'''
media_r = 11.22
mediana_r = 7.0
desv_pad_r = 17.57
moda_r = 6

# Geração com permutação:

media_p = 6.62
mediana_p = 3.0
desv_pad_p = 15.91
moda = 0


#hist_r = pd.read_csv('./data/rainhas_dados_aleatorio.csv', delimiter = ',', header=None)
#hist_p = pd.read_csv('./data/rainhas_dados_permutacao.csv', delimiter = ',', header=None)
hist_r = [6, 8, 3, 6, 98, 7, 13, 6, 3, 10, 4, 20, 4, 6, 4, 20, 9, 7, 3, 42, 4, 7, 5, 21, 42, 2, 9, 22, 12, 9, 8, 38, 8, 8, 7, 2, 12, 42, 4, 4, 4, 8, 13, 9, 4, 3, 7, 9, 4, 6, 4, 25, 6, 8, 4, 4, 5, 16, 6, 47, 31, 4, 4, 2, 5, 2, 4, 4, 6, 43, 9, 4, 5, 6, 5, 10, 5, 8, 6, 7, 6, 10, 6, 5, 6, 6, 4, 5, 3, 2, 3, 3, 6, 7, 6, 5, 13, 98, 5, 9, 12, 7, 3, 5, 9, 10, 6, 4, 3, 1, 14, 6, 2, 7, 4, 15, 7, 5, 4, 4, 4, 8, 8, 87, 8, 5, 4, 20, 17, 7, 6, 4, 5, 2, 7, 5, 17, 9, 4, 4, 5, 2, 5, 3, 3, 10, 4, 10, 2, 10, 2, 5, 8, 5, 8, 6, 7, 3, 4, 6, 7, 5, 17, 9, 8, 10, 3, 4, 8, 4, 7, 11, 6, 7, 16, 10, 5, 4, 4, 6, 9, 18, 98, 1, 4, 26, 6, 4, 8, 8, 6, 5, 4, 6, 4, 13, 4, 11, 4, 6, 9, 16, 11, 3, 6, 14, 8, 2, 6, 6, 5, 5, 68, 10, 40, 19, 7, 12, 5, 17, 2, 91, 8, 8, 6, 8, 35, 2, 2, 6, 10, 98, 6, 4, 6, 10, 8, 4, 4, 3, 7, 4, 8, 27, 12, 5, 6, 5, 5, 12, 2, 6, 13, 6, 3, 98, 5, 7, 5, 14, 4, 10, 5, 5, 6, 15, 10, 5, 5, 8, 2, 3, 32, 2, 6, 6, 7, 7, 10, 6, 5, 4, 9, 5, 98, 3, 9, 4, 98, 12, 7, 2, 6, 18, 7, 3, 2, 4, 98, 7, 5, 15, 6, 5, 7, 5, 7, 8, 7, 8, 98, 7, 4, 4, 8, 5, 9, 9, 19, 9, 11, 2, 6, 45, 10, 8, 2, 5, 5, 9, 5, 6, 4, 6, 6, 7, 6, 3, 6, 5, 3, 15, 8, 10, 3, 4, 30, 30, 8, 4, 3, 98, 3, 4, 5, 10, 12, 11, 6, 10, 3, 4, 8, 5, 7, 51, 10, 6, 10, 9, 2, 8, 17, 74, 11, 11, 11, 6, 8, 3, 7, 8, 5, 98, 8, 5, 4, 7, 7, 10, 2, 8, 0, 3, 5, 10, 11, 6, 11, 9, 39, 4, 5, 3, 3, 5, 8, 5, 4, 10, 7, 6, 20, 8, 4, 3, 98, 4, 7, 6, 33, 2, 8, 5, 11, 8, 4, 5, 9, 6, 7, 6, 7, 20, 11, 8, 98, 11, 5, 10, 5, 4, 4, 5, 4, 6, 12, 1, 10, 4, 11, 4, 72, 98, 4, 77, 9, 8, 8, 11, 12, 6, 6, 4, 6, 5, 6, 7, 5, 8, 10, 7, 8, 98, 5, 3, 9, 7, 11, 3, 6, 3, 82, 12, 7, 5, 3, 98, 7, 4, 6, 7, 7, 7, 6, 10, 9, 98, 2, 11, 13, 9, 3, 8, 6, 6, 5, 10, 5, 5, 42, 4, 6, 98, 8, 6, 3, 9, 12, 6, 4, 9, 6, 8, 6, 10, 8, 9, 3, 26, 16, 8, 6, 4, 4, 4, 8, 3, 10, 6, 8, 5, 5, 6, 6, 14, 14, 98, 38, 9, 10, 7, 10, 2, 12, 5, 46, 4, 3, 4, 6, 9, 5, 16, 10, 4, 3, 7, 12, 8, 9, 10, 5, 3, 5, 7, 14, 6, 9, 7, 7, 4, 5, 10, 3, 10, 7, 3, 7, 4, 5, 6, 5, 9, 6, 3, 8, 3, 1, 10, 10, 5, 4, 7, 17, 4, 2, 14, 3, 8, 4, 4, 11, 5, 17, 26, 3, 8, 5, 4, 7, 18, 9, 7, 4, 6, 98, 9, 6, 9, 4, 4, 11, 2, 4, 6, 13, 5, 3, 4, 9, 13, 10, 6, 5, 6, 4, 8, 6, 10, 10, 5, 5, 5, 6, 7, 6, 8, 5, 12, 12, 18, 7, 6, 5, 6, 3, 26, 4, 6, 5, 6, 3, 7, 5, 4, 12, 4, 3, 6, 5, 3, 6, 11, 3, 5, 6, 3, 5, 12, 7, 4, 7, 3, 6, 12, 98, 4, 5, 4, 7, 6, 5, 7, 2, 13, 6, 11, 7, 3, 13, 6, 4, 8, 4, 5, 13, 7, 5, 6, 7, 6, 6, 8, 2, 11, 12, 5, 8, 7, 10, 6, 4, 7, 9, 14, 8, 37, 8, 98, 4, 2, 43, 4, 2, 19, 8, 7, 6, 7, 8, 8, 7, 5, 11, 10, 8, 9, 11, 9, 6, 4, 7, 12, 98, 8, 6, 8, 12, 7, 6, 5, 7, 98, 4, 13, 3, 5, 7, 12, 7, 6, 15, 4, 8, 6, 12, 5, 20, 7, 8, 3, 8, 7, 10, 5, 8, 7, 98, 7, 6, 9, 19, 6, 17, 8, 2, 8, 98, 9, 18, 5, 5, 10, 8, 3, 4, 6, 6, 8, 5, 6, 6, 5, 37, 3, 12, 11, 5, 6, 9, 4, 14, 12, 6, 6, 15, 3, 6, 98, 7, 5, 3, 13, 5, 7, 6, 7, 3, 6, 6, 7, 14, 9, 3, 6, 3, 5, 7, 8, 7, 9, 3, 22, 9, 14, 11, 20, 7, 5, 10, 9, 6, 9, 4, 6, 7, 4, 3, 5, 5, 14, 9, 4, 4, 8, 7, 3, 7, 7, 16, 8, 11, 26, 9, 8, 9, 22, 8, 4, 4, 4, 3, 6, 5, 7, 3, 1, 9, 14, 16, 8, 11, 6, 4, 12, 7, 9, 6, 6, 4, 6, 8, 5, 15, 7, 5, 7, 4, 3, 7, 8, 5, 16, 5, 5, 98, 8, 17, 12, 8, 4, 4, 4, 13, 10, 11, 3, 4, 16, 5, 12, 15, 5, 5, 13, 8, 8, 27, 98, 12, 12, 14, 7, 33, 8, 6, 11, 5, 7, 3, 8, 9, 11, 7, 8, 3, 5, 10, 5, 8, 12, 10, 6, 9, 3, 5, 5, 4, 12, 6, 6, 2, 9, 6, 12, 2, 98, 10, 9, 2, 3, 4, 7, 9, 6, 15, 7, 11, 12, 6, 98, 6, 6, 6, 6, 27, 4, 4, 4, 8, 7, 27, 4, 10, 8, 9, 14, 12, 6, 11, 6, 3, 11, 3, 10, 8, 6, 7, 4, 6, 17, 11, 7, 10, 11, 9, 4, 8, 24, 4, 4, 7, 7, 6, 6, 5, 6, 3, 3, 9, 94, 5, 4, 6, 10, 24, 9, 4, 8, 4, 69, 11, 7, 4, 10, 9, 10, 11, 13, 12, 3, 5, 7, 9, 3, 12, 6, 6, 11, 98, 7, 6, 28, 7, 9, 7, 15, 3, 4, 6, 4, 7, 5, 12, 7, 7, 2, 5, 8, 6, 5, 9, 19, 7, 7, 4, 9, 7, 6, 7, 7, 5, 5, 6, 5, 98, 11, 13, 2, 4, 30, 11, 4, 6, 49, 12, 9, 98, 4, 11, 4, 26, 4, 9, 7, 9, 2, 4, 4, 4, 13, 4, 72, 6, 6, 6, 11, 11, 10, 98, 9, 98, 10, 3, 5, 6, 5, 3, 8, 7, 4, 4, 9, 6, 4, 55, 10, 3, 10, 8, 5, 5, 7, 7, 9, 7, 5, 9, 10, 9, 4, 11, 5, 5, 8, 11, 7, 8, 9, 6, 6, 4, 15, 3, 8, 7, 42, 5, 5, 73, 7, 5, 4, 3, 6, 5, 3, 13, 29, 60, 4, 8, 9, 5, 11, 9, 10, 60, 8, 5, 7, 3, 5, 8, 11, 5, 12, 13, 6, 28, 5, 3, 16, 10, 7, 8, 84, 5, 5, 4, 37, 7, 12, 7, 2, 7, 2, 6, 10, 7, 8, 9, 5, 7, 11, 3, 7, 6, 8, 4, 4, 18, 13, 15, 6, 7, 8, 6, 3, 3, 9, 4, 5, 6, 10, 2, 8, 3, 6, 4, 11, 21, 4, 5, 4, 98, 8, 8, 4, 7, 6, 4, 12, 9, 8, 2, 6, 4, 19, 8, 12, 4, 7, 5, 12, 10, 9, 6, 3, 6, 5, 7, 7, 7, 3, 3, 2, 8, 12, 5, 4, 4, 11, 5, 9, 9, 12, 6, 9, 5, 3, 8, 6, 6, 6, 4, 13, 6, 5, 9, 4, 7, 13, 7, 10, 4, 10, 34, 6, 5, 7, 6, 14, 7, 9, 6, 6, 7, 3, 5, 5, 6, 11, 6, 7, 19, 6, 5, 14, 5, 2, 9, 8, 5, 23, 2, 34, 6, 8, 9, 3, 9, 4, 4, 7, 6, 5, 7, 23, 98, 4, 4, 5, 5, 9, 8, 8, 5, 7, 6, 9, 7, 9, 3, 9, 4, 98, 5, 4, 4, 6, 19, 28, 7, 6, 4, 5, 26, 6, 6, 16, 4, 5, 6, 19, 6, 15, 5, 1, 8, 9, 5, 14, 5, 4, 9, 5, 7, 98, 6, 4, 6, 6, 10, 8, 5, 9, 4, 3, 8, 11, 5, 5, 8, 4, 5, 11, 5, 7, 5, 13, 4, 13, 4, 2, 13, 4, 10, 6, 19, 6, 8, 4, 5, 5, 6, 6, 28, 8, 6, 5, 6, 7, 10, 5, 8, 5, 5, 7, 8, 9, 33, 8, 6, 5, 9, 8, 98, 6, 9, 3, 10, 4, 4, 10, 2, 10, 8, 12, 6, 7, 12, 3, 5, 7, 6, 11, 11, 2, 3, 5, 4, 8, 8, 8, 13, 7, 6, 8, 19, 8, 16, 10, 7, 63, 5, 98, 4, 2, 6, 4, 6, 7, 2, 9, 9, 5, 3, 4, 6, 9, 98, 9, 4, 5, 7, 3, 4, 8, 8, 2, 5, 3, 9, 7, 6, 8, 2, 7, 13, 8, 6, 7, 8, 9, 6, 7, 4, 9, 3, 9, 9, 98, 4, 34, 2, 4, 6, 7, 24, 7, 2, 5, 5, 12, 4, 6, 3, 5, 7, 8, 4, 22, 4, 8, 6, 6, 19, 1, 4, 8, 3, 10, 23, 6, 11, 6, 9, 10, 2, 2, 4, 4, 3, 7, 5, 8, 6, 6, 51, 98, 17, 7, 10, 9, 5, 98, 21, 5, 98, 3, 8, 3, 7, 98, 3, 3, 6, 3, 8, 7, 12, 3, 6, 3, 8, 5, 5, 3, 13, 9, 5, 3, 6, 5, 6, 7, 6, 4, 3, 16, 5, 5, 4, 4, 10, 6, 13, 32, 6, 3, 12, 5, 8, 12, 4, 2, 7, 8, 8, 10, 20, 5, 10, 4, 5, 7, 3, 98, 7, 4, 9, 13, 1, 6, 22, 1, 6, 6, 98, 5, 14, 3, 98, 2, 6, 6, 98, 10, 9, 5, 6, 4, 6, 7, 12, 3, 14, 9, 9, 4, 5, 4, 13, 22, 12, 9, 10, 8, 12, 4, 10, 6, 9, 8, 3, 5, 3, 4, 3, 3, 10, 6, 5, 8, 18, 12, 20, 21, 12, 7, 3, 20, 6, 98, 3, 5, 3, 6, 98, 6, 12, 8, 6, 22, 11, 7, 98, 14, 9, 10, 6, 6, 9, 8, 6, 9, 10, 36, 6, 5, 8, 4, 10, 3, 6, 6, 6, 5, 5, 8, 6, 13, 4, 4, 7, 6, 17, 7, 7, 7, 9, 7, 5, 9, 6, 3, 12, 3, 5, 8, 41, 1, 9, 5, 2, 4, 12, 6, 17, 6, 5, 11, 7, 8, 3, 11, 3, 7, 4, 11, 7, 4, 4, 6, 98, 6, 4, 4, 2, 6, 6, 8, 8, 7, 3, 3, 2, 4, 7, 7, 16, 3, 7, 21, 7, 3, 3, 15, 5, 12, 16, 7, 49, 15, 3, 16, 3, 17, 8, 9, 8, 4, 7, 98, 8, 5, 6, 5, 8, 4, 2, 7, 6, 3, 1, 6, 4, 5, 5, 8, 5, 8, 3, 6, 2, 7, 4, 17, 8, 9, 6, 5, 12, 10, 4, 7, 6, 8, 7, 7, 3, 7, 7, 98, 98, 8, 5, 2, 3, 3, 5, 8, 5, 2, 9, 5, 4, 4, 8, 8, 7, 4, 98, 18, 4, 5, 6, 5, 4, 9, 10, 12, 13, 4, 6, 9, 8, 3, 7, 22, 6, 4, 4, 4, 16, 7, 17, 16, 8, 11, 14, 4, 9, 10, 36, 11, 12, 7, 5, 5, 98, 10, 7, 2, 3, 17, 54, 7]
hist_p = [4, 1, 1, 10, 4, 5, 4, 1, 1, 0, 0, 5, 2, 0, 5, 0, 57, 0, 1, 5, 3, 2, 3, 1, 98, 1, 0, 11, 9, 2, 6, 0, 3, 22, 0, 0, 12, 3, 3, 0, 4, 2, 2, 2, 6, 1, 4, 5, 0, 0, 5, 11, 5, 0, 1, 2, 3, 2, 9, 1, 5, 10, 0, 14, 6, 2, 6, 0, 1, 7, 2, 5, 5, 14, 3, 88, 1, 9, 2, 4, 0, 0, 4, 12, 8, 9, 13, 0, 9, 3, 10, 3, 1, 5, 0, 7, 0, 2, 0, 0, 9, 4, 0, 4, 4, 0, 5, 3, 4, 4, 7, 6, 3, 3, 0, 98, 4, 1, 1, 7, 1, 1, 10, 7, 6, 5, 3, 5, 3, 4, 1, 8, 1, 5, 11, 27, 4, 0, 4, 5, 4, 1, 8, 0, 14, 1, 4, 3, 0, 5, 5, 6, 12, 4, 1, 0, 5, 1, 2, 7, 15, 4, 6, 41, 3, 18, 4, 2, 3, 0, 6, 7, 5, 1, 3, 1, 0, 7, 8, 4, 2, 4, 7, 1, 8, 0, 3, 2, 5, 5, 6, 0, 0, 8, 0, 1, 14, 15, 0, 0, 1, 2, 2, 0, 12, 6, 97, 4, 98, 4, 8, 8, 5, 2, 12, 5, 8, 3, 2, 8, 1, 5, 5, 5, 5, 2, 1, 2, 13, 0, 12, 4, 0, 4, 43, 2, 3, 3, 6, 1, 5, 2, 6, 36, 0, 7, 0, 4, 0, 13, 0, 2, 2, 0, 4, 2, 4, 9, 7, 1, 3, 0, 9, 12, 9, 0, 10, 13, 2, 1, 1, 11, 1, 6, 4, 0, 0, 6, 2, 2, 3, 4, 0, 3, 4, 0, 10, 4, 5, 4, 0, 3, 2, 1, 3, 2, 3, 2, 7, 98, 0, 1, 0, 11, 0, 11, 11, 5, 0, 4, 0, 0, 1, 9, 6, 14, 1, 1, 5, 1, 3, 0, 0, 3, 0, 5, 5, 0, 98, 0, 4, 1, 98, 0, 3, 2, 98, 11, 47, 8, 2, 0, 11, 3, 3, 2, 2, 2, 1, 4, 2, 3, 0, 98, 4, 1, 4, 2, 5, 2, 38, 3, 4, 2, 4, 7, 6, 3, 2, 2, 3, 5, 98, 1, 0, 1, 0, 6, 2, 3, 6, 3, 0, 2, 5, 3, 3, 5, 3, 5, 8, 6, 2, 1, 10, 1, 28, 33, 0, 0, 5, 5, 2, 3, 4, 3, 0, 5, 0, 98, 6, 8, 2, 7, 26, 4, 3, 6, 4, 4, 3, 0, 5, 0, 2, 0, 1, 6, 1, 3, 3, 1, 0, 0, 32, 1, 2, 0, 1, 3, 3, 3, 7, 1, 0, 0, 8, 10, 6, 4, 2, 98, 3, 0, 0, 2, 6, 6, 3, 9, 5, 4, 5, 3, 2, 3, 1, 57, 3, 19, 1, 0, 2, 2, 16, 0, 17, 8, 0, 10, 0, 5, 2, 4, 2, 3, 2, 0, 0, 0, 5, 4, 6, 0, 2, 3, 3, 5, 2, 1, 6, 6, 0, 5, 4, 4, 0, 3, 0, 9, 7, 4, 2, 6, 1, 3, 98, 9, 0, 1, 0, 0, 4, 5, 2, 2, 0, 6, 6, 5, 5, 0, 0, 3, 8, 1, 5, 3, 16, 0, 1, 0, 0, 10, 0, 4, 4, 3, 4, 0, 6, 0, 3, 9, 4, 0, 5, 6, 5, 3, 1, 9, 8, 0, 0, 4, 8, 2, 2, 98, 1, 8, 75, 4, 1, 2, 21, 9, 6, 9, 2, 6, 0, 0, 2, 1, 4, 1, 2, 5, 1, 98, 5, 0, 9, 78, 92, 5, 5, 1, 0, 2, 23, 12, 11, 0, 1, 3, 2, 98, 0, 4, 3, 11, 3, 0, 1, 3, 3, 3, 0, 2, 1, 14, 5, 7, 0, 4, 2, 6, 2, 1, 5, 4, 5, 3, 0, 98, 4, 1, 2, 0, 4, 53, 4, 5, 4, 3, 6, 2, 1, 15, 6, 4, 0, 5, 8, 10, 4, 2, 6, 4, 4, 0, 4, 0, 3, 5, 4, 3, 0, 1, 3, 4, 2, 5, 0, 1, 4, 1, 14, 0, 2, 4, 1, 7, 4, 1, 1, 4, 3, 4, 0, 6, 0, 10, 3, 3, 2, 1, 2, 1, 1, 0, 1, 7, 14, 5, 0, 5, 63, 98, 5, 6, 3, 98, 0, 2, 3, 6, 5, 1, 4, 7, 0, 1, 3, 0, 3, 8, 0, 5, 0, 22, 4, 3, 2, 0, 98, 9, 4, 4, 3, 5, 2, 4, 6, 10, 8, 6, 4, 40, 3, 0, 1, 0, 2, 1, 1, 2, 5, 1, 1, 8, 4, 6, 10, 0, 3, 6, 61, 3, 7, 5, 0, 6, 6, 3, 13, 5, 1, 3, 1, 8, 4, 0, 3, 3, 0, 0, 9, 10, 4, 5, 1, 0, 6, 0, 3, 2, 1, 23, 0, 3, 17, 0, 5, 2, 11, 9, 4, 0, 3, 3, 0, 9, 24, 5, 0, 8, 1, 1, 7, 8, 2, 3, 4, 2, 1, 0, 4, 12, 6, 4, 0, 0, 3, 7, 5, 0, 2, 3, 5, 5, 0, 6, 9, 4, 3, 6, 4, 5, 0, 0, 0, 0, 2, 0, 6, 2, 11, 5, 4, 7, 12, 1, 4, 0, 1, 0, 0, 5, 0, 1, 2, 0, 98, 0, 4, 9, 2, 0, 2, 5, 7, 2, 0, 6, 3, 3, 3, 0, 5, 3, 0, 4, 4, 0, 3, 0, 3, 3, 8, 52, 3, 49, 0, 1, 6, 8, 98, 4, 3, 2, 3, 6, 0, 24, 19, 5, 1, 4, 0, 1, 3, 0, 3, 2, 98, 4, 5, 5, 3, 8, 5, 5, 2, 4, 0, 5, 1, 0, 6, 0, 1, 0, 10, 4, 9, 17, 6, 2, 0, 0, 1, 0, 8, 1, 0, 2, 7, 1, 0, 5, 1, 2, 1, 0, 3, 9, 0, 1, 98, 16, 1, 2, 1, 5, 3, 8, 10, 2, 3, 4, 2, 0, 98, 0, 6, 1, 6, 2, 2, 4, 7, 1, 8, 4, 15, 98, 2, 3, 6, 2, 5, 3, 2, 10, 6, 2, 3, 6, 2, 7, 5, 0, 2, 7, 3, 6, 3, 0, 1, 4, 7, 4, 6, 0, 3, 2, 6, 2, 8, 1, 4, 22, 8, 4, 3, 0, 0, 6, 2, 5, 5, 1, 6, 0, 0, 28, 0, 0, 6, 5, 2, 0, 1, 0, 0, 1, 5, 0, 2, 0, 2, 1, 4, 1, 1, 0, 5, 0, 1, 0, 0, 4, 7, 4, 3, 6, 1, 4, 0, 0, 0, 0, 50, 2, 1, 1, 4, 2, 3, 6, 0, 2, 6, 4, 17, 2, 6, 5, 6, 7, 7, 15, 2, 3, 8, 3, 9, 4, 2, 4, 3, 4, 6, 4, 3, 3, 5, 8, 3, 5, 7, 2, 3, 0, 15, 3, 5, 6, 0, 2, 1, 3, 4, 0, 98, 9, 13, 0, 10, 3, 1, 3, 1, 2, 4, 2, 4, 2, 0, 2, 0, 0, 10, 1, 5, 5, 0, 1, 0, 3, 0, 3, 8, 3, 0, 1, 1, 5, 4, 3, 4, 4, 5, 2, 2, 6, 2, 6, 0, 1, 3, 6, 4, 4, 0, 4, 1, 6, 7, 2, 12, 11, 3, 5, 7, 4, 4, 2, 7, 51, 3, 4, 4, 1, 1, 98, 3, 6, 0, 5, 1, 4, 2, 3, 16, 0, 2, 0, 11, 7, 0, 5, 1, 2, 1, 5, 2, 6, 1, 20, 4, 8, 1, 0, 2, 3, 2, 2, 7, 7, 4, 3, 3, 3, 1, 1, 5, 6, 0, 5, 1, 8, 0, 0, 5, 4, 3, 3, 6, 0, 1, 0, 4, 4, 1, 2, 1, 3, 1, 2, 3, 0, 3, 5, 4, 8, 7, 0, 5, 6, 0, 2, 8, 4, 6, 0, 1, 3, 15, 2, 2, 0, 4, 2, 1, 9, 5, 0, 2, 4, 0, 4, 3, 2, 0, 8, 1, 2, 0, 4, 3, 0, 4, 11, 0, 0, 5, 98, 1, 3, 10, 0, 98, 11, 4, 3, 0, 0, 7, 3, 2, 6, 5, 0, 0, 5, 0, 2, 3, 4, 3, 0, 2, 3, 2, 3, 4, 2, 3, 7, 6, 0, 7, 6, 98, 0, 0, 6, 19, 2, 4, 6, 5, 3, 4, 2, 11, 21, 0, 6, 16, 2, 2, 2, 2, 9, 1, 0, 0, 3, 1, 6, 4, 5, 1, 7, 3, 13, 2, 1, 2, 0, 3, 2, 1, 98, 1, 9, 3, 1, 2, 2, 8, 8, 1, 2, 2, 1, 5, 2, 4, 0, 4, 0, 5, 4, 98, 3, 2, 12, 3, 7, 6, 13, 2, 1, 2, 6, 2, 4, 5, 7, 15, 1, 3, 11, 2, 5, 1, 0, 1, 6, 6, 5, 2, 0, 0, 2, 7, 0, 9, 5, 0, 0, 5, 1, 2, 0, 0, 2, 0, 7, 3, 0, 1, 4, 0, 16, 0, 3, 2, 5, 8, 5, 3, 11, 0, 0, 1, 7, 98, 0, 5, 0, 7, 2, 6, 5, 0, 5, 0, 0, 0, 5, 1, 2, 2, 3, 3, 0, 5, 0, 2, 0, 2, 7, 98, 2, 4, 2, 98, 5, 3, 3, 24, 3, 5, 0, 2, 4, 3, 9, 0, 3, 0, 0, 4, 3, 2, 1, 1, 5, 3, 0, 3, 4, 0, 3, 0, 1, 6, 0, 2, 98, 10, 98, 0, 1, 0, 0, 0, 4, 9, 0, 2, 9, 2, 6, 4, 0, 2, 4, 0, 0, 1, 0, 4, 2, 98, 0, 0, 4, 9, 0, 3, 0, 0, 1, 0, 16, 3, 0, 4, 60, 6, 7, 4, 9, 0, 1, 1, 0, 7, 5, 1, 6, 6, 4, 2, 0, 8, 3, 0, 8, 8, 7, 3, 98, 1, 4, 2, 9, 7, 3, 1, 5, 12, 0, 3, 5, 7, 0, 3, 3, 0, 2, 3, 0, 10, 0, 38, 4, 1, 5, 5, 4, 0, 4, 2, 8, 3, 0, 3, 3, 6, 7, 4, 0, 0, 1, 2, 98, 42, 98, 0, 1, 23, 2, 6, 7, 0, 19, 0, 4, 0, 1, 11, 3, 11, 0, 7, 4, 1, 6, 3, 1, 3, 0, 4, 6, 5, 4, 4, 0, 0, 0, 0, 23, 5, 6, 0, 6, 0, 4, 0, 1, 0, 8, 3, 9, 1, 1, 4, 0, 4, 0, 1, 4, 10, 1, 0, 5, 1, 6, 0, 4, 4, 6, 2, 10, 2, 1, 2, 7, 12, 3, 0, 0, 4, 7, 8, 0, 0, 3, 7, 3, 0, 5, 4, 4, 14, 10, 11, 8, 8, 4, 98, 1, 6, 4, 2, 0, 4, 2, 6, 3, 2, 2, 3, 3, 1, 0, 0, 4, 0, 98, 2, 4, 6, 0, 2, 3, 0, 3, 6, 3, 9, 6, 0, 2, 4, 7, 0, 3, 2, 0, 1, 0, 3, 5, 0, 1, 0, 3, 2, 0, 12, 6, 0, 2, 3, 3, 0, 3, 6, 0, 4, 0, 8, 4, 13, 1, 4, 0, 1, 6, 7, 4, 1, 0, 98, 98, 6, 3, 1, 1, 1, 8, 3, 2, 4, 2, 3, 3, 1, 1, 7, 4, 1, 0, 1, 4, 0, 0, 4, 5, 4, 3, 3, 3, 4, 0, 3, 1, 5, 0, 2, 4, 0, 13, 16, 6, 11, 4, 6, 2, 1, 3, 1, 6, 2, 0, 1, 3, 0, 3, 1, 5, 0, 3, 4, 2, 8, 4, 5, 5, 4, 2, 5, 3, 0, 0, 1, 3, 2, 0, 2, 2, 0, 8, 5, 1, 1, 1, 14, 19, 3, 3, 4, 1, 8, 5, 2, 1, 14, 1, 7, 3, 1, 0, 1, 0, 3, 3, 8, 0, 3, 0, 0, 0, 7, 0, 6, 11, 1, 2, 6, 23, 3, 2, 1, 8, 50, 3, 7, 4, 1, 2, 0, 1, 17, 11, 6, 7, 3, 5, 4, 9, 1, 98, 3, 0, 8, 0, 1, 9, 1, 0, 11, 3, 5, 7, 6, 98, 8, 6, 2, 4, 22, 27, 3, 2, 0, 0, 6, 1, 5, 10, 1, 2, 5, 5, 2, 6, 0, 0]
hist_r = np.array(hist_r)
hist_p = np.array(hist_p)
hist_r_media = np.mean(hist_r)
hist_p_media = np.mean(hist_p)
hist_r_sd = np.std(hist_r)
hist_p_sd = np.std(hist_p)
ttest,pval = stats.ttest_ind(hist_r,hist_p)
# print("p-value: ", pval)
def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(z, 3), pval
z,p = twoSampZ(hist_r_media, hist_p_media,hist_r_media - hist_p_media, hist_r_sd, hist_p_sd,len(hist_r), len(hist_p))
print(p)
'''
