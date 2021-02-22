import psutil
import platform
from datetime import datetime
from subprocess import call, check_output, run, STDOUT, CalledProcessError
import datetime
import os
import os.path


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def executaComando(parametro):
    start = datetime.datetime.now()
    try:
        exec_com = ["./transitorio"] + parametro
        if os.name == 'nt':
            exec_com = ["transitorio.exe"] + parametro
        # print(exec_com)
        #tmp = check_output(exec_com)
        str_exec = ' '.join(map(str, exec_com))
        return_code = call(str_exec, shell=True)
        if (return_code != 0):
            print("Error: ", return_code)
            return datetime.datetime.max - start
        elapsed_time = datetime.datetime.now() - start
        return elapsed_time
    except CalledProcessError as ex:
        #print ("Error: ", ex, ex.output, ex.returncode)
        return datetime.datetime.max - start


def testaSimulacao(param, threads_list):
    menor_tempo = 0.0
    menor_vector = datetime.datetime.now(), threads_list[0]
    for threads in threads_list:
        #print("Threads: ", threads)
        parametro = param + ["-t", threads]
        elapsed_time = executaComando(parametro)
        if (menor_tempo == 0.0):
            menor_tempo = elapsed_time
        #print ("Resultado GPU Cuda")
        #print (tmp)
        #print("Segundos: ", elapsed_time.seconds,":",elapsed_time.microseconds)
        if (elapsed_time < menor_tempo):
            menor_tempo = elapsed_time
            menor_vector = elapsed_time, threads
    return menor_vector


def grava_best_param(arq_entrada, threads_mult, threads_cuda):

    file1 = open(arq_entrada + ".simul", "w")
    file1.writelines(threads_mult + "\n")
    file1.writelines(threads_cuda + "\n")
    file1.close()


def obtem_best_param(arq_entrada):

    file1 = open(arq_entrada + ".simul", "r+")
    threads_mult = file1.readline()
    threads_cuda = file1.readline()
    file1.close()

    return threads_mult, threads_cuda


def simulacao(mult_problem_ini, mult_problem_final, arq_entrada, param=False):

    parametro_ini = ["-e", arq_entrada, "-d", mult_problem_ini]
    if (param):
        parametro_ini = ["-e", arq_entrada, "-d", mult_problem_ini, "-p"]

    if os.path.exists(arq_entrada + ".simul") == False:
        print("Testando Single Core")
        parametro = parametro_ini + ["-m", "0"]
        elapsed_time = executaComando(parametro)
        print("Single Segundos: {}".format(elapsed_time))

        print("Testando CUDA")
        threads_list = list()
        for i in range(1, 14):
            threads_list.append(str(pow(2, i)))

        parametro = parametro_ini + ["-m", "2"]
        menor_vector_cuda = testaSimulacao(parametro, threads_list)
        tempinho_cuda, threads_cuda = menor_vector_cuda
        print("Menor thread: " + threads_cuda +
              " tempo {}".format(tempinho_cuda))

        print("Testando Multi")
        threads_list = list()
        for i in range(1, 14):
            threads_list.append(str(pow(2, i)))

        parametro = parametro_ini + ["-m", "1"]
        menor_vector_mult = testaSimulacao(parametro, threads_list)
        tempinho_mult, threads_mult = menor_vector_mult
        print("Menor thread: " + threads_mult +
              " tempo {}".format(tempinho_mult))

        grava_best_param(arq_entrada, threads_mult, threads_cuda)

    threads_mult, threads_cuda = obtem_best_param(arq_entrada)

    print("="*20, "Processamento Final:", "="*20)
    parametro_ini = ["-e", arq_entrada, "-d", mult_problem_final, "-v"]

    best = "NONE"
    menor_tempo = datetime.timedelta.max

    ### MULTI ###
    parametro = parametro_ini + ["-m", "1", "-t", threads_mult]
    elapsed_time = executaComando(parametro)

    if (elapsed_time < menor_tempo):
        menor_tempo = elapsed_time
        best = "MULTI"
    print("Tempo Multi: {}".format(elapsed_time))

    os.system("pause")
 ### SINGLE ###
    parametro = parametro_ini + ["-m", "0"]
    elapsed_time = executaComando(parametro)

    if (elapsed_time < menor_tempo):
        menor_tempo = elapsed_time
        best = "SINGLE"
    print("Tempo Single: {}".format(elapsed_time))


    return best, menor_tempo

    ### CUDA ###
    parametro = parametro_ini + ["-m", "2", "-t ", threads_cuda]
    # print(parametro)
    elapsed_time = executaComando(parametro)

    if (elapsed_time < menor_tempo):
        menor_tempo = elapsed_time
        best = "CUDA"
    print("Tempo CUDA: {}".format(elapsed_time))
    return best, menor_tempo


print("="*40, "System Information", "="*40)
uname = platform.uname()
print(f"System: {uname.system}")
print(f"Node Name: {uname.node}")
print(f"Release: {uname.release}")
print(f"Version: {uname.version}")
print(f"Machine: {uname.machine}")
print(f"Processor: {uname.processor}")

# let's print CPU information
print("="*40, "CPU Info", "="*40)
# number of cores
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))

################

print("="*10, "Estratégia: Problema Mestrado Grande (chamadas sequenciais)", "="*10)
mult_problem_inicial = "1"
mult_problem_final = "1"
arq_entrada = "entrada_gg.dat"

best, menor_tempo = simulacao(
    mult_problem_inicial, mult_problem_final, arq_entrada, False)
print("Best: " + best + " tempo {}".format(menor_tempo))

#print("="*10, "Estratégia: Problema Mestrado Grande (chamadas paralelas)", "="*10)
#best, menor_tempo = simulacao(mult_problem_inicial, mult_problem_final, arq_entrada, True)
#print ("Best: " + best + " tempo {}".format(menor_tempo))

################

print("="*10, "Estratégia: Problema Duto OSPLAN II (chamadas sequenciais)", "="*10)
mult_problem_inicial = "1"
mult_problem_final = "1"
arq_entrada = "entrada_OSPLAN_II.dat"

best, menor_tempo = simulacao(
    mult_problem_inicial, mult_problem_final, arq_entrada, False)
print("Best: " + best + " tempo {}".format(menor_tempo))

#print("="*10, "Estratégia: Problema Duto OSPLAN II (chamadas paralelas)", "="*10)
#best, menor_tempo = simulacao(mult_problem_inicial, mult_problem_final, arq_entrada, True)
#print ("Best: " + best + " tempo {}".format(menor_tempo))

print("="*10, "Estratégia: Problema Duto OPASA 10pol (chamadas sequenciais)", "="*10)
mult_problem_inicial = "1"
mult_problem_final = "1"
arq_entrada = "entrada_OPASA_10.dat"

best, menor_tempo = simulacao(
    mult_problem_inicial, mult_problem_final, arq_entrada, False)
print("Best: " + best + " tempo {}".format(menor_tempo))

exit(0)
