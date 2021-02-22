#include "cuda.h"
#include "common/book.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>

#ifdef LINUX
#include <getopt.h>
#include <sys/time.h>
#include <unistd.h>
#include <wait.h>
#else
#include <ctype.h>
#define __GNU_LIBRARY__

#define MESTRADOCUDA
#include "gettimeofday_win.h"
#include "getopt.h"

#endif

#define PI 3.14159265
#define G 9.806

float hr; 
float f;
float l;            // Tamanho horizontal do pipe em metros
float d;            // Diametro do pipe
float dt;           // Delta T 
float tmax;         // Tempo máximo da Simulação em seungos (Parametro bom pra mexer)

float cdao;

float r;
float a;            // Celeridade da Onda
float qo;
float b;
float cm;
float cp;
float dx;           // Delta X (Comprimento do tubo pelo número de seguimentos)
float tf;

float as;
float t;
float bp;
float bm;

struct _HQT {
    float H;
    float Q;
    float T;
};

struct _HQ {
    float H;
    float Q;
};

struct _HQT *matrizHQT;           //Vetor da Strutura H (Pressão) Q(Vazão) e T(Tempo)
struct _HQT *_dMatrizHQT;         //Matriz da Strutura H (Pressão) Q(Vazão) e T(Tempo) em device

struct _HQ *atualHQ;
struct _HQ *proximolHQ;

unsigned int n;              //Número de segmentos
unsigned int N;              //Numeros de segmentos * Delta T
unsigned int ns;             // Numero de segmentos + segmentos montante + segmentos justante
unsigned int linha;

struct ConstantesPrograma {
    unsigned int ns;
    float qo;
    float hr;
    float r;
    float b;
    float t;
    float dt;
    float tmax;
    unsigned int controle;
    unsigned int nthreads;

    __device__ unsigned int get_ns() { return ns; }
    __device__ float get_qo() { return qo; }
    __device__ float get_hr() { return hr; }
    __device__ float get_r() { return r; }
    __device__ float get_b() { return b; }
    __device__ float get_t() { return t; }
    __device__ float get_dt() { return dt; }
    __device__ float get_tmax() { return tmax; }
    __device__ unsigned int get_controle() { return controle; }
    __device__ unsigned int get_nthreads() { return nthreads; }
};

__constant__ __device__ ConstantesPrograma ConstantDeviceConstantes;

/* Globals set by command line args */
int verbosity = 0; /* print trace if set */
int exibe_tempos = 0;

unsigned int nblocks; //Número de blocos que vão ser chamados
unsigned int nthreads; //Número de threads por blocls que vão ser chamados

unsigned int multiplicador = 1;

int gravarDisco = 0;
char* narq_e = NULL;
char* narq_s = NULL;

char narqe_default[] = "entrada.dat";
char narqs_default[] = "saida.dat";

// NUm Threads
int chunk = 512;
 
enum _modo {SINGLE = 0, MULTI = 1, CUDA = 2};
enum _modo modo = SINGLE;

extern int paralelo = 0;

//Inicializa vetor com ZEROS
void initZero(float *vector, int size) {
    int i;
    for (i = 0; i < size; i++) {
        vector[i] = 0.0;
    }  
}

void initZeroHQT(struct _HQT *vector, int size) {
    int i;
    #pragma omp parallel for if(paralelo)
    for (i = 0; i < size; i++) {
        vector[i].H = 0.0;
        vector[i].Q = 0.0;
        vector[i].T = 0.0;
    }  
}

void initZeroHQ(struct _HQ *vector, int size) {
    int i;
    #pragma omp parallel for if(paralelo)
    for (i = 0; i < size; i++) {
        vector[i].H = 0.0;
        vector[i].Q = 0.0;
    }  
}

/*
 * printUsage - Print usage info
 */
void printUsage(char* argv[]) {
    printf("Uso: %s [-hv] -e <file> -s <file> -m 0|1|2 -t <num> -b <num> -d <num> -p -t\n", argv[0]);
    printf("Options:\n");
    printf("  -h          Print this help message.\n");
    printf("  -v          Optional verbose flag.\n");
    printf("  -e <file>   Arquivo de entrada.\n");
    printf("  -s <file>   Arquivo de saida com o resultado.\n");
    printf("  -m <modo>   0 = Singlecore | 1 = Multicore | 2 = CUDA: Utiliza a versão indicada.\n");
    printf("  -t <num>    Número de Threads.\n");
    printf("  -d <num>    Número de vezes que vai rodar o simulação (Simula o tamanho do problema).\n");
    printf("  -p          Roda as instâncias do problema de forma paralela.\n");
    printf("  -t          Exibe os tempos apenasa.\n");
    
    printf("\nExamples:\n");
    printf("  >  %s -v -e entrada.dat -s resuldado.dat -m 2 -b 16 -t 512 -d 5000\n", argv[0]);
    printf("  >  %s -v -e entrada.dat -s resuldado.dat -m 1 -c 512 -d 5000\n", argv[0]);
    printf("  >  %s -v -e entrada.dat -s resuldado.dat -m 0 -d 5000\n", argv[0]);
    exit(0);
}

void printInitParam() {
    printf("Processando com os parametros:\n");
    printf("\t Arquivo de entrada: %s\n", narq_e);
    printf("\t Arquivo de resultados: %s\n", narq_s);
    printf("\t Verbosity: %d\n", verbosity);
    printf("\t Modo (0 = Singlecore | 1 = Multicore | 2 = CUDA): %d\n", modo);
    printf("\t Threads OpenMP: %d\n", chunk);
    printf("\t Threads Cuda: %d\n", nthreads);
    printf("\t Problema aumentado em: %d\n", multiplicador);
}

void printInitMoc() {
    printf("Dados Iniciais:\n");
    printf("\tElementos da matriz = %d\n", N);
    printf("\tlinha = %d\n", linha);
    printf("\ttmax = %4.2f\n", tmax);
    printf("\tdt = %4.2f\n", dt);
    printf("\tdx = %4.2f\n", dx);
    printf("\tl = %f\n", l);
    printf("\tn = %d\n", n);
    printf("\tns = %d\n", ns);
    printf("\tN = %d\n", N);  
}


//Grava o resultado obtido no arquivo definido via paramêtro
void gravaHQTDisco(struct _HQT *matrizHQT, int tamanho) {
    if (gravarDisco == 0) {
        return;
    }
    FILE *arq_s;    // Arquivo de Saída
    if (verbosity) printf("Abrindo %s..\n", narq_s);
    arq_s = fopen(narq_s, "w"); 

    if (arq_s == NULL) {
        if (verbosity) perror ("Erro ao abrir arquivo de saida");
        #ifndef LINUX
        if (verbosity) printf( "Value of errno: %d\n", errno );
        #endif
    } else {
        if (verbosity) printf("Gravando saída\n");
        int i;
        for(i = 0; i < tamanho; i++) {
            fprintf(arq_s, "%4.2f      %4.5f      %4.5f\n", matrizHQT[i].T, matrizHQT[i].H, matrizHQT[i].Q);
        }

        fclose(arq_s);
    }
    gravarDisco = 0; //Só preciso gravar 1 resultado
}

__device__ void printConstantes(ConstantesPrograma *ct) {
    printf("Constantes no DEVICE: \n");
    printf("\tns %d", ct->ns);
    printf("\tqo %4.5f", ct->qo);
    printf("\thr %4.5f", ct->hr);
    printf("\tr %4.5f", ct->r);
    printf("\tb %4.5f", ct->b);
    printf("\tt %4.5f", ct->t);
    printf("\tdt %4.5f", ct->dt);
    printf("\ttmax %4.5f", ct->tmax);
    printf("\tcontrole %d", ct->controle);
    printf("\tnthreads %d", ct->nthreads);

}

//Cálculo do MOC (single core)
void calc_HQ_CPU();

//Cálculo do MOC (multi core)
void calc_HQ_MCPU();

//Cálculo do MOC (GPU)
__global__ void calc_HQ_RegimeTransiente(_HQT *_dMatrizHQT);
void calc_HQ_GPU();

int main(int argc, char* argv[]) {

    //Dá pau pra mostrar o tempo
    //setlocale(LC_ALL, "pt_BR.UTF-8");
    
    char str[100];       //Buffer

    float tcpu, tfunc;
    struct timeval p1start, p1stop, totalt;
    struct timeval k1start, k1stop, ktotalt;
    
    narq_e = narqe_default;
    narq_s = narqs_default;
    gravarDisco = 0;
    chunk = 512;
    verbosity = 0;
    modo = SINGLE;
    multiplicador = 1;
    nthreads = 512;
    paralelo = 0;
    exibe_tempos = 0;

    char c;
    int temp_mode = 0;
    while ( (c = getopt (argc, argv, "e:s:m:c:t:d:pvhk")) != -1) {
        
        switch(c) {
            case 'e':
                narq_e = optarg;
                break;
            case 's':
                narq_s = optarg;
                gravarDisco = 1;
                break;
            case 'm':
                temp_mode = atoi(optarg);
                if (temp_mode > 2) {
                    printUsage(argv);
                    exit(1);
                }
                break;
            case 't':
                nthreads = atoi(optarg);
                chunk = nthreads;
                break;
            case 'd':
                multiplicador = atoi(optarg);
                break;
            case 'v':
                verbosity = 1;
                break;
            case 'k':
                exibe_tempos = 1;
                break;
            case 'p':
                paralelo = 1;
                break;
            case 'h':
                printUsage(argv);
                exit(0);
            default:
                printUsage(argv);
                exit(1);
        }

    }
    
    if (temp_mode > 0 && temp_mode < 3) {
        modo = MULTI;
        if (temp_mode > 1) 
            modo = CUDA;
    }
        
    if (verbosity) printInitParam();

    if (verbosity)
        printf("Inicio do processamento\n");

    //Teste
    //Leitura dos Dados
    FILE *arq_e;    // Arquivo de Entrada        
    arq_e = fopen(narq_e, "r");
    if (arq_e == NULL) {
        if (verbosity) {
            perror ("Erro ao abrir arquivo");
            #ifndef LINUX
            printf( "Value of errno: %d\n", errno );
            #endif
        }
        exit(1);
    } else {
        //fscanf(arq_e, "%s", str);
        
        fscanf(arq_e, "%f", &hr);
            fscanf(arq_e, "%s", str);
        fscanf(arq_e, "%f", &f);
            fscanf(arq_e, "%s", str);
        fscanf(arq_e, "%f", &l);
            fscanf(arq_e, "%s", str);
        fscanf(arq_e, "%f", &d);
            fscanf(arq_e, "%s", str);            
        fscanf(arq_e, "%d", &n);
            fscanf(arq_e, "%s", str);  
        fscanf(arq_e, "%f", &tmax);
            fscanf(arq_e, "%s", str);              
        fscanf(arq_e, "%f", &a);
            fscanf(arq_e, "%s", str);

        fscanf(arq_e, "%f", &cdao);
        fclose(arq_e);    
    }   

    if (verbosity) {
        printf("Carregou o arquivo com os dados:\n");
        printf("\thr = %4.2f\n", hr);
        printf("\tf = %4.2f\n", f);
        printf("\t(comprimento) l = %4.2f\n", l);
        printf("\t(diametro) d = %4.5f\n", d);
        printf("\t(Segmentos) n = %d\n", n);
        printf("\t(Tempo Max) tmax =  %4.2f\n", tmax);
        printf("\t(celeridade) a = %4.2f\n", a);
        printf("\tcdao = %4.5f\n", cdao);
    }

    //Limpa Contadores CPU
    timerclear(&p1start); 
    timerclear(&p1stop); 

    gettimeofday(&p1start, NULL);

    int cont;
    timerclear(&k1start); 
    timerclear(&k1stop); 

    gettimeofday(&k1start, NULL);
    for (cont = 0; cont < multiplicador; cont++) {
        //printf("Rodando %d modo %d\n", cont, modo);
        switch(modo) {
            case SINGLE:
                calc_HQ_CPU();  
                break;
            case MULTI:
                calc_HQ_MCPU();    
                break;
            case CUDA:
                //printf("Cuda %d\n", cont);
                calc_HQ_GPU();    
                break;
    
        }
    
    }
    gettimeofday(&k1stop, NULL);
    timersub(&k1stop, &k1start, &ktotalt);
    tfunc = 0.001 * (ktotalt.tv_sec * 1000000 + ktotalt.tv_usec);

    if (exibe_tempos)
        printf("Tempo transitorio = %f ms\n", tfunc);

    gettimeofday(&p1stop, NULL);

    //Gasto de tempo ms
    timersub(&p1stop, &p1start, &totalt);

    tcpu = 0.001 * (totalt.tv_sec * 1000000 + totalt.tv_usec);

    if (exibe_tempos)
        printf("=== Tempo total de execução = %f ms, Modo %d ===\n", tcpu, modo);

}

void calc_HQ_CPU() {
    int i;
    int controle;       //Controle para linkar vetores H e Q as matrizes (Controla o pulo para as linhas)

    // Constantes //
    as = (PI * d * d) / 4;
    dx = l / n;
    dt = dx / a;
    ns = n + 2; //segmentos + montante + justrante
    r = f * dx / ( 2 * G * d * as * as);
    b = a/(G * as);

    //Alocação Vetor e Matrizes
    linha = (int)(tmax / dt);
    N = ns + (ns * linha); //Delta T para 0 + conjunhto de delta T + ultima iteração Delta T

    if (verbosity) printInitMoc();

    controle = 0;

    size_t sizeMATRIX = N; //* sizeof(float);
    size_t sizeVECTOR = ns; //* sizeof(float);

    matrizHQT = (struct _HQT *)malloc(sizeof(struct _HQT) * sizeMATRIX);
    
    atualHQ = (struct _HQ *)malloc(sizeof(struct _HQ) * sizeVECTOR);
    proximolHQ = (struct _HQ *)malloc(sizeof(struct _HQ) * sizeVECTOR);

    // Inicializa H e Q

    initZeroHQT(matrizHQT , sizeMATRIX);

    initZeroHQ(atualHQ, sizeVECTOR);
    initZeroHQ(proximolHQ, sizeVECTOR);

    t = 0;

    //Cálculo do Regime Permanente inicial
    if (verbosity) printf("Regime Permanente inicial\n");
    qo = sqrt(((cdao * cdao) * 2 * hr * hr)/(1 + (cdao * cdao) * 2 * G * n * r));

    for(i = 0; i < ns; i++) {
        atualHQ[i].H = hr - (i) * r * qo * qo;
        atualHQ[i].Q = qo;
        matrizHQT[i].H = atualHQ[i].H;
        matrizHQT[i].Q = atualHQ[i].Q;
        matrizHQT[i].T = t;    
    }

    //Inicio Transitório
    while (t < tmax) {
        t = t + dt;
        controle++;

        //Pontos interiores;
        for(i = 1; i < ns; i++) {
            cp = atualHQ[i-1].H + b * atualHQ[i-1].Q;
            bp = b + r * fabs(atualHQ[i-1].Q);
            cm = atualHQ[i+1].H - b * atualHQ[i+1].Q;
            bm = b + r * fabs(atualHQ[i+1].Q);
            proximolHQ[i].Q = (cp - cm) / (bp + bm);
            proximolHQ[i].H = cp - bp * proximolHQ[i].Q;
        }

        //Condição de contorno de montante: Reservatório

        proximolHQ[0].H = hr;
        cm = atualHQ[1].H - b * atualHQ[1].Q;
        bm = b + r * fabs(atualHQ[1].Q);
        proximolHQ[0].Q = (hr - cm)/bm;

        //condição de contorno de jusante: Válvula

        cp = atualHQ[ns-2].H + b * atualHQ[ns-2].Q;
        proximolHQ[ns-1].Q = 0;
        proximolHQ[ns-1].H = cp;

        //Atualização

        for (i = 0; i < ns; i++) {
            atualHQ[i].H = proximolHQ[i].H;
            atualHQ[i].Q = proximolHQ[i].Q;   
            
            unsigned int ind = (controle * ns) + i;
            if (ind < sizeMATRIX) {
                matrizHQT[ind].H = atualHQ[i].H;
                matrizHQT[ind].Q = atualHQ[i].Q;
                matrizHQT[ind].T = t; 
            }
        }

    }

    if (verbosity) printf("Processamento Completo\n");
    //Impressão completa
    gravaHQTDisco(matrizHQT, sizeMATRIX);
    
    free(matrizHQT);

    free(atualHQ);
    free(proximolHQ);
}

void calc_HQ_MCPU() {
    int i;
    int linha;
    int controle;       //Controle para linkar vetores H e Q as matrizes (Controla o pulo para as linhas)

    // Constantes //
    as = (PI * d * d) / 4;
    dx = l / n;
    dt = dx / a;
    ns = n + 2; //segmentos + montante + justrante
    r = f * dx / ( 2 * G * d * as * as);
    b = a/(G * as);

    //Alocação Vetor e Matrizes
    //linha = ((int)(tmax/dt))+1;;
    //N = ns + (ns * linha)+ns; //Delta T para 0 + conjunhto de delta T + ultima iteração Delta T
    linha = (int)(tmax / dt);  
    N = ns + (ns * linha);

    if (verbosity) printInitMoc();

    controle = 0;

    size_t sizeMATRIX = N;
    size_t sizeVECTOR = ns;

    matrizHQT = (struct _HQT *)malloc(sizeof(struct _HQT) * sizeMATRIX);
    
    atualHQ = (struct _HQ *)malloc(sizeof(struct _HQ) * sizeVECTOR);
    proximolHQ = (struct _HQ *)malloc(sizeof(struct _HQ) * sizeVECTOR);

    // Inicializa H e Q

    initZeroHQT(matrizHQT , sizeMATRIX);

    initZeroHQ(atualHQ, sizeVECTOR);
    initZeroHQ(proximolHQ, sizeVECTOR);

    t = 0;

    //Cálculo do Regime Permanente inicial
    if (verbosity) printf("Regime Permanente inicial\n");
    qo = sqrt(((cdao * cdao) * 2 * hr * hr)/(1 + (cdao * cdao) * 2 * G * n * r));

    #pragma omp parallel shared(atualHQ,matrizHQT,hr,r,qo,t,ns,chunk) private(i) 
    {
        #pragma omp for schedule(dynamic,chunk) nowait
            for(i = 0; i < ns; i++) {
                atualHQ[i].H = hr - (i) * r *qo * qo;
                atualHQ[i].Q = qo;
                matrizHQT[i].H = atualHQ[i].H;
                matrizHQT[i].Q = atualHQ[i].Q;
                matrizHQT[i].T = t;    
            }
    }

    //Inicio Transitório
    while (t < tmax) {
        if (verbosity) printf("Passo T %f de %f\n", t, tmax);
        t = t + dt;
        controle++;

        #pragma omp parallel shared(atualHQ,proximolHQ,r,b,ns,chunk) private(i,cm,bm,cp,bp) 
        {
            #pragma omp for schedule(dynamic,chunk) nowait
                //Pontos interiores;
                for(i = 1; i < ns; i++) {
                    cp = atualHQ[i-1].H + b * atualHQ[i-1].Q;
                    bp = b + r * fabs(atualHQ[i-1].Q);
                    cm = atualHQ[i+1].H - b * atualHQ[i+1].Q;
                    bm = b + r * fabs(atualHQ[i+1].Q);
                    proximolHQ[i].Q = (cp - cm) / (bp + bm);
                    proximolHQ[i].H = cp - bp * proximolHQ[i].Q;
                }
        }
        //Condição de contorno de montante: Reservatório

        proximolHQ[0].H = hr;
        cm = atualHQ[1].H - b * atualHQ[1].Q;
        bm = b + r * fabs(atualHQ[1].Q);
        proximolHQ[0].Q = (hr - cm)/bm;

        //condição de contorno de jusante: Válvula

        cp = atualHQ[ns-2].H + b * atualHQ[ns-2].Q;
        proximolHQ[ns-1].Q = 0;
        proximolHQ[ns-1].H = cp;

        //Atualização
        #pragma omp parallel shared(atualHQ,proximolHQ,matrizHQT,ns,t,chunk) private(i) 
        {
            #pragma omp for schedule(dynamic,chunk) nowait
                for (i = 0; i < ns; i++) {
                    atualHQ[i].H = proximolHQ[i].H;
                    atualHQ[i].Q = proximolHQ[i].Q;                    
                    unsigned int ind = (controle * ns) + i;
                    if (ind < sizeMATRIX) {
                        matrizHQT[ind].H = atualHQ[i].H;
                        matrizHQT[ind].Q = atualHQ[i].Q;
                        matrizHQT[ind].T = t; 
                    }
            
                }
        }        
    }

    if (verbosity) printf("Processamento Completo\n");
    //Impressão completa
    gravaHQTDisco(matrizHQT, sizeMATRIX);
    
    free(matrizHQT);
    
    free(atualHQ);
    free(proximolHQ);
}

void calc_HQ_GPU() {
    //int linha;
    int controle;       //Controle para linkar vetores H e Q as matrizes (Controla o pulo para as linhas)

    // Constantes //
    as = (PI * d * d) / 4;
    dx = l / n;
    dt = dx / a;
    ns = n + 2; //segmentos + montante + justrante
    r = f * dx / ( 2 * G * d * as * as);
    b = a/(G * as);

    //Alocação Vetor e Matrizes
    linha = (int)(tmax / dt);
    N = ns + (ns * linha); //Delta T para 0 + conjunhto de delta T + ultima iteração Delta T

    //printf("Uso da memória %d para N=%d \n", (int)(sizeof(struct _HQT) * N), N);
    if (verbosity) printInitMoc();

    controle = 0;

    size_t sizeMATRIX = N; //* sizeof(float);

    matrizHQT = (_HQT *)malloc(sizeof(_HQT) * sizeMATRIX);
    
    // Inicializa H e Q
    initZeroHQT(matrizHQT , sizeMATRIX);
    //Aloca vetores na memória do device
    HANDLE_ERROR( cudaMallocManaged(&_dMatrizHQT, (sizeof(_HQT) * sizeMATRIX)) );

    //Dados : Host -> Device
    HANDLE_ERROR( cudaMemcpy(_dMatrizHQT, matrizHQT, (sizeof(_HQT) * sizeMATRIX), cudaMemcpyHostToDevice) );
    
    //Cálculo do Regime Permanente inicial
    t = 0;
    qo = sqrt(((cdao * cdao) * 2 * hr * hr)/(1 + (cdao * cdao) * 2 * G * n * r));
 
    nblocks = (ns + nthreads) / nthreads;
    
    //no c++ daria pra passar um objeto ?
    ConstantesPrograma *host_constantes = (ConstantesPrograma *)malloc(sizeof(ConstantesPrograma));
    host_constantes->ns = ns;
    host_constantes->hr = hr;
    host_constantes->r = r;
    host_constantes->b = b;
    host_constantes->dt = dt;
    host_constantes->tmax = tmax;
    host_constantes->controle = controle;
    host_constantes->t = t;
    host_constantes->qo = qo;
    host_constantes->nthreads = nthreads;

    //HANDLE_ERROR( cudaMallocManaged(&ConstantDeviceConstantes, sizeof(ConstantesPrograma)) );
    HANDLE_ERROR( cudaMemcpyToSymbol(ConstantDeviceConstantes, host_constantes, sizeof(ConstantesPrograma)) );
    free(host_constantes);

    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    calc_HQ_RegimeTransiente<<<nblocks, nthreads, nthreads*sizeof(_HQ)>>>(_dMatrizHQT);

    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	
	float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    
    if (exibe_tempos)
        printf("Tempo do Kernel = %f ms\n", elapsedTime);
    
    //Dados : Device -> Host
    HANDLE_ERROR( cudaMemcpy(matrizHQT, _dMatrizHQT, (sizeof(_HQT) * sizeMATRIX), cudaMemcpyDeviceToHost) );

    if (verbosity) printf("Processamento Completo\n");
    //Impressão completa
    gravaHQTDisco(matrizHQT, sizeMATRIX);

    free(matrizHQT);
    cudaFree(_dMatrizHQT);
}

__global__ void calc_HQ_RegimeTransiente(_HQT *_dMatrizHQT) {
    /*
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    */

    float _cp = 0.0;
    float _bp = 0.0;
    float _cm = 0.0;
    float _bm = 0.0;  

    unsigned int _ns = ConstantDeviceConstantes.get_ns();
    float _qo = ConstantDeviceConstantes.get_qo();
    float _hr = ConstantDeviceConstantes.get_hr();
    float _r = ConstantDeviceConstantes.get_r();
    float _b = ConstantDeviceConstantes.get_b();
    float _t = ConstantDeviceConstantes.get_t();
    float _dt = ConstantDeviceConstantes.get_dt();
    float _tmax = ConstantDeviceConstantes.get_tmax();
    unsigned int _nthreads = ConstantDeviceConstantes.get_nthreads();

    unsigned int _controle = 0;  

    //printConstantes(&ConstantDeviceConstantes);

    int _i = threadIdx.x + blockIdx.x * blockDim.x;
    int indice = (int)(_i % _nthreads);
  
    extern __shared__ _HQ tempoAtual[]; 
    extern __shared__ _HQ tempoProximo[]; 

    //Regime Permanente inicial
    if (_i < _ns) {
        tempoAtual[indice].H = _hr - (_i) * _r *_qo * _qo;
        tempoAtual[indice].Q = _qo;

        __syncthreads();

        _dMatrizHQT[(_controle * _ns) + _i].H = tempoAtual[indice].H;
        _dMatrizHQT[(_controle * _ns) + _i].Q = tempoAtual[indice].Q;
        _dMatrizHQT[(_controle * _ns) + _i].T = _t;  
    }

    //Regime Transitório
    if (_i < _ns) {

        while (_t < _tmax) {
            _t = _t + _dt;
            _controle++;

            //Pontos interiores;
            if (_i < _ns && _i >= 1) {
                _cp = tempoAtual[indice-1].H + _b * tempoAtual[indice-1].Q;
                _bp = _b + _r * fabs(tempoAtual[indice-1].Q);
                _cm = tempoAtual[indice+1].H - _b * tempoAtual[indice+1].Q;
                _bm = _b + _r * fabs(tempoAtual[indice+1].Q);
                tempoProximo[indice].Q = (_cp - _cm) / (_bp + _bm);
                tempoProximo[indice].H = _cp - _bp * tempoProximo[indice].Q;
            }

            //Condição de contorno de montante: Reservatório
            if (_i == 0) {
                tempoProximo[0].H = _hr;
                _cm = tempoAtual[1].H - _b * tempoAtual[1].Q;
                _bm = _b + _r * fabs(tempoAtual[1].Q);
                tempoProximo[0].Q = (_hr - _cm)/_bm;
            }


            //condição de contorno de jusante: Válvula
            if (_i == _ns - 1) {
                _cp = tempoAtual[_ns-2].H + _b * tempoAtual[_ns-2].Q;
                tempoProximo[_ns-1].Q = 0;
                tempoProximo[_ns-1].H = _cp;
            }

            //Atualização

            tempoAtual[indice].H = tempoProximo[indice].H;
            tempoAtual[indice].Q = tempoProximo[indice].Q;

            __syncthreads();

            _dMatrizHQT[(_controle * _ns) + _i].H = tempoAtual[_i].H;
            _dMatrizHQT[(_controle * _ns) + _i].Q = tempoAtual[_i].Q;
            _dMatrizHQT[(_controle * _ns) + _i].T = _t;

        }
    }
}
