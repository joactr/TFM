import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-c', metavar="archivo_cliente", help="Archivo con los scores de los clientes", required=True, type=str)
required.add_argument('-i', metavar="archivo_impostor", help="Archivo con los scores de los impostores", required=True, type=str)
parser.add_argument('-fn', metavar="falsos_negativos", help="Valor de los falsos negativos para cálculo",type=float)
parser.add_argument('-fp', metavar="falsos_positivos", help="Valor de los falsos positivos para cálculo",type=float)
parser._action_groups.append(optional)
parsed_args = parser.parse_args()
args = vars(parsed_args)

clientes = [] #FN si por debajo de umbral
impostores = [] #1 - FP por debajo de umbral

# Leer archivos
with open(args['c'], 'r') as fc:
    for line in fc:
        elem = line.split()
        clientes.append(float(elem[0])) #Solo probabilidades
    clientes.sort()

with open(args['i'], 'r') as fi:
    for line in fi:
        elem = line.split()
        impostores.append(float(elem[0]))
    impostores.sort()

if args['fn']:
    fn_entrada = float(args['fn'])
    fn_cercano = float('inf')

if args['fp']:
    fp_entrada = float(args['fp'])
    fp_cercano = float('inf')

#Prueba U de Mann-Whitney, permite calcular AUC
def calcAUC(c, i):
    if c > i:
        return 1
    elif c < i:
        return 0
    else:
        return 0.5

coord_x = []
coord_y = []

#Variables de índice
cli_id = 0
imp_id = 0
dist = float('inf')

#Calcular fp y fn para cada score
fp = len(impostores)
fn = 0
coord_x, coord_y = [], []

#Recorremos las scores
while True:
    fp_norm = fp / len(impostores)
    fn_norm = fn / len(clientes)
    coord_x.append(fp_norm)
    coord_y.append(1 - fn_norm)
    
    if fp_norm == 0 or fn_norm == 1:
        break
    
    if cli_id == len(clientes):
        umb = impostores[imp_id]
        fp -= 1
        imp_id += 1
    elif imp_id == len(impostores):
        umb = clientes[cli_id]
        fn += 1
        cli_id += 1
    elif clientes[cli_id] <= impostores[imp_id]:
        umb = clientes[cli_id]
        fn += 1
        cli_id += 1
    else:
        umb = impostores[imp_id]
        fp -= 1
        imp_id += 1
    
    if args['fp'] and abs(fp_norm - fp_entrada) < fp_cercano:
        fn_objetivo, umbral_fn, fp_cercano = fn_norm, umb, abs(fp_norm - fp_entrada)
    
    if args['fn'] and abs(fn_norm - fn_entrada) < fn_cercano:
        fp_objetivo, umbral_fp, fn_cercano = fp_norm, umb, abs(fn_norm - fn_entrada)
    
    if abs(fp_norm - fn_norm) < dist:
        umbral_eq, f_equal, dist = umb, fp_norm, abs(fp_norm - fn_norm)

#Valores con umbral 1
fp = 0
fn = len(clientes)
coord_x.append(fp/len(impostores))
coord_y.append(1-fn/len(clientes))

auc = 0
for c in clientes:
    for i in impostores:
        auc += calcAUC(c,i)

auc = auc / (len(clientes) * len(impostores))

print("Umbral con FN == FP: %.6f. FN == FP: %.6f" % (umbral_eq, f_equal))

if args['fn']:
    print("FP(FN = %.6f) = %.6f con el umbral correspondiente: %.6f\n" % (fn_entrada, fp_objetivo, umbral_fp))
if args['fp']:
    print("FN(FP = %.6f) = %.6f con el umbral correspondiente: %.6f\n" % (fp_entrada, fn_objetivo, umbral_fn))

#Cálculo de métrica D-prime
pos = np.array(clientes)
neg = np.array(impostores)
upos = np.mean(pos)
uneg = np.mean(neg)

d_prime = (upos - uneg) / np.sqrt(np.var(pos) + np.var(neg))
print("Valor D-prime (d'): %.6f\n" % d_prime)

print("Area bajo la curva ROC (AUC): %.5f\n" % auc)
plt.plot(coord_x, coord_y)
plt.xlabel('FP')
plt.ylabel('1-FN')
plt.grid(alpha=0.15)
plt.title('CURVA ROC')
plt.annotate('AUC = %.5f'% (auc), xy=(0.75, 0.05), xycoords='axes fraction')
plt.waitforbuttonpress()
plt.savefig("rocB.png",dpi=300)