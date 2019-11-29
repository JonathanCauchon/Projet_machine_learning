#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""

loader pour pytorch
attention: premier brouillon
sera amélioré d'ici rencontre lundi 2 décembre
sera amélioré selon feedbacks de l'équipe


pyfortin -- 20191128


TODO -- utiliser données à JC aussi
TODO -- mieux séparer les échantillons (ex: Re vs Im)
TODO -- modulariser et documenter script



===================================
Description des fichiers de données
===================================

. en concaténant tous les fichiers de données on totalise nbTotalDonnees
. chaque fichier de données contient

	nbLignes = 3, 11 ou 21
	nbDonnees = nbLignes - 1
	nbAttributs = 8

. la 1ere ligne décrit les nbAttributs séparés par des virgules
. certains attributs sont en plusieurs dimensions
. les attributs multi-dimensionnels sont enregistrés sur plus d'une colonne
. les nbDonnees lignes suivantes contiennent les valeurs des attributs
. pour chaque donnée, les valeurs d'attributs sont séparées par des espaces

utilisons des indices d'attributs débutant à 0
utilisont des indices de colonnes débutant à 0

ainsi, le décalage en colonnes de la première -- ou seule -- valeur
numérique rattachée à chaque attribut sera donnée par

	$\delta_i = \sum_{j=0}^{i-1}{d_i}$

indice   symbole        type       domaine   dimensions    décalage
$i$      $\sigma_i$     $\tau_i$   $\rho_i$  $d_i$         $\delta_i$
----------------------------------------------------------------------
0        a              float      R+        1             0
1        N              int        N+        1             1
2        kappa          float      R+        1             2
3        lambdaB        float      R+        1             3
4        apodization    float      R+        101           4
5        period         float      R+        101           105
6        real(E_drop)   float      R         1001          206
7        imag(E_drop)   float      R         1001          1207
----------------------------------------------------------------------



==========
Références
==========

[1]@mastersthesis{ st2017contra,
  title={Contra-directional couplers as optical filters on the silicon on insulator platform},
  author={St-Yves, Jonathan},
  year={2017},
  school={Université Laval},
}


[2]@article{ shi2013silicon,
  title={Silicon photonic grating-assisted, contra-directional couplers},
  author={Shi, Wei
		    and Wang, Xu and Lin, Charlie and Yun, Han
		    and Liu, Yang and Baehr-Jones, Tom and Hochberg, Michael
			and Jaeger, Nicolas A. F. and Chrostowski, Lukas},
  journal={Optics Express},
  volume={21},
  number={3},
  pages={3633--3650},
  year={2013},
  publisher={Optical Society of America}

[3]@mastersthesis{ hammond2019machine,
  title={Machine Learning Methods for Nanophotonic Design, Simulation, and Operation},
  author={Hammond, Alec Michael},
  year={2019},
  school={Brigham Young University},
}


=========
Questions
=========

???: Quel-est le champ en entrée (*input*) qui cause le champ E_drop
	en sortie (*drop*)? Il devrait être le même pour toutes les données,
	mais quel est-il?

???: Quelle-est la longueur d'onde centrale du champ d'entrée et
	du port *through*? Était-ce 1550 nm ?

???: Est-ce que lambdaB est la longueur d'onde centrale du port *drop*?
	Est-ce qu'elle correspond à $\lambda_D$ dans l'article de Wei Shi [2]
	et dans le mémoire de Jonathan St-Yves [1]?

???: Est-ce que la période donnée à chaque extrémité des blocs d'analyse
	de la méthode par matrice de transfert correspond à une moyenne
	locale de la période à cette position? Sinon, à quoi correspond-elle?

???: Est-ce une bonne idée de convertir les valeurs de "N" en int?
	Ne serait-il pas plus naturel de les laisser en float (REELPOS)?
	Je pense qu'il vaut mieux le forcer en int pour être en mesure de
	facilement le feeder à scikit-learn ou autre, quitte à le reconvertir
	en float pour les np.array --> tensor de pytorch.

"""

import numpy as np
import pandas as pd
import pickle as pk
from os import path, listdir
from matplotlib import pyplot



pd.set_option('display.max_columns',8)
pd.set_option('display.max_rows',8)
pd.set_option('display.width',40)






FIC_DONNEES_JC0 = '../jc/donnees/Dataset_v0.txt'
FIC_DONNEES_JC1 = '../jc/donnees/Dataset_v1.txt'
DIR_DONNEES_JSY = '../jsy/donnees'
EXT_DONNEES_JSY = '.txt'
FIC_DONNEES_PKL = 'donneesRemachees.pickle'






r"""
\brief
Domaines possibles pour les valeurs numériques des attributs.
"""
DOMAINES = ['REEL','REELPOS','NAT','NATPOS']



r"""
\brief
Nombre de blocs successifs de longueurs égales dans lesquels
on décompose le réseau afin d'en effectuer l'analyse par méthode
de matrice de transfert. Commes les positions des extrémités inférieures
et supérieures des blocs seront aussi considérées, il y aura donc
1+NB_BLOCS positions dans les profils.
"""
NB_BLOCS = 100



r"""
\brief
Nombre d'intervalles de longueurs d'ondes dans lesquels les spectres
de sortie sont discrétisés. On suppose ces intervalles successifs sont
également espacés. Commes les extrémités inférieures et supérieures
seront aussi considérées, il y aura donc 1+NB_LAMBDAS longueurs d'ondes
disctinctes dans les spectres.
"""
NB_LAMBDAS = 1000



r"""
\brief
Dictionnaire aidant au chargement des données étiquettées.

\note
L'ordre des attributs est assuré par l'emploi de clés numériques.
"""
# FIXME: vérifier les unités
# FIXME: vérifier les descriptions des attributs
dicoDesAttributs = {
		0:{
			 'symbole':'a',
			 'unites':'1/(m.m)',
			 'description':
				  "Coefficient de l'exposant de la gaussienne qui "
				  "module le profil d'accentuation des "
				  "corrugations causant le couplage; d'après [1] section 2.6 "
				  "son expression semble être : $\exp(-a*(z)^2)$, où $z=0$ "
				  "serait le plan de coupe central perpendiculaire "
				  "à l'axe du réseau.",
			 'domaine':'REELPOS',
			 'dimensions': 1,
		  },
		1:{
			 'symbole':'N',
			 'unites':'-',
			 'description':
				 "Nombre total de périodes du réseau; chacune "
				 "de ces périodes (ou franges) coincide également avec "
				 "une période de corrugation (perturbation) permettant "
				 "de générer du couplage.",
			 'domaine':'NATPOS',
			 'dimensions': 1,
		  },
		2:{
			 'symbole':'kappa',
			 'unites':'1/m',
			 'description':
				 "Valeur maximale de l'enveloppe de la gaussienne qui "
				 "module le profil d'accentuation des "
				 "corrugations causant le couplage; cette valeur "
				 "maximale survient au milieu du réseau.",
			 'domaine':'REELPOS',
			 'dimensions': 1,
		  },
		3:{
			 'symbole':'lambdaB',
			 'unites':'m',
			 'description':
				 "Longueur d'onde centrale du port *drop* dans le "
				 "coupleur contra-directionnel; correspond à $\lambda_D$ "
				 "dans l'article de Wei Shi [2] et le mémoire de "
				 "Jonathan St-Yves [1].",
			 'domaine':'REELPOS',
			 'dimensions': 1,
		  },
		4:{
			 'symbole':'apodization',
			 'unites':'1/m',
			 'description':
				  "Valeur de l'enveloppe d'apodisation "
				  "discrétisée à chaque extrémité des blocs d'analyse "
				  "employés par la méthode des matrices de transfert",
			 'domaine':'REELPOS',
			 'dimensions': 1+NB_BLOCS,
		  },
		5:{
			 'symbole':'period',
			 'unites':'1/m',
			 'description':
				 "Période [moyenne?] des franges du réseau "
				 "discrétisée à chaque extrémité des blocs d'analyse "
				 "employés par la méthode des matrices de transfert",
			 'domaine':'REELPOS',
			 'dimensions': 1+NB_BLOCS,
		  },
		6:{
			 'symbole':'real(E_drop)',
			 'unites':'V/m',
			 'description':
				  "Partie réelle "
				  "du champ électrique mesuré au port *drop* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
		  },
		7:{
			 'symbole':'imag(E_drop)',
			 'unites':'V/m',
			 'description':
				  "Partie imaginaire "
				  "du champ électrique mesuré au port *drop* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
		  },
}



def domaineValide(valeur,domaine):
	r"""
	\brief
	Valide l'appartenance d'une valeur à un domaine donné

	\note
	Pourrait être amélioré en considérant l'epsilon machine,
	mais nécessiterait alors de fixer la précision des nombres
	à virgules flottantes manipulés, par exemple:

		numpy.finfo(numpy.float32).eps = 1.1920929e-07
		numpy.finfo(numpy.float64).eps = 2.220446049250313e-16
		numpy.finfo(numpy.float128).eps = 1.084202172485504434e-19

	"""
	try:
		assert(domaine in DOMAINES)
		if domaine == 'NATPOS':
			if isinstance(valeur, float):
				assert(valeur%1 == 0)
				assert(valeur>0)
			elif isinstance(valeur, int):
				assert(valeur>0)
			else:
				raise
		if domaine == 'NAT':
			if isinstance(valeur, float):
				assert(valeur%1 == 0)
				assert(valeur>=0)
			elif isinstance(valeur, int):
				assert(valeur>=0)
			else:
				raise
		if domaine == 'REELPOS':
			assert(isinstance(valeur, float))
			assert(valeur>0)
		if domaine == 'REEL':
			assert(isinstance(valeur, float))
	except:
		return False
	return True





def decalageEnColonnes(attribut):
	r"""
	\brief
	Calcule le décalage en colonnes de la première (ou seule) valeur
	numérique d'un des attributs contenus dans les fichiers de données.
	"""
	try:
		assert(domaineValide(attribut,'NAT'))
		assert(attribut in dicoDesAttributs.keys())
	except:
		return None
	decalage = 0
	for i in range(attribut):
		decalage+=dicoDesAttributs[i]['dimensions']
	return decalage



# -----------------------------------------------------------------------------

print("IMPORTATION DES DONNÉES BRUTES")
donneesJSY=[]

for nomFichier in listdir(DIR_DONNEES_JSY):

	# chargement des données brutes
	if nomFichier.endswith(EXT_DONNEES_JSY):
		df0=pd.read_csv(
				path.join(DIR_DONNEES_JSY,nomFichier),
				sep='\s+',
				header=None,
				skiprows=1,
		)
		print("nomFichier,df0.shape = %s,%s"%(nomFichier,df0.shape))
		donneesJSY.append(df0)

# -----------------------------------------------------------------------------

print()
print("TRANSFORMATION EN VALEURS UTILISABLES")
dicoDesDonnees = {}

# pour chaque fichier de données chargé dans une DataFrame
nbDataFrames = len(donneesJSY)
compteurFichiers=0
idDonnee=0
for df in donneesJSY:
	compteurFichiers+=1
	# pour chaque rangée de la DataFrame
	print("--- n/%d = %d/%d"%(nbDataFrames,compteurFichiers,nbDataFrames))
	for i in range(len(df)):
		# récupérer cette rangée (ligne) de donnée brute
		ligne = df.iloc[i]
		# créer un dictionnaire ayant les mêmes clés principales
		# que dicoDesAttributs mais un seul champ numérique par attribut
		dicoDesValeurs = {}
		# le compléter avec les valeurs associées à chaque attribut
		for k in dicoDesAttributs.keys():
			# retenir le domaine de la valeur de cet attribut
			domaine = dicoDesAttributs[k]['domaine']
			# calculer le décalage en colonnes des valeurs
			decalage = decalageEnColonnes(k)
			# retenir le nombre de valeurs à charger
			dimensions = dicoDesAttributs[k]['dimensions']
			# si plus d'une dimension, employer un NumPy array
			# sinon, enregistrer la valeur directement
			if dimensions>1:
				# préparer un NumPy array
				plusieursValeurs = np.array([])
				# configurer son type numérique
				if domaine == 'REEL' or domaine == 'REELPOS':
					plusieursValeurs.astype(float)
				elif domaine == 'NAT' or domaine == 'NATPOS':
					plusieursValeurs.astype(int)
				else:
					raise

				for j in range(dimensions):
					# extraire la valeur
					valeur = ligne[j+decalage]
					# valider le domaine de la valeur
					assert(domaineValide(valeur,domaine))
					# ajouter cette valeur au NumPy array
					plusieursValeurs = np.append(plusieursValeurs,valeur)

				# ajouter une valeur correspondant à cet attribut
				dicoDesValeurs.update({k:plusieursValeurs})

			else:
				assert(dimensions==1) # check
				# extraire la valeur
				uneValeur = ligne[decalage]
				# valider le domaine de la valeur
				assert(domaineValide(uneValeur,domaine))
				# forcer l'emploi d'un entier si approprié
				if domaine == 'NAT' or domaine == 'NATPOS':
					uneValeur = int(uneValeur)
				# ajouter une valeur correspondant à cet attribut
				dicoDesValeurs.update({k:uneValeur})

		# ajouter le dictionnaire rempli pour cette donnée
		# à la liste des données
		dicoDesDonnees.update({idDonnee:dicoDesValeurs})
		idDonnee+=1


# -----------------------------------------------------------------------------

print()
print("ENREGISTREMENT EN FORMAT BINAIRE PICKLE")

dicoPourEnregistrer = {
		'lesAttributs':dicoDesAttributs,
		'lesValeurs':dicoDesDonnees,
}



with open(FIC_DONNEES_PKL, 'wb') as fichierPkl:
    pk.dump(dicoPourEnregistrer,fichierPkl,protocol=pk.HIGHEST_PROTOCOL)

with open(FIC_DONNEES_PKL, 'rb') as fichierPkl:
    dicoReCharge = pk.load(fichierPkl)

# checks
assert(dicoReCharge['lesAttributs'] == dicoPourEnregistrer['lesAttributs'])
for k1 in dicoReCharge['lesValeurs'].keys():
	for k2 in dicoReCharge['lesValeurs'][k1].keys():
		if dicoReCharge['lesAttributs'][k2]['dimensions']>1:
			assert(np.all(dicoReCharge['lesValeurs'][k1][k2]==
				 dicoPourEnregistrer['lesValeurs'][k1][k2]))
		else:
			assert(dicoReCharge['lesValeurs'][k1][k2]==
				  dicoPourEnregistrer['lesValeurs'][k1][k2])

# ménage mémoire
del dicoDesDonnees
del dicoPourEnregistrer
del dicoReCharge

# -----------------------------------------------------------------------------

print()
print("CRÉATION DES ÉCHANTILLONNAGES ÉTIQUETTÉS")


with open(FIC_DONNEES_PKL, 'rb') as fichierPkl:
    dicoDonnees = pk.load(fichierPkl)

# -----------------------------------------------------------------------------




def retrouverIdDunSymbole(dicoAttributs,symbole):
	nbAttributs = len(dicoAttributs.keys())
	i=0
	while dicoAttributs[i]['symbole']!=symbole and i<nbAttributs:
		i+=1
	if i<nbAttributs:
		return i
	return None

def creerEchantillonnage(dicoDonnees,symbX,symby,complexe=False):

	dicoAttributs = dicoDonnees['lesAttributs']
	dicoValeurs = dicoDonnees['lesValeurs']

	idDonnees = []
	idEtiquettes = []

	for sX in symbX:
		id_sX = retrouverIdDunSymbole(dicoAttributs,sX)
		assert( id_sX is not None )
		idDonnees.append(id_sX)

	for sy in symby:
		id_sy = retrouverIdDunSymbole(dicoAttributs,sy)
		assert( id_sy is not None )
		idEtiquettes.append(id_sy)

	X = np.array([])
	y = np.array([])

	# attention: les symboles hardcodés ne sont pas recommendables!
	sidRe = retrouverIdDunSymbole(dicoAttributs,'real(E_drop)')
	sidIm = retrouverIdDunSymbole(dicoAttributs,'imag(E_drop)')

	# note:
	# np.append n'est pas la meilleur façon;
	# préallouer l'espace serait mieux

	nbDonnees = len(dicoValeurs)
	for i in range(nbDonnees):

		dX = np.array([])
		if complexe:
			for re,im in zip(dicoValeurs[i][sidRe],dicoValeurs[i][sidIm]):
				dX = np.append(dX,complex(re,im))
		else:
			for sid in idDonnees:
				dX = np.append(dX,dicoValeurs[i][sid])
		X = np.append(X,dX)

		dy = np.array([])
		for sid in idEtiquettes:
			dy = np.append(dy,dicoValeurs[i][sid])
		y = np.append(y,dy)

	nbValeursX = int(len(X) / nbDonnees)
	nbValeursy = int(len(y) / nbDonnees)

	X=X.reshape((nbDonnees,nbValeursX))
	y=y.reshape((nbDonnees,nbValeursy))

	assert(X.ndim == 2)
	assert(y.ndim == 2)
	assert(X[0].ndim == 1)
	assert(y[0].ndim == 1)

	return X,y

	# ----------
	#
	# D'après la partie "III. Jeu de données" rédigée par Jonathan St-Yves
	# dans la proposition de projet, il conviendrait d'effectuer le
	# remodellage des données d'entrée pour en faire un échantillonnage
	#
	#	de t = 1..nbTotalDonnees dont les étiquettes sont
	#
	#	         r^t = [ a , N , kappa , lambdaB ]
	#
	# et dont les données sont
	#
	# . soit basées sur le champ électrique mesuré en sortie du port *drop*
	#
	#	    $\mathcal{X}_E$ = ( x_E^t, r^t )
	#
	#	         avec
	#
	# 	         x_E^t = [ real(E_drop) , imag(E_drop) ]
	#
	#            note: |x_E^t| = 2*(1+NB_LAMBDAS) composantes
	#


X1,y = creerEchantillonnage(
		dicoDonnees,
		['real(E_drop)','imag(E_drop)'],
		['a','N','kappa','lambdaB'])

	#
	#
	# . soit basées sur les paramètres géométriques du réseau
	#
	#	    $\mathcal{X}_G$ = ( x_G^t, r^t )
	#
	#	        avec
	#
	#	        x_G^t = [ apodization , period ]
	#
	#	        note: |x_G^t| = 2*(1+NB_BLOCS) composantes
	#


X2,_ = creerEchantillonnage(
		dicoDonnees,
		['apodization','period'],
		['a','N','kappa','lambdaB'])


	#
	# ----------
	#
    # Peut être sera-t-il nécessaire de combiner les deux selon
	#
	#	    $\mathcal{X}$ = ( x^t, r^t )
	#
	#	        avec
	#
	#	        x^t = [ apodization , period , real(E_drop) , imag(E_drop) ]
	#
	#	        note: |x^t| = 2*(2+NB_BLOCS+NB_LAMBDAS) composantes
	#


X3,_ = creerEchantillonnage(
		dicoDonnees,
		['real(E_drop)','imag(E_drop)','apodization','period'],
		['a','N','kappa','lambdaB'])

	#
	# ----------
	#
    # Un échantillonnage possible serait aussi E_drop complexe

X4,_ = creerEchantillonnage(
		dicoDonnees,
		['real(E_drop)','imag(E_drop)'],
		['a','N','kappa','lambdaB'],
		complexe=True)

	#
	# ----------
	#
	# Hammond [3]
	#
	# vérifier la soit-disant "compression" qu'il effectue
	# au Chapitre 3, en particulier à la section 3.2.1
	#
	# Figure 4.3: c'est exactement ce qu'on fait ici
	#
	#

# -----------------------------------------------------------------------------

print()
print("ENREGISTREMENT EN FORMAT PYTORCH")

import torch
ty = torch.from_numpy(y)
tX1 = torch.from_numpy(X1)
tX2 = torch.from_numpy(X2)
tX3 = torch.from_numpy(X3)
#tX4 = torch.from_numpy(X4)
#
# TypeError: can't convert np.ndarray of type numpy.complex128.
# The only supported types are:
# 	float64, float32, float16,
# 	int64, int32, int16, int8,
# 	uint8, and bool.
#
# une solution serait peut-être d'utiliser des packages tel
#
# 	http://wavefrontshaping.net/index.php/component/content/article
#	      /69-community/tutorials/others
#         /157-complex-valued-networks-with-pytorch-for-physics-applications
#
# ou bien
#
#	https://github.com/williamFalcon/pytorch-complex-tensor
#
# note:
# sinon, peut-etre vaudra-t-il mieux séparer complèterment la
# partie réelle de la partie imaginaire ???

torch.save(ty,'ty.pt')
torch.save(tX1,'tX1.pt')
torch.save(tX2,'tX2.pt')
torch.save(tX3,'tX3.pt')


from torch.utils.data import Dataset, DataLoader

class ProjetDataset(Dataset):
	r"""
	Cette classe ...
	Args:
		...

	\note
	Entre le {map|iterable}-style datasets, on implémente ici
	le map-style dataset dans lequel on doit redéfinir les
	méthodes __getitem__() et __len__()
	"""

	def __init__(self, mode=1):
		r"""
		"""
		super().__init__()
		# garde les paramètres en mémoire
		self.mode = mode
		# charger les données
		if mode == 1:
			self.data = torch.load('tX1.pt')
		if mode == 2:
			self.data = torch.load('tX2.pt')
		if mode == 3:
			self.data = torch.load('tX3.pt')
		self.targets = torch.load('ty.pt')

	def __getitem__(self, index):
		r"""
		"""
		return self.data[index]


	def __len__(self):
		r"""
		"""
		return len(self.data)



# -----------------------------------------------------------------------------

print()
print("CRÉATION D'UN DATASET")


# Creation du dataset
train_set = ProjetDataset(mode=1)
# Creation du dataloader d'entraînement
train_loader = DataLoader(train_set, batch_size=16)


# -----------------------------------------------------------------------------

print()
print("ILLUSTRATION DU DATASET CRÉÉ")


fig, subfigs = pyplot.subplots(2, 2, tight_layout=False)
# Affichage de données aléatoires
# l'affichage 2x2 suppose au moins 4 données disponibles
assert(len(train_set)>=4)
# on présentera 4 données distinctes en ordre d'indice croissant
idx = np.random.randint(0,len(train_set),size=1)
while len(idx)<4:
        i = np.random.randint(0,len(train_set))
        if not i in idx:
                idx = np.append(idx,i)
idx.sort()
for i,subfig in zip(idx,subfigs.reshape(-1)):
        subfig.scatter(np.arange(len(train_set[i])),train_set[i])
        subfig.set_title("(i,y)=(%g,%s)"
						  %(i,train_set.targets[i]))
        subfig.set_axis_off()
fig.suptitle('Affichage de quatre données aléatoires\n'\
			 '(i,y)=(indice,étiquettes)')



# -----------------------------------------------------------------------------
# EOF