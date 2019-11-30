#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""

pyfortin -- 20191129

loader pour pytorch
attention: un peu nettoyé mais encore brouillon
sera amélioré d'ici rencontre lundi 2 décembre
sera amélioré selon feedbacks de l'équipe


TODO: utiliser données à JC aussi
TODO: illustre des diagnostics graphiques sur le Dataset global 	ex: mu,std (X,y) , histogrammes , ...
TODO: faire une dox d'en-tête de fichier appropriée (\file ...)
TODO: mieux séparer les échantillons (ex: Re vs Im)
TODO: modulariser et documenter script
TODO: améliorer nommage dico*
TODO: améliorer rooting du main (args?)



==========================================
Description des fichiers de données de JSY
==========================================

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
  title={Contra-directional couplers as optical filters
		   on the silicon on insulator platform},
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
  title={Machine Learning Methods for Nanophotonic
		   Design, Simulation, and Operation},
  author={Hammond, Alec Michael},
  year={2019},
  school={Brigham Young University},
}


=========
Questions
=========

???: Quel-est le champ en entrée (port *input*) qui cause le champ E_drop
	en sortie (port *drop*)? Il devrait être le même pour toutes les données,
	mais quel est-il?

???: Quelle-est la longueur d'onde centrale du champ d'entrée -- la
	même que le port *through* : était-ce 1550 nm ?

???: Est-ce que lambdaB est la longueur d'onde centrale du port *drop*?
	Est-ce qu'elle correspond à $\lambda_D$ dans l'article de Wei Shi [2]
	et dans le mémoire de Jonathan St-Yves [1]? Si oui, faudrait peut-être
	la renommer ainsi; sinon, pourquoi l'indice "B"?

???: Est-ce que la période donnée à chaque extrémité des blocs d'analyse
	de la méthode par matrice de transfert correspond à une moyenne
	locale de la période à cette position? Sinon, à quoi correspond-elle?

???: Est-ce une bonne idée de convertir les valeurs de "N" en int?
	Ne serait-il pas plus naturel de les laisser en float (REELPOS)?
	Je pense qu'il vaut mieux le forcer en int pour être en mesure de
	facilement le feeder à scikit-learn, tensorflow ou autre, quitte
	à le reconvertir en float pour les np.array --> tensor de pytorch.




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
	#
	# ----------
	#
    # Un échantillonnage possible serait aussi E_drop complexe
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



# Lorsque
#
#	type(X) == complex
#
# alors
#
# 	torch.from_numpy(X)
#
# chiâle:
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


"""

# TODO: purifier en évitant les as et le from ...
import numpy as np
import pandas as pd
import pickle as pk
from os import path, listdir
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot
import sys # exit
import os # mkdirs


pd.set_option('display.max_columns',8)
pd.set_option('display.max_rows',8)
pd.set_option('display.width',40)





r"""
\var DIR_DONNEES_JC

\brief
Répertoire contenant données brutes générées par
code python de Jonathan Cauchon.
"""
DIR_DONNEES_JC = '../jc/donnees'


r"""
\var EXT_DONNEES_JC

\brief
Extention fichiers de données brutes générées par
code python de Jonathan Cauchon.
"""
EXT_DONNEES_JC = '.txt'


r"""
\var DIR_DONNEES_JSY

\brief
Répertoire contenant données brutes générées par
code matlab de Jonathan St-Yves.
"""
DIR_DONNEES_JSY = '../jsy/donnees'


r"""
\var EXT_DONNEES_JSY

\brief
Extention fichiers de données brutes générées par
code matlab de Jonathan St-Yves.
"""
EXT_DONNEES_JSY = '.txt'


r"""
\var DIR_DONNEES_STRUCTUREES

\brief
Répertoire d'enregistrement des données structurées par ce script.
"""
DIR_DONNEES_STRUCTUREES = '../structurees'


r"""
\var NOM_FICHIER_PICKLE

\brief
Nom du fichier pickle contenant toutes les données structurées.
"""
NOM_FICHIER_PICKLE = 'donneesStructurees.pickle'


r"""
\var NOM_FICHIER_PYTORCH_DONNEES

\brief
Nom du fichier pytorch contenant les donnees de l'échantillonnage.
"""
NOM_FICHIER_PYTORCH_DONNEES = 'donnees.pt'


r"""
\var NOM_FICHIER_PYTORCH_ETIQUETTES

\brief
Nom du fichier pytorch contenant les étiquettes de l'échantillonnage.
"""
NOM_FICHIER_PYTORCH_ETIQUETTES = 'etiquettes.pt'



r"""
\var DOMAINES

\brief
Domaines possibles pour l'appartenance des valeurs numériques des attributs.
"""
DOMAINES = ['REEL','REELPOS','NAT','NATPOS']


r"""
\var NB_BLOCS

\brief
Nombre de segments égaux dans lesquels sont analysés le réseau.

\details
Nombre de blocs successifs de longueurs physiques (m) égales dans
lesquels on décompose le réseau afin d'en effectuer l'analyse par méthode
de matrice de transfert. Comme les positions des extrémités inférieures
et supérieures de l'union de ces blocs seront aussi considérées, il y
aura donc 1+NB_BLOCS positions dans les profils.
"""
NB_BLOCS = 100


r"""
\var NB_LAMBDAS

\brief
Nombre de segments égaux de 1 nm dans le domaine des longueurs d'ondes.

\details
Nombre d'intervalles de longueurs d'ondes (m) dans lesquels les spectres
de sortie sont discrétisés. On suppose ces intervalles successifs mesurent
tous $10^{-9}$ (m). Commes les extrémités inférieures et supérieures de
l'union de ces intervalles seront aussi considérées, il y aura
donc 1+NB_LAMBDAS longueurs d'ondes disctinctes dans les spectres.
"""
NB_LAMBDAS = 1000


r"""
\var dicoDesAttributs

\brief
Dictionnaire aidant au chargement des données étiquettées.

\note
La maintenance de l'ordre des attributs est assurée par
l'emploi de clés numériques entières débutant par zéro.

\note
Permet d'afficher la documentation avec une interface gui.

\note
Permet de facilement adapter le processus de chargement à une modification
du format des données brutes.

\warning
Les unités et les descriptions doivent être vérifiées par les experts.
"""
# FIXME: svp vérifier les unités
# FIXME: svp vérifier les descriptions des attributs
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
	Valide l'appartenance d'une valeur à un domaine donné.

	\param[in] valeur
	Valeur numérique dont on veut vérifier l'appartenance à un domaine;
	les glyphes de la valeur peuvent être contenues dans une chaîne de
	caractères, mais la chaîne doit alors être convertible en valeur
	numérique.

	\param[in] valeur
	Domaine numérique auquel la valeur doit appartenir;
	doit faire partie d'une liste pré-définie de domaines.

	\return
	Vrai seulement si valeur appartient au domaine; faux sinon.

	\note
	Pourrait être amélioré pour vérifier la proximité de zéro des
	nombres de faible grandeur en considérant l'epsilon machine.
	Mais cela nécessiterait alors de fixer la précision des nombres
	à virgules flottantes manipulés. Par exemple:

		numpy.finfo(numpy.float32).eps = 1.1920929e-07
		numpy.finfo(numpy.float64).eps = 2.220446049250313e-16
		numpy.finfo(numpy.float128).eps = 1.084202172485504434e-19

	et ces valeurs dépendent de la machine employée.
	"""
	try:
		assert(domaine in DOMAINES)
		if type(valeur) == str:
			assert(valeur.isnumeric())
			valeur = float(valeur)
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


def decalageEnColonnes(idAttribut):
	r"""
	\brief
	Calcule le décalage en colonnes de la valeur d'un attribut.

	\param[in] idAttribut
	Clé numérique entière identifiant l'attribut.

	\return
	Décalage en colonnes de la première (ou seule) valeur numérique
	appartenant à l'attribut demandé dans les fichiers de données brutes.
	None si idAttribut ne correspond pas à un identifiant numérique valide.

	\note
	La première colonne, celle la plus à gauche sur une rangée donnée,
	est numérotée 0; la seconde est numérotée 1, ...
	"""
	try:
		assert(domaineValide(idAttribut,'NAT'))
		assert(idAttribut in dicoDesAttributs.keys())
	except:
		return None
	decalage = 0
	for i in range(idAttribut):
		decalage+=dicoDesAttributs[i]['dimensions']
	return decalage




def importerDonneesBrutes():
	r"""
	\brief
	Lit fichiers de données brutes et récolte valeurs dans dataFrames.

	\return
	Liste de dataFrames, une pour chaque fichier lu.
	"""
	print("IMPORTATION DES DONNÉES BRUTES")
	donneesJSY=[]
	for nomFichier in listdir(DIR_DONNEES_JSY):

		if nomFichier.endswith(EXT_DONNEES_JSY):
			df0=pd.read_csv(
					path.join(DIR_DONNEES_JSY,nomFichier),
					sep='\s+',
					header=None,
					skiprows=1,
			)
			print("nomFichier,df.shape = %s,%s"%(nomFichier,df0.shape))
			donneesJSY.append(df0)

	return donneesJSY



def refondreDonnees(listeDeDataFrames):
	r"""
	\brief
	Transforme la liste de dataFrames de données brutes en un
	seul dictionnaire de données structurées.

	\return
	Dictionnaire de données structurées; les clés principales sont
	des entiers naturels successifs (un par donnée distincte); à chaque
	clé principale est attaché un sous-dictionnaire de valeurs; les
	clés de ce sous-dictionnaires coorespondent aux identifiants des
	différents attributs provenant des données brutes.

	\note
	Selon la quantité de dimensions des attributs, les valeurs contenues
	dans les sous-dictionnaires sont soit des nombres (si la quantité de
	dimensions est 1), soit des 	numpy.array de nombres (si la quantité
	de dimensions est plus de 1).
	"""
	print("REFONTE DES DONNÉES DANS STRUCTURE UNIFIÉE")
	dicoDesDonnees = {}

	# pour chaque fichier de données chargé dans une DataFrame
	nbDataFrames = len(listeDeDataFrames)
	compteurFichiers=0
	idDonnee=0
	for df in listeDeDataFrames:
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

	return dicoDesDonnees


def enregistrementPickle(dicoDesDonnees):
	r"""
	\brief
	Enregistre un dictionnaire de données structurées dans un fichier binaire.

	\details
	Le dictionnaire enregistré consiste en deux dictionnaires distincts
	mais inter-reliés: le premier contient la description des attributs
	et le second contient des valeurs associées à chaque attribut, groupées
	dans leur propre dictionnaire, un par donnée enregistrée.

		d = {d1:{},d2:{}}

			P = nbAttributs - 1
			Q = nbTotalDonnees - 1

		d1 = {0:da0,1:da1,...,P:daP} <--- cf. dicoDesAttributs
		d2 = {0:dv0,2:dv1,...,Q:dvQ}

			da<p>.keys() = ['symbole','unites','description',
							 'domaine','dimensions']

			dv<q>.keys() = [0,1,...,P]

	\post
	Les dictionnaires d'attributs et de valeurs ont été enregistrés
	dans un fichier binaire en format Pickle. Leur re-chargement a été
	testé et procure les mêmes objets.

	\note
	Un test d'intégrité est effectué afin de vérifier que les valeurs
	peuvent être rechargées et qu'elles sont alors les mêmes que celles
	existant avant l'enregistrement.
	"""
	print("ENREGISTREMENT EN FORMAT BINAIRE PICKLE")

	dicoPourEnregistrer = {
			'lesAttributs':dicoDesAttributs,
			'lesValeurs':dicoDesDonnees,
	}

	if not path.exists(DIR_DONNEES_STRUCTUREES):
		os.makedirs(DIR_DONNEES_STRUCTUREES)

	nomFichier = path.join(DIR_DONNEES_STRUCTUREES,NOM_FICHIER_PICKLE)

	with open(nomFichier, 'wb') as fichierPkl:
	    pk.dump(dicoPourEnregistrer,fichierPkl,protocol=pk.HIGHEST_PROTOCOL)

	# test de rechargement
	dicoReCharge = chargementPickle(silencieux=True)

	# test d'intégrité
	assert(dicoReCharge['lesAttributs'] ==
		dicoPourEnregistrer['lesAttributs'])

	for k1 in dicoReCharge['lesValeurs'].keys():
		for k2 in dicoReCharge['lesValeurs'][k1].keys():
			if dicoReCharge['lesAttributs'][k2]['dimensions']>1:
				assert(np.all(dicoReCharge['lesValeurs'][k1][k2]==
					 dicoPourEnregistrer['lesValeurs'][k1][k2]))
			else:
				assert(dicoReCharge['lesValeurs'][k1][k2]==
					  dicoPourEnregistrer['lesValeurs'][k1][k2])


def chargementPickle(silencieux=False):
	r"""
	\brief
	Recharge dictionnaire de données structurées depuis fichier binaire.

	\param[in] silencieux
	Si vrai, alors ne pas imprimer de message; si faux, alors
	imprimer message qui avertit l'usager que la recharge depuis
	fichier binaire est effectuée.

	\return
	Dictionnaire de données structurées dont la structure est
	définie dans la documentation de enregistrementPickle().
	"""
	if not silencieux:
		print("CHARGEMENT DEPUIS FORMAT BINAIRE PICKLE")
	nomFichier = path.join(DIR_DONNEES_STRUCTUREES,NOM_FICHIER_PICKLE)

	with open(nomFichier, 'rb') as fichierPkl:
	    dicoDonneesRechargees = pk.load(fichierPkl)

	return dicoDonneesRechargees



def retrouverIdDunSymbole(symbole):
	r"""
	\brief
	Retrouve l'identifiant numérique de l'attribut relié à un symbole.

	\param[in] symbole
	Chaîne de caractères qui identifie un attribut.

	\return
	Identifiant numérique de l'attribut ayant le symbole demandé;
	None s'il n'y a pas eu de correspondance.
	"""
	nbAttributs = len(dicoDesAttributs.keys())
	i=0
	while dicoDesAttributs[i]['symbole']!=symbole and i<nbAttributs:
		i+=1
	if i<nbAttributs:
		return i
	return None



def creerEchantillonnage(dicoValeurs,symbX,symby,complexe=False):
	r"""
	\brief
	Crée un échantillonnage de données et d'étiquettes.

	\param[in] dicoValeurs
	Dictionnaire contenant uniquement des *VALEURS* pour
	chaque attribut et ce pour toutes les données disponibles.

	\param[in] symbX
	Liste des symboles d'attributs choisis pour constituer
	les données d'entrée.

	\param[in] symby
	Liste des symboles d'attributs choisis pour constituer
	les étiquettes des données d'entrée.

	\param[in] complexe
	Drapeau indiquant si on doit combiner les deux seuls attributs
	contenus dans les symbX entre eux pour former une seule
	valeur complexe.

	\return
	Tuple (X,y) de deux numpy.array contenant en X les valeurs de
	données d'entrée et en y les valeurs des étiquettes reliées
	à ces données d'entrée. La forme de X et de y est respectivement
	(nbTotalDonnees,nbValeurs{X|y}) où nbValeursX est le nombre de
	valeurs numériques distinctes contenues 	dans chaque donnée X et
	nbValeursy est le nombre de valeurs 	numériques distinctes
	contenues dans chaque étiquette y.

	\warning
	Itérer avec la méthode numpy.append n'est pas optimal.
	Si on connaît les dimensions, préallouer l'espace est
	plus rapide.
	"""
	idDonnees = []
	idEtiquettes = []

	for sX in symbX:
		id_sX = retrouverIdDunSymbole(sX)
		assert( id_sX is not None )
		idDonnees.append(id_sX)

	for sy in symby:
		id_sy = retrouverIdDunSymbole(sy)
		assert( id_sy is not None )
		idEtiquettes.append(id_sy)

	X = np.array([])
	y = np.array([])


	if complexe:
		# il ne doit y avoir que deux attributs dans symbX
		assert(len(symbX)==2)
		# l'attribut réel sera le premier symbX
		sidRe = retrouverIdDunSymbole(symbX[0])
		# l'attribut imaginaire sera le second symbX
		sidIm = retrouverIdDunSymbole(symbX[-1])


	#FIXME: préallouer l'espace au-lieu d'itérer des numpy.append

	nbTotalDonnees = len(dicoValeurs)
	for i in range(nbTotalDonnees):

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

	nbValeursX = int(len(X) / nbTotalDonnees)
	nbValeursy = int(len(y) / nbTotalDonnees)

	X=X.reshape((nbTotalDonnees,nbValeursX))
	y=y.reshape((nbTotalDonnees,nbValeursy))

	# vérifications de la consistance de la forme des numpy.array
	assert(X.ndim == 2)
	assert(y.ndim == 2)
	assert(X[0].ndim == 1)
	assert(y[0].ndim == 1)

	return X,y


def enregistrementPyTorch(X,y):
	r"""
	\brief
	Enregistre les données et leurs étiquettes en tenseurs PyTorch.

	\param[in] X
	NumPy array des données.

	\param[in] y
	NumPy array des étiquettes.

	\post
	Les données X et les étiquettes y ont été convertis en
	tenseurs PyTorch et enregistrées dans un format directement
	utilisable par PyTorch. Leur re-chargement a été
	testé et procure les mêmes objets.
	"""
	assert(type(X)!=complex)

	print("ENREGISTREMENT EN FORMAT PYTORCH")

	if not path.exists(DIR_DONNEES_STRUCTUREES):
		os.makedirs(DIR_DONNEES_STRUCTUREES)

	nomFichierX = path.join(
			DIR_DONNEES_STRUCTUREES,
			NOM_FICHIER_PYTORCH_DONNEES)

	nomFichiery = path.join(
			DIR_DONNEES_STRUCTUREES,
			NOM_FICHIER_PYTORCH_ETIQUETTES)

	tX = torch.from_numpy(X)
	ty = torch.from_numpy(y)

	torch.save(tX,nomFichierX)
	torch.save(ty,nomFichiery)

	# checks

	rtX = torch.load(nomFichierX)
	rty = torch.load(nomFichiery)
	assert(torch.all(torch.eq(rtX,tX)))
	assert(torch.all(torch.eq(rty,ty)))


class EchantillonnagePourProjet(Dataset):
	r"""
	\brief
	Classe qui instancie des objets dérivés de torch.utils.data.Dataset

	\details
	Les objets de la classe EchantillonnagePourProjet sont spécialisés
	pour lire les fichiers de données créés pour PyTorch par ce script.
	Les valeurs brutes provenant des simulations de réseaux contra-
	directionnels par matrice de transfert ont été restructurées
	et manipulées pour être séparées en données X et en étiquettes y.

	\note
	Entre le {map|iterable}-style datasets, on implémente ici
	le map-style dataset dans lequel on doit redéfinir les
	méthodes __getitem__() et __len__()
	"""

	def __init__(self):
		r"""
		"""
		# appeler contructeur de la classe de base (Dataset)
		super().__init__()

		# définir en quoi consiste le constructeur
		# de cette classe dérivée (ProjetDataset)

		nomFichierX = path.join(
			DIR_DONNEES_STRUCTUREES,
			NOM_FICHIER_PYTORCH_DONNEES)

		nomFichiery = path.join(
			DIR_DONNEES_STRUCTUREES,
			NOM_FICHIER_PYTORCH_ETIQUETTES)

		# charger les données
		self.X = torch.load(nomFichierX)
		# charger les étiquettes
		self.y = torch.load(nomFichiery)

		# check
		assert(len(self.X)==len(self.y))
		# enregistrer le nombre total de donnees
		self.nbTotalDonnes = self.__len__


	def __getitem__(self, i):
		r"""
		"""
		return self.X[i]


	def __len__(self):
		r"""
		"""
		return len(self.X)



def illustrerEchantillonnage(chi):
	r"""
	\brief
	Trace le graphique de quatre données d'entrée tirées au hasard.

	\param[in] chi
	Échantillonnage global, instance de la classe EchantillonnagePourProjet.

	\post
	Une figures avec 4 sous-graphiques est tracée sur laquelle
	on a les indices i des données X tracées ainsi que la valeur de
	leurs étiquettes y.
	"""

	print("ILLUSTRATION DE L'ÉCHANTILLONNAGE")


	fig, subfigs = pyplot.subplots(2, 2, tight_layout=False)
	# Affichage de données aléatoires
	# l'affichage 2x2 suppose au moins 4 données disponibles
	assert(len(chi)>=4)
	# on présentera 4 données distinctes en ordre d'indice croissant
	idx = np.random.randint(0,len(chi),size=1)
	while len(idx)<4:
	        i = np.random.randint(0,len(chi))
	        if not i in idx:
	                idx = np.append(idx,i)
	idx.sort()
	for i,subfig in zip(idx,subfigs.reshape(-1)):
	        subfig.scatter(np.arange(len(chi[i])),chi[i])
	        subfig.set_title("(i,y)=(%g,%s)"
							  %(i,chi.y[i]))
	        subfig.set_axis_off()
	fig.suptitle('Affichage de quatre données aléatoires\n'\
				 '(i,y)=(indice,étiquettes)')



def main():
	r"""
	\brief
	Simule la chaîne de processus consistant à

		- importer les données brutes,
		- refondre les données dans une structure unifiée,
		- enregistrer la refonte en Pickle,
		- créer un échantillonnage,
		- enregistrer l'échantillonnage en tenseurs PyTorch,
		- instancier un objet dérivé de torch.utils.data.Dataset
		- illustrer des échantillons depuis l'objet dérivé de Dataset


	"""


	#TODO: ajouter condition sur présence du fichier pickle
	donneesBrutes = importerDonneesBrutes()
	dicoDesDonnees = refondreDonnees(donneesBrutes)
	enregistrementPickle(dicoDesDonnees)
	#dicoDonneesRechargees = chargementPickle()
	#dicoDesDonnees = dicoDonneesRechargees['lesValeurs']

	symbX = [['real(E_drop)','imag(E_drop)'],
		   ['apodization','period'],
			['real(E_drop)','imag(E_drop)','apodization','period']]

	symby = ['a','N','kappa','lambdaB']


	X,y = creerEchantillonnage(
			dicoDesDonnees,symbX[0],symby,complexe=False)

	enregistrementPyTorch(X,y)

	# Creation de l'échantillonnage (données X et étiquettes y)
	chi = EchantillonnagePourProjet()
	# Creation du chargeur d'entraînement
	chargeur = DataLoader(chi, batch_size=16)

	illustrerEchantillonnage(chi)




if __name__=='__main__':
	main()
else:
	sys.exit(0)


# -----------------------------------------------------------------------------
# EOF