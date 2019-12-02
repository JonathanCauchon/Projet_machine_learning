#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
\file chargeurPyTorch.py


\author pyfortin


\date 2019.12.02


\version 1.0 (celle présentée à l'équipe le 2 Décembre)


\brief
Paréparateur de données brutes pour chargement PyTorch


\details
Sera amélioré ensuite selon feedbacks de l'équipe


\todo
TODO: illustrer diagnostics des "y" du Dataset: mu, std, histogrammes , ...


\todo
TODO: normaliser balises et générer documentation.


\todo
TODO: Vérifier la soi-disant "compression" que Hammond [3] effectue
au Chapitre 3, en particulier à la section 3.2.1. Noter que
la Figure 4.3 illustre exactement ce qu'on tente de faire ici.


\note
{
	Description des fichiers de données
	===================================

	.en concaténant tous les fichiers de données on totalise nbTotalDonnees
	.chaque fichier de données contient

		nbDonneesJC = selon argument de ChirpedContraDC.createDataSet
		nbDonneesJSY = 2, 10 ou 20
		nbLignes = 1+nbDonnees
		nbAttributsJC = 9
		nbAttributsJSY = 8

	.la 1ere ligne décrit les nbAttributs séparés par des virgules
	.les nbDonnees* lignes suivantes contiennent les valeurs des attributs
	.certains attributs sont en plusieurs dimensions
	.les attributs multi-dimensionnels sont enregistrés sur plus d'une colonne
	.pour chaque donnée, les valeurs d'attributs sont séparées par des espaces

	utilisons des indices d'attributs débutant à 0
	utilisont des indices de colonnes débutant à 0

	ainsi, le décalage en colonnes de la première -- ou seule -- valeur
	numérique rattachée à chaque attribut sera donnée par

		$\delta_i = \sum_{j=0}^{i-1}{d_\kappa(j)}$

	(a): colonne $i$ de la 1ere ligne de donnée brute (en-tête)
	(b): idAttribut de cette colonne $\kappa(i)$
	(c): symbole texte $\sigma_{\kappa(i)}$ relié à cet attribut
	(d): type $\tau_{\kappa(i)}$ relié à cet attribut
	(e): domaine $\rau_{\kappa(i)}$ relié à cet attribut
	(f): dimensions $d_{\kappa(i)}$ relié à cet attribut
	(g): décalage $\delta_i$ en nombre de colonnes pour les lignes de valeurs

	Données JC
	----------------------------------------------------------------------
	(a) (b) (c)            (d)        (e)       (f)tot=4209   (g)
	----------------------------------------------------------------------
	0   0   a              float      R+        1             0
	1   1   N              int        N*        1             1
	2   2   kappa          float      R+*       1             2
	3   4   apodization    float      R+        101           3
	4   5   period         float      R+*       101           104
	5   6   real(E_drop)   float      R         1001          205
	6   7   imag(E_drop)   float      R         1001          1206
	7   8   real(E_thru)   float      R         1001          2207
	8   9   imag(E_thru)   float      R         1001          3208
	----------------------------------------------------------------------

	Données JSY
	----------------------------------------------------------------------
	(a) (b) (c)            (d)        (e)       (f)tot=2208   (g)
	----------------------------------------------------------------------
	0   0   a              float      R+        1             0
	1   1   N              int        N*        1             1
	2   2   kappa          float      R+*       1             2
	3   3   lambdaB        float      R+*       1             3
	4   4   apodization    float      R+        101           4
	5   5   period         float      R+*       101           105
	6   6   real(E_drop)   float      R         1001          206
	7   7   imag(E_drop)   float      R         1001          1207
	----------------------------------------------------------------------

	On distingue l'ensemble des réels R de l'ensemble R+ des réels positifs
	incluant zéro et de l'ensemble R+* des réels exclusivement positifs.

	On distingue l'ensemble des naturels N incluant le zéro de
	l'ensemble N* des naturels exclusivement positifs.
}


\note
{
	D'après la partie "III. Jeu de données" rédigée par Jonathan St-Yves
	dans la proposition de projet, il conviendrait d'effectuer le
	remodellage des données d'entrée pour en faire un échantillonnage
	de t = 1..nbTotalDonnees dont les étiquettes sont

		r^t = [ a , N , kappa , lambdaB ]

	et dont les données sont soit basées sur le champ électrique
	mesuré en sortie du port *drop*

		$\mathcal{X}_E$ = ( x_E^t, r^t )

	avec

		x_E^t = [ real(E_drop) , imag(E_drop) ]
		||x_E^t|| = 2*(1+NB_LAMBDAS) <--- cardinalité

	soit basées sur les paramètres géométriques du réseau

		$\mathcal{X}_G$ = ( x_G^t, r^t )

	avec

		x_G^t = [ apodization , period ]
		||x_G^t|| = 2*(1+NB_BLOCS) <--- cardinalité

	Peut être sera-t-il nécessaire de combiner les deux selon

		$\mathcal{X}$ = ( x^t, r^t )

	avec

		x^t = [ apodization , period , real(E_drop) , imag(E_drop) ]

	Un échantillonnage possible serait aussi E_drop complexe
	Ou bien peut-être vaudra-t-il mieux séparer complètement la
	partie réelle de la partie imaginaire...
}


\warning
{
	Lorsque `type(X) == complex` alors `torch.from_numpy(X)` chiâle:

	\verbatim
	TypeError: can't convert numpy.ndarray of type numpy.complex128. The only
	supported types are: float{64|32|16}, int{64|32|16|8}, uint8, and bool.
	\endverbatim

	Une solution serait peut-être d'utiliser des modules externes
	tels que

	<a href="http://wavefrontshaping.net/index.php/component
	/content/article/69-community/tutorials/others
	/157-complex-valued-networks-with-pytorch-for-physics-applications">
	`complexPyTorch`</a>

	ou bien

	<a href="https://github.com
	/williamFalcon/pytorch-complex-tensor">`pytorch-complex-tensor`</a>.
}
"""

r"""
references.bib[debut]----------------------------------------------------------
% ici, lien naïf [1]
@mastersthesis{ st2017contra,
  title={Contra-directional couplers as optical filters
		   on the silicon on insulator platform},
  author={St-Yves, Jonathan},
  year={2017},
  school={Université Laval},
}


% ici, lien naïf [2]
@article{ shi2013silicon,
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

% ici, lien naïf [3]
@mastersthesis{ hammond2019machine,
  title={Machine Learning Methods for Nanophotonic
		   Design, Simulation, and Operation},
  author={Hammond, Alec Michael},
  year={2019},
  school={Brigham Young University},
}
references.bib[fin]------------------------------------------------------------
"""


# ------------------ # --------------------------------------------------------
# imports            # méthodes et sous-modules et utilisés
# ------------------ # --------------------------------------------------------

import pandas        # read_csv

import pickle        # dump, load, HIGHEST_PROTOCOL

import numpy         # arange, array, reshape, ndim, append,
                     # all, sort, astype, random.randint

import torch         # from_numpy, save, load, eq, all,
                     # utils.data.{Dataset, DataLoader}

import matplotlib    # pyplot.subplots
                     # figure.Figure.suptitle
					 # axes.Axes.{set_title,set_axis_off,scatter}

import os            # listdir, makedirs,
                     # path.{normpath, exists, join}

import sys           # exit

import argparse      # ArgumentParser, RawDescriptionHelpFormatter

import textwrap      # indent



#DIR_JC = '../jc/donnees'
DIR_JC = '../jc/d.petit'
r"""
\var DIR_JC

\brief
Répertoire contenant données brutes générées par
code python de Jonathan Cauchon.
"""


EXT_JC = '.txt'
r"""
\var EXT_JC

\brief
Extention fichiers de données brutes générées par
code python de Jonathan Cauchon.
"""


ATTRIBUTS_JC = [0,1,2,4,5,6,7,8,9]
r"""
\var ATTRIBUTS_JC

\brief
Identifiants des attributs des données brutes générées par
code python de Jonathan Cauchon.
"""



#DIR_JSY = '../jsy/donnees'
DIR_JSY = '../jsy/d.petit'
r"""
\var DIR_JSY

\brief
Répertoire contenant données brutes générées par
code matlab de Jonathan St-Yves.
"""


EXT_JSY = '.txt'
r"""
\var EXT_JSY

\brief
Extention fichiers de données brutes générées par
code matlab de Jonathan St-Yves.
"""


ATTRIBUTS_JSY = [0,1,2,3,4,5,6,7]
r"""
\var ATTRIBUTS_JSY

\brief
Identifiants des attributs des données brutes générées par
code matlab de Jonathan St-Yves.
"""


DIR_STRUCT = '../struct'
r"""
\var DIR_STRUCT

\brief
Répertoire d'enregistrement des données structurées par ce script.
"""


NOM_PICKLE = 'dStruct.pickle'
r"""
\var NOM_PICKLE

\brief
Nom du fichier pickle contenant toutes les données structurées.
"""


ATTRIBUTS_X = set([6])
r"""
\var ATTRIBUTS_X

\brief
Attributs par défaut choisis pour constituer les données d'entrée.
"""


ATTRIBUTS_y = set([0,1,2])
r"""
\var ATTRIBUTS_y

\brief
Attributs par défaut reliés aux données d'entrée.
"""

NOM_PYTORCH_X = 'donnees.pt'
r"""
\var NOM_PYTORCH_X

\brief
Nom du fichier pytorch contenant les donnees de l'échantillonnage.
"""


NOM_PYTORCH_y = 'etiquettes.pt'
r"""
\var NOM_PYTORCH_y

\brief
Nom du fichier pytorch contenant les étiquettes de l'échantillonnage.
"""


DOMAINES = ['REEL','REELPOS','REELPOS*','NAT','NAT*']
r"""
\var DOMAINES

\brief
Domaines possibles pour l'appartenance des valeurs numériques des attributs.
"""


NB_BLOCS = 100
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


NB_LAMBDAS = 1000
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


# FIXME, TODO, ??? : svp vérifier les unités
# FIXME, TODO, ??? : svp vérifier les descriptions des attributs
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
			 'usage':'y',
		  },
		1:{
			 'symbole':'N',
			 'unites':'-',
			 'description':
				 "Nombre total de périodes du réseau; chacune "
				 "de ces périodes (ou franges) coincide également avec "
				 "une période de corrugation (perturbation) permettant "
				 "de générer du couplage.",
			 'domaine':'NAT*',
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
			 'domaine':'REELPOS*',
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
			 'domaine':'REELPOS*',
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
			 'domaine':'REELPOS*',
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
		8:{
			 'symbole':'real(E_thru)',
			 'unites':'V/m',
			 'description':
				  "Partie réelle "
				  "du champ électrique mesuré au port *thru* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
		  },
		9:{
			 'symbole':'imag(E_thru)',
			 'unites':'V/m',
			 'description':
				  "Partie imaginaire "
				  "du champ électrique mesuré au port *thru* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
		  },
}
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
	Pourrait être amélioré pour vérifier la proximité avec zéro des
	nombres de faible grandeur en considérant l'epsilon machine.
	Mais cela nécessiterait alors de fixer la précision des nombres
	à virgules flottantes manipulés. Par exemple, pour
	une machine donnée:
		numpy.finfo(numpy.float32).eps = 1.1920929e-07
		numpy.finfo(numpy.float64).eps = 2.220446049250313e-16
		numpy.finfo(numpy.float128).eps = 1.084202172485504434e-19
	"""
	try:
		assert(domaine in DOMAINES)
		if type(valeur) == str:
			assert(valeur.isnumeric())
			valeur = float(valeur)
		if domaine == 'NAT*':
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
		if domaine == 'REELPOS*':
			assert(isinstance(valeur, float))
			assert(valeur>0)
		if domaine == 'REELPOS':
			assert(isinstance(valeur, float))
			assert(valeur>=0)
		if domaine == 'REEL':
			assert(isinstance(valeur, float))
	except:
		print("!!! Erreur de domaineValide "
			"(valeur,domaine) = (%s,%s)"
			%(valeur,domaine))
		return False
	return True


def decalageEnColonnes(idAttribut,listeDesAttributs):
	r"""
	\brief
	Calcule le décalage en colonnes de la valeur d'un attribut.

	\param[in] idAttribut
	Clé numérique entière identifiant l'attribut.

	\param[in] listeDesAttributs
	Liste ordonnée des attributs présents dans le fichier de données.

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
		assert(idAttribut in listeDesAttributs)
		assert(idAttribut in dicoDesAttributs.keys())
	except:
		print("!!! Erreur de calcul de décalage "
			"(idAttribut,listeDesAttributs) = (%s,%s)"
			%(idAttribut,listeDesAttributs))
		return None
	decalage = 0
	for i in listeDesAttributs:
		if i<idAttribut:
			decalage+=dicoDesAttributs[i]['dimensions']
	return decalage


def importerDonneesBrutes(
		sourcerJC = False,
		dirJC = DIR_JC,
		sourcerJSY = False,
		dirJSY = DIR_JSY,
	):
	r"""
	\brief
	Lit fichiers de données brutes et récolte valeurs dans dataFrames.

	\param[in] sourcerJC
	Vrai si on doit utiliser les fichiers de données
	brutes de Jonathan Cauchon, faux sinon.

	\param[in] sourcerJSY
	Vrai si on doit utiliser les fichiers de données
	brutes de Jonathan St-Yves, faux sinon.

	\return
	Listes de dataFrames, une dataFrame pour chaque fichier lu,
	une liste par source de données brutes.
	"""
	print("IMPORTATION DES DONNÉES BRUTES")

	donnees=[]

	INTER = ' '
	EN_TETE = '{:20s}'.format("répertoire")+INTER+\
		'{:30s}'.format("fichier")+INTER+\
		'{:16s}'.format("(r,c)")
	print()
	print(EN_TETE)
	print('-'*65)

	# FIXME: ne pas répéter deux fois; modulariser selon source (JSY|JC)
	if sourcerJC:

# ???, FIXME: les trois (3) premières lignes de données du fichier de JC
#	    disponible sur le site github manquent une colonne de donnée;
#       il n'y en a que 4208 alors qu'il devrait y en avoir 4209


# ???, FIXME: l'apodisation telle que générée par le script de JC
#		commence par une valeur nulle à la première position, ce qui
#		force l'emploi de domaine de validation R+ au lieu de R+*;
#		Est-ce normal? En comparison, les données d'apodisation générées
#		par JSY sont exclusivement limitées au domaine R+*.


		assert(os.path.exists(os.path.normpath(dirJC))) # check

		fichiersJC = [f for f in os.listdir(os.path.normpath(dirJC))
						if f.endswith(EXT_JC)]

		assert(len(fichiersJC)>0) # check


		for nomFichier in fichiersJC:
			df0=pandas.read_csv(
					os.path.normpath(os.path.join(
						dirJC,nomFichier)),
					sep='\s+',
					header=None,
					skiprows=1,
			)
			donnees.append((ATTRIBUTS_JC,df0))
			EN_TETE = '{:20s}'.format(dirJC)+INTER+\
				'{:30s}'.format(nomFichier)+INTER+\
				'{:16s}'.format(str(df0.shape))
			print(EN_TETE)


	if sourcerJSY:
		assert(os.path.exists(os.path.normpath(dirJSY))) # check

		fichiersJSY = [f for f in os.listdir(os.path.normpath(dirJSY))
						if f.endswith(EXT_JSY)]

		assert(len(fichiersJSY)>0) # check

		for nomFichier in fichiersJSY:
			df0=pandas.read_csv(
					os.path.normpath(os.path.join(
						dirJSY,nomFichier)),
					sep='\s+',
					header=None,
					skiprows=1,
			)
			donnees.append((ATTRIBUTS_JSY,df0))
			EN_TETE = '{:20s}'.format(dirJSY)+INTER+\
				'{:30s}'.format(nomFichier)+INTER+\
				'{:16s}'.format(str(df0.shape))
			print(EN_TETE)

	print()
	return donnees


def refondreDonnees(donneesBrutes):
	r"""
	\brief
	Transforme les listes de dataFrames de données brutes en un
	seul dictionnaire de données structurées.

	\param[in] donneesBrutes
	Liste de couples de identifiants d'attributs et de dataFrames de valeurs.

	\return
	Dictionnaire de données structurées; les clés principales sont
	des entiers naturels successifs (un par donnée distincte); à chaque
	clé principale est attaché un sous-dictionnaire de valeurs; les
	clés de ce sous-dictionnaires correspondent aux identifiants des
	différents attributs provenant des données brutes.

	\note
	Selon la quantité de dimensions des attributs, les valeurs contenues
	dans les sous-dictionnaires sont soit des nombres (si la quantité de
	dimensions est 1), soit des 	numpy.array de nombres (si la quantité
	de dimensions est plus de 1).
	"""
	print("REFONTE DES DONNÉES DANS STRUCTURE UNIFIÉE")

	# initialiser dictionnaire unifié des valeurs
	dicoDesValeurs = {}

	# pour chaque fichier de données chargé dans une DataFrame
	nbDataFrames = len(donneesBrutes)
	nbDigits = len(str(nbDataFrames))
	compteurFichiers=0
	idDonnee=0
	print()
	for attributs,df in donneesBrutes:
		compteurFichiers+=1
		# pour chaque rangée de la DataFrame
		print("---> depuis fichier {0:{n}d}/{1}".format(
				compteurFichiers,nbDataFrames,n=nbDigits))
		for i in numpy.arange(len(df)):
			# récupérer cette rangée (ligne) de données brutes
			ligne = df.iloc[i]
			# créer un dictionnaire ayant les mêmes clés principales
			# que dicoDesAttributs mais un seul champ numérique par attribut
			dicoValeursPourUneDonnee = {}
			# le compléter avec les valeurs associées à chaque attribut
			for k in attributs:
				# retenir le domaine de la valeur de cet attribut
				domaine = dicoDesAttributs[k]['domaine']
				# calculer le décalage en colonnes des valeurs
				decalage = decalageEnColonnes(k,attributs)
				# retenir le nombre de valeurs à charger
				dimensions = dicoDesAttributs[k]['dimensions']
				# si plus d'une dimension, employer un NumPy array
				# sinon, enregistrer la valeur directement
				if dimensions>1:
					# préparer un NumPy array
					plusieursValeurs = numpy.array([])
					# configurer son type numérique
					if 'REEL' in domaine:
						plusieursValeurs.astype(float)
					elif 'NAT' in domaine:
						plusieursValeurs.astype(int)
					else:
						raise

					for j in numpy.arange(dimensions):
						# extraire la valeur
						valeur = ligne[j+decalage]
						# valider le domaine de la valeur
						try:
							assert(domaineValide(valeur,domaine))
						except:
							print(
								"!!! domaineValide_plusieursValeurs"
								"(i,k,j,decalage) = (%d,%d,%d,%d)"
								%(i,k,j,decalage))
						# ajouter cette valeur au NumPy array
						plusieursValeurs = numpy.append(
								plusieursValeurs,valeur)

					# ajouter une valeur correspondant à cet attribut
					dicoValeursPourUneDonnee.update({k:plusieursValeurs})

				else:
					assert(dimensions==1) # check
					# extraire la valeur
					uneValeur = ligne[decalage]
					# valider le domaine de la valeur
					try:
						assert(domaineValide(uneValeur,domaine))
					except:
						print(
							"!!! domaineValide_uneValeur"
							"(i,k,decalage) = (%d,%d,%d)"
							%(i,k,decalage))
					# forcer l'emploi d'un entier si approprié
					if domaine == 'NAT' or domaine == 'NATPOS':
						uneValeur = int(uneValeur)
					# ajouter une valeur correspondant à cet attribut
					dicoValeursPourUneDonnee.update({k:uneValeur})

			# ajouter le dictionnaire rempli pour cette donnée
			# au dictionnaire unifié des valeurs
			dicoDesValeurs.update({idDonnee:dicoValeursPourUneDonnee})
			idDonnee+=1

	print()
	# retourner dictionnaire unifié des valeurs
	return dicoDesValeurs


def enregistrementPickle(dicoDesValeurs,pfxPickle=""):
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
		d2 = {0:dv0,2:dv1,...,Q:dvQ} <--- cf. dicoDesValeurs

			da<p>.keys() = ['symbole','unites','description',
							 'domaine','dimensions']

			dv<q>.keys() = [0,1,...,P]

	\param[in] dicoDesValeurs
	Dictionnaire des valeurs à enregistrer.

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
			'lesValeurs':dicoDesValeurs,
	}

	if not os.path.exists(os.path.normpath(DIR_STRUCT)):
		os.makedirs(os.path.normpath(DIR_STRUCT))

	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	with open(nomFichier, 'wb') as fichierPkl:
	    pickle.dump(
				dicoPourEnregistrer,
				fichierPkl,
				protocol=pickle.HIGHEST_PROTOCOL)

	# test de rechargement
	dicoReCharge = chargementPickle(pfxPickle,silencieux=True)

	# test d'intégrité
	assert(dicoReCharge['lesAttributs'] ==
		dicoPourEnregistrer['lesAttributs'])

	for k1 in dicoReCharge['lesValeurs'].keys():
		for k2 in dicoReCharge['lesValeurs'][k1].keys():
			if dicoReCharge['lesAttributs'][k2]['dimensions']>1:
				assert(numpy.all(dicoReCharge['lesValeurs'][k1][k2]==
					 dicoPourEnregistrer['lesValeurs'][k1][k2]))
			else:
				assert(dicoReCharge['lesValeurs'][k1][k2]==
					  dicoPourEnregistrer['lesValeurs'][k1][k2])


def chargementPickle(pfxPickle="",silencieux=False):
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
	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	with open(nomFichier, 'rb') as fichierPkl:
	    dicoDonneesRechargees = pickle.load(fichierPkl)

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


def printProgressBar(
    iteration,
    total,
    prefix='--->',
    suffix='Complété',
    decimals=1,
    length=40,
    fill='█'):
    r"""
    """
    percent=("{0:."+str(decimals)+"f}").format(
        100*(numpy.true_divide(iteration,float(total))))
    filledLength=int(length*iteration/total)
    bar=fill*filledLength+'-'*(length-filledLength)
    print('\r%s |%s| %s%% %s' %(prefix,bar,percent,suffix),end='\r')
    if iteration==total:
        print()



def creerEchantillonnage(
		dicoValeurs,
		idAttribX,
		idAttriby,
		amplitude=False,
		phase=False):
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
	Si aucune donnée compatible, alors retourne None.

	\warning
	Les données disponibles doivent être compatibles avec
	les attributs demandés.
	"""
	print("CRÉATION DE L'ÉCHANTILLONNAGE")

	idDonnees = list(idAttribX)
	idEtiquettes = list(idAttriby)


	# il faudra seulement sélectionner parmi les données disponibles
	# celles qui sont compatibles avec les attributs demandés!

	# les données X et les étiquettes y doivent être des ensembles disjoints
	assert(idAttribX.isdisjoint(idAttriby))

	attributsDemandes = idAttribX.union(idAttriby)


	X = numpy.array([])
	y = numpy.array([])

	# on veut calculer soit l'amplitude, soit la phase, mais pas les deux
	assert((amplitude and phase) == False)
	complexe = amplitude or phase
	if complexe:
		# il ne doit y avoir que deux attributs dans symbX
		assert(len(idAttribX)==2)
		# l'attribut réel doit être le premier
		sidRe = idDonnees[0]
		# l'attribut imaginaire doit être le second
		sidIm = idDonnees[-1]

	print()
	nbDeDonneesCompatibles = 0
	nbDicoValeurs = len(dicoValeurs)
	printProgressBar(0,nbDicoValeurs)
	for i in numpy.arange(nbDicoValeurs):

		if set(dicoValeurs[i].keys()).\
			intersection(attributsDemandes)\
				==attributsDemandes:

			dX = numpy.array([])
			if complexe:
				#for re,im in zip(dicoValeurs[i][sidRe],dicoValeurs[i][sidIm]):
				re = dicoValeurs[i][sidRe]
				im = dicoValeurs[i][sidIm]
				if amplitude:
					dX = numpy.sqrt(re**2+im**2)
				else:
					dX = numpy.arctan2(im,re)


			else:
				for sid in idDonnees:
					dX = numpy.append(dX,dicoValeurs[i][sid])
			X = numpy.append(X,dX)

			dy = numpy.array([])
			for sid in idEtiquettes:
				dy = numpy.append(dy,dicoValeurs[i][sid])
			y = numpy.append(y,dy)
			nbDeDonneesCompatibles+=1

		printProgressBar(1+i,nbDicoValeurs)

	print()


	if nbDeDonneesCompatibles == 0:
		print("!!! Attributs demandés incompatbles "
		"avec données refondues disponibles.")
		raise

	nbValeursX = int(len(X) / nbDeDonneesCompatibles)
	nbValeursy = int(len(y) / nbDeDonneesCompatibles)

	X=X.reshape((nbDeDonneesCompatibles,nbValeursX))
	y=y.reshape((nbDeDonneesCompatibles,nbValeursy))

	# vérifications de la consistance de la forme des numpy.array
	assert(X.ndim == 2)
	assert(y.ndim == 2)
	assert(X[0].ndim == 1)
	assert(y[0].ndim == 1)

	return X,y


def enregistrementPyTorch(X,y,pfxPyTorch):
	r"""
	\brief
	Enregistre les données et leurs étiquettes en tenseurs PyTorch.

	\param[in] X
	NumPy array des données.

	\param[in] y
	NumPy array des étiquettes.

	\param[in] pfxPyTorch
	Préfixe de nom à donner aux fichiers PyTorch.

	\post
	Les données X et les étiquettes y ont été convertis en
	tenseurs PyTorch et enregistrées dans un format directement
	utilisable par PyTorch. Leur re-chargement a été
	testé et procure les mêmes objets.
	"""
	assert(type(X)!=complex)

	print("ENREGISTREMENT EN FORMAT PYTORCH")

	if not os.path.exists(os.path.normpath(DIR_STRUCT)):
		os.makedirs(os.path.normpath(DIR_STRUCT))

	nomFichierX = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_X))

	nomFichiery = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_y))

	tX = torch.from_numpy(X)
	ty = torch.from_numpy(y)

	torch.save(tX,nomFichierX)
	torch.save(ty,nomFichiery)

	# checks

	rtX = torch.load(nomFichierX)
	rty = torch.load(nomFichiery)
	assert(torch.all(torch.eq(rtX,tX)))
	assert(torch.all(torch.eq(rty,ty)))


class Echantillonnage(torch.utils.data.Dataset):
	r"""
	\brief
	Classe qui instancie des objets dérivés de torch.utils.data.Dataset

	\details
	Les objets de la classe Echantillonnage sont spécialisés
	pour lire les fichiers de données créés pour PyTorch par
	le présent script de traitement.
	Ces fichiers contiennent la transformation des valeurs brutes
	provenant des simulations de réseaux contra-directionnels par
	méthode de matrice de transfert.
	Les valeurs ont été restructurées et manipulées pour être
	séparées en données X et en 	étiquettes y puis entreposées
	sous forme de tenseurs.

	\note
	Entre le {map|iterable}-style datasets, on implémente ici
	le map-style dataset dans lequel on doit redéfinir les
	méthodes __getitem__() et __len__()
	"""
	def __init__(self,pfxPyTorch=""):
		r"""
		\brief
		Initialisateur des objets instanciés

		\details
		Appelle premièrement le contructeur de la classe de base
		et ensuite définit en quoi consiste le constructeur
		spécialisé de de cette classe dérivée.
		"""
		super().__init__()
		# ???: on pourrait recevoir des noms de fichiers

		nomFichierX = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_X))

		nomFichiery = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_y))

		# charger les données dans une variable membre
		self.X = torch.load(nomFichierX)

		# charger les étiquettes dans une variable membre
		self.y = torch.load(nomFichiery)

		# véfification basique de la consistance entre X et y
		assert(len(self.X)==len(self.y))

		# enregistrer la cardinalité de l'échantillonnage
		self.nbTotalDonnes = self.__len__()


	def __getitem__(self, i):
		r"""
		\brief
		Retourne un des échantillons de l'échantillonnage.
		"""
		return self.X[i]


	def __len__(self):
		r"""
		\brief
		Retourne la cardinalité de l'échantillonnage.
		"""
		return len(self.X)



def illustrerEchantillonnage(chi,attrib_y):
	r"""
	\brief
	Trace le graphique de quatre données d'entrée tirées au hasard.


	\param[in] chi
	Instance de la classe Echantillonnage.


	\param[in] attrib_y
	Identifiants des étiquettes.


	\post
	Une figures avec 4 sous-graphiques est tracée sur laquelle
	on a les indices i des données X tracées ainsi que la valeur de
	leurs étiquettes y.


	\note
	Le symbole employé dans le cours pour désigner l'échantillonnage
	global est $\mathcal{X}$, ce qui ressemble beaucoup à $\chi$.
	C'est pourquoi j'utilise "chi" comme nom de variable pour
	le paramètre d'entrée.
	"""

	print("ILLUSTRATION DE L'ÉCHANTILLONNAGE")


	fig, subfigs = matplotlib.pyplot.subplots(2, 2, tight_layout=False)
	# Affichage de données aléatoires
	# l'affichage 2x2 suppose au moins 4 données disponibles
	assert(len(chi)>=4)
	# on présentera 4 données distinctes en ordre d'indice croissant
	idx = numpy.random.randint(0,len(chi),size=1)
	while len(idx)<4:
	        i = numpy.random.randint(0,len(chi))
	        if not i in idx:
	                idx = numpy.append(idx,i)
	idx.sort()

	for i,subfig in zip(idx,subfigs.reshape(-1)):

		yTexte = ""

		for s,v in zip(attrib_y,chi.y[i]):
			if v.item()%1==0:
				vTexte = '{:d}'.format(int(v.item()))
			else:
				vTexte = '{:.2f}'.format(v.item())
			symbole = dicoDesAttributs[s]['symbole']
			yTexte += symbole+"="+vTexte+"; "

		subfig.scatter(numpy.arange(len(chi[i])),chi[i])
		subfig.set_title("(i,y)=(%g,%s)"%(i,yTexte))
		subfig.set_axis_off()

	fig.suptitle('Affichage de quatre données aléatoires\n'\
				 '(i,y)=(indice,étiquettes)')
	fig.show()



def statsEchantillonnage(chi,attrib_y):
	r"""
	\brief
	Trace le graphique des statistiques des étiquettes de l'échantillonnage.


	\param[in] chi
	Instance de la classe Echantillonnage.


	\param[in] attrib_y
	Identifiants des étiquettes.


	\post
	Une figures avec autant de sous-graphiques que d'étiquettes
	est tracée sur laquelle 	on a ...


	\note
	Le symbole employé dans le cours pour désigner l'échantillonnage
	global est $\mathcal{X}$, ce qui ressemble beaucoup à $\chi$.
	C'est pourquoi j'utilise "chi" comme nom de variable pour
	le paramètre d'entrée.
	"""

	print("STATISTIQUES DES ÉTIQUETTES")

	print()
	print("---> Nombre total de données = %d"%(chi.nbTotalDonnes))
	print()

	# commencer par faire un tableau texte

	LIGNE = '-'*58
	#LIGNEn = LIGNE+'\n'
	INTER = ' '*2
	EN_TETE='{:>10s}'.format("y")+INTER+\
			'{:>10s}'.format("min")+INTER+\
			'{:>10s}'.format("max")+INTER+\
			'{:>10s}'.format("moy")+INTER+\
			'{:>10s}'.format("std")+'\n'
	EN_TETE+=LIGNE
	print(EN_TETE)


	nbEtiquettes = len(attrib_y)
	assert(nbEtiquettes>1) # FIXME: sinon problème avec subplots ...

	fig, subfigs = matplotlib.pyplot.subplots(1, nbEtiquettes, tight_layout=False)

	pos=0
	for yId in attrib_y:
		# retrouver symbole attaché à cette étiquette
		symbole = dicoDesAttributs[yId]['symbole']
		domaine = dicoDesAttributs[yId]['domaine']
		unites = dicoDesAttributs[yId]['unites']
		# extraire ensemble des valeurs de cette étiquette
		lesYvals = chi.y[:,pos].numpy()
		# calculer des statistiques
		valmin = lesYvals.min()
		valmax = lesYvals.max()
		moyenne = lesYvals.mean()
		stddev = lesYvals.std()
		if 'NAT' in domaine:
			dformat = 'd'
			valmin = int(valmin)
			valmax = int(valmax)
		else:
			dformat = '.2f'


		STATS='{:>10s}'.format(symbole)+INTER+\
				'{:>10{dom}}'.format(valmin,dom=dformat)+INTER+\
				'{:>10{dom}}'.format(valmax,dom=dformat)+INTER+\
				'{:>10.2f}'.format(moyenne)+INTER+\
				'{:>10.2f}'.format(stddev)

		ANNOT="min = "+'{:>10{dom}}'.format(valmin,dom=dformat)+'\n'+\
				"max = "+'{:>10{dom}}'.format(valmax,dom=dformat)+'\n'+\
				"moy = "+'{:>10.2f}'.format(moyenne)+'\n'+\
				"std = "+'{:>10.2f}'.format(stddev)

		subfigs[pos].hist(lesYvals)
		subfigs[pos].set_title(symbole)
		subfigs[pos].set_xlabel(unites)
		subfigs[pos].annotate(
				ANNOT,
				xy=(0.5,0.05),
				xycoords='axes fraction',
				horizontalalignment='center',
				verticalalignment='bottom',
				bbox = dict(boxstyle="round", fc="0.8"))


		print(STATS)
		pos+=1

	print()

	fig.suptitle("Répartition des étiquettes dans\n"
			  "l'échantillonnage de %d données"%(chi.nbTotalDonnes))
	fig.show()


def procederAuTraitement(
		forcerImportation=False,
		sourcerJC=False,
		dirJC=DIR_JC,
		sourcerJSY=False,
		dirJSY=DIR_JSY,
		attrib_X = ATTRIBUTS_X,
		attrib_y = ATTRIBUTS_y,
		pfxPickle = "",
		pfxPyTorch = "",):
	r"""
	\brief
	Simule la chaîne de processus de traitement en un bloc

	\details
	La chaîne de processus qui permet de préparer les
	données brutes pour chargement PyTorch consiste à
		- importer les données brutes,
		- refondre les données dans une structure unifiée,
		- enregistrer la refonte en Pickle,
		- choisir des attributs de données et des attributs d'étiquettes
		- créer un échantillonnage en fonction des choix d'attributs,
		- enregistrer l'échantillonnage en tenseurs PyTorch,
		- instancier un objet dérivé de torch.utils.data.Dataset
		- illustrer des échantillons depuis l'objet dérivé de Dataset
	"""
	# Il est possible d'ajouter condition sur présence du fichier pickle
	# mais on doit pouvoir la contourner si on veut recréer ce fichier
	# à cause, par exemple, de l'ajout de sources de données brutes.

	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	if os.path.exists(nomFichier) and not forcerImportation:
		dicoDonneesRechargees = chargementPickle(pfxPickle)
		dicoDesValeurs = dicoDonneesRechargees['lesValeurs']
	else:
		if not (sourcerJC or sourcerJSY):

			print("!!! Fichier pickle absent et aucune source demandée ...")
			return

		else:
			# importer les données brutes,
			donneesBrutes = importerDonneesBrutes(
					sourcerJC,dirJC,
					sourcerJSY,dirJSY)
			# refondre les données dans une structure unifiée,
			dicoDesValeurs = refondreDonnees(donneesBrutes)
			# enregistrer la refonte en Pickle,
			enregistrementPickle(dicoDesValeurs,pfxPickle)



	#??? amplitude de E_drop
	#??? phase de E_drop


	# créer un échantillonnage en fonction des choix d'attributs
	try:
		X,y = creerEchantillonnage(
				dicoDesValeurs,
				attrib_X,
				attrib_y,
				amplitude=False,
				phase=False)
	except:

		print("!!! Impossible de créer échantillonnage ...")
		return

	# enregistrer l'échantillonnage en tenseurs PyTorch
	enregistrementPyTorch(X,y,pfxPyTorch)

	# instancier un objet dérivé de torch.utils.data.Dataset
	chi = Echantillonnage(pfxPyTorch)

	print()
	print("---> Nombre total de données = %d"%(chi.nbTotalDonnes))
	print()

	chargeur = torch.utils.data.DataLoader(chi, batch_size=16)


	# illustrer des échantillons depuis l'objet dérivé de Dataset
	illustrerEchantillonnage(chi,attrib_y)
	statsEchantillonnage(chi,attrib_y)



def gererArguments():

	LIGNE = '-'*40
	LIGNEn = LIGNE+'\n'
	INTER = ' '*2
	LISTE_ATTRIBUTS='\n\n'+'{:>3s}'.format("id")+INTER+\
			'{:16s}'.format("attribut")+INTER+\
			'{:16s}'.format("usage possible")+'\n'
	LISTE_ATTRIBUTS+=LIGNEn
	XSet = set()
	ySet = set()
	for k in dicoDesAttributs.keys():
		if dicoDesAttributs[k]['dimensions']>1:
			usage = 'donnée X'
			XSet.add(k)
		else:
			usage = 'étiquette y'
			ySet.add(k)
		LISTE_ATTRIBUTS+='{:3d}'.format(k)+INTER+\
				'{:16s}'.format(dicoDesAttributs[k]['symbole'])+INTER+\
				'{:16s}'.format(usage)+'\n'
	LISTE_ATTRIBUTS+=LIGNEn

	analyseur = argparse.ArgumentParser(
			formatter_class=argparse.RawDescriptionHelpFormatter,
			description=textwrap.indent(
"""
Introduction à l'apprentissage machine
ULaval, GIF-7005-85309, A2019, équipe 13
%s
Script de traitement des données brutes
et de génération d'échantillonnages (X,y)
compatibles avec PyTorch afin de procéder
à l'entraînement de réseaux de neurones.

         DIR_JC = %s
         EXT_JC = %s
        DIR_JSY = %s
        EXT_JSY = %s
     DIR_STRUCT = %s
     NOM_PICKLE = %s
    ATTRIBUTS_X = %s
    ATTRIBUTS_y = %s
  NOM_PYTORCH_X = %s
  NOM_PYTORCH_y = %s

%s"""%(
			LIGNE,
			DIR_JC,
			EXT_JC,
			DIR_JSY,
			EXT_JSY,
			DIR_STRUCT,
			NOM_PICKLE,
			ATTRIBUTS_X,
			ATTRIBUTS_y,
			NOM_PYTORCH_X,
			NOM_PYTORCH_y,
			LISTE_ATTRIBUTS),"     "))


	groupeJC = analyseur.add_argument_group(
			"groupeJC",
			"importation données brutes de Jonathan Cauchon")
	groupeJC .add_argument(
			"-c",#"--sourcerJC",
			help="importer données brutes",
			action="store_true",
			default=False)
	groupeJC .add_argument(
			"-dc",#"--repertoireJC",
			help="répertoire où elles se trouvent",
			type=str,
			default=DIR_JC)


	groupeJSY = analyseur.add_argument_group(
			"groupeJSY",
			"importation données brutes de Jonathan St-Yves")
	groupeJSY.add_argument(
			"-s",#"--sourcerJSY",
			help="importer données brutes",
			action="store_true",
			default=False)
	groupeJSY .add_argument(
			"-ds",#"--repertoireJSY",
			help="répertoire où elles se trouvent",
			type=str,
			default=DIR_JSY)


	analyseur.add_argument(
			"-X",#"--Xattrib",
			nargs='+',
			type=int,
			choices=XSet,
			help="choix des attributs qui composent les données X",
			default=ATTRIBUTS_X)
	analyseur.add_argument(
			"-y",#"--yattrib",
			type=int,
			nargs='+',
			choices=ySet,
			help="choix des attributs qui composent les étiquettes y",
			default=ATTRIBUTS_y)
	analyseur.add_argument(
			"-k",#"--pkPrefixe",
			type=str,
			help="préfixe de nom à donner au fichier Pickle",
			default="")
	analyseur.add_argument(
			"-p",#"--ptPrefixe",
			type=str,
			help="préfixe de nom à donner aux fichiers PyTorch",
			default="")

	args = analyseur.parse_args()


	importerDonnees = args.c or args.s

	procederAuTraitement(
		forcerImportation=importerDonnees,
		sourcerJC = args.c,
		dirJC = args.dc,
		sourcerJSY = args.s,
		dirJSY = args.ds,
		attrib_X = args.X,
		attrib_y = args.y,
		pfxPickle = args.k,
		pfxPyTorch = args.p,
	)


if __name__=='__main__':
	r"""
	"""
	gererArguments()

else:
	sys.exit(0)


# -----------------------------------------------------------------------------
# EOF