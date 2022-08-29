* * *

# Cheatsheet Python3
==================

Content : 

1.  Python general
    
2.  Python for data science
    

  

* * *
<details>
  <summary>1.  Python general cheatsheet </summary>
    

see also 2-pager  [Python3 cheatsheet - perso.limsi.fr](evernote:///view/6367254/s57/e1f90f88-1307-423b-80b8-482d4c4b5825/e1f90f88-1307-423b-80b8-482d4c4b5825/)

code formating , I/ O 

indentation = [Landin's](https://en.wikipedia.org/wiki/Peter_Landin) pseudo law: Treat the indentation of your programs as if it determines the meaning of your programs… Because sometimes it does.

par défaut print retourne à la ligne, sauf si on remplace la variable end: to avoid newline

print("text", variable, sep = 'texte de separation ', end = ' texte qui remplace new-line ')

proper format: put space before and after operators.

an instruction (like a print() ) can exceed one line and finish on the next line (use indentation)

input("this is a prompt message") # will return a string

x = int(input("please enter an integer")) # will return ValueError if input is not an int

There is no difference between input in Python 3 and raw\_input in Python 2 except for the keywords.

\# this is a comment There is no multi-line comment. 

'''' This is a [docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)  . Using docstring in place of multiline comment is possible but not recommended.

Operators

 +  - / \* % (mod aka euclidian remainder)  // (euclidian division) \*\* (power, also pow(a, b) ) 

round(3.14)

<  >  <=  >=  ==

(a // b, a % b).

  

Variables

assignation with =  and variable names must start by a letter or by \_ . Letters, numbers, and underscores will work. Use camelCase, long and explicit variable names. 

print("texte1 ", variable) = print(f"texte2 {variable}.") = print("texte3 {} ".format(variable))

del myVariable # to remove a (global/local) variable from memory

round numbers considered integer, with a decimal 3.0 becomes float.

dynamic casting : existing variable changes type if assigned value of incompatible type.

typecasting = force into a different type. Use the name of the type as typecasting function : 

int() float() str()

interets = int(interets) #converts in place the variable into an integer via dynamic casting

type() # to get type of a variable (as a string)

  

from math import sqrt, pi, exp #fonctions de math de base à importer

  

  

Strings

with double quotes or single quotes (either way is acceptable, typically single-quotes for short strings).

the triple-double-quotes allow to type a whole text on several lines, including returns equivalent to \\n.

f-string is a formated string (with a variable inside marked by {}). any string wih {} has a method .format

concatenate with + 

multiply same text 10 time with 'copy '\* 10

single quotes can be included in strings if they use double quotes.

var1 = 2

var2 = "one and two"

text = "text of the message with {} formatted variables, {}"

print(text.format(var1, var2))

 in Python any empty list, string, or tuple is [falsy](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)

usefull methods for strings : 

*   upper()  lower()  capitalize() (title(): formating case
    
*   replace(old, new)  
    
*   find(chaîne) returns -1 (absent) or indice of 1st occurence ;  in true or false;
    
*   .startwith(myInitialText) return true or false
    

  

[list, tuple, set and dict](https://docs.python.org/3/tutorial/datastructures.html)

myList = \[ value 1, value2\]

myDict = {"Georges Dupont": 10000, "Luc Martin": 150, "Lucas Anderson": 300, "Alexandre Petit": 1800.74}

myTuple = (1, 'pomme') # immutable !!

#NB: array is not native Python, but comes from Numpy package/ array module

list = general table, mixes heterogeneous types including list of list, 

    index 0 for first, -1 for last, negative indices cycle back, \[:2\] from 0(first) to 2. 

    lists are ordered, mutable, can contain duplicates & heterogeneous types   

⚠ range returns an immutable sequence type (in Python3), not a list

force a list with : \`\`\`list(range(4))\`\`

dict = access by key instead of index

tuple = immutable list, used to return multiples values in a function. 

Also : 

from enum import Enum #another type of data to chose in a fixed enum

class Strategie(Enum):

    CHANGER = 1

    GARDER = 2

from array import array #another type for math calculation - see Numpy

  

functions for all iterables/iterators: 

*   liste.insert(myindex, myvalue)  ;
    
*   .append(myvalue) .extend(my\_other\_list)  to append / concatenate;
    
*   r.remove(myvalue)  finds and remove first occurence;
    
*   i.index(myvalue)  finds first occurence;
    
*   mot clé del(myindex)  // for Dict =  .pop(myKey)
    

del liste\[3\] # \[4, 5, 1, 3\]

len(liste)

a, b = (1, 'pomme') # shorcut to open a tuple

  

Booleans

if not my\_list: print('List is empty!')  # Any empty list, string, or tuple is falsy

*   and ; or ; not() ; \== != < <= >  >= ;
    
*   myValue in myString ;
    

  

Command line

passing argument from the command line with ARGV - first is the script name, second is 

from sys import argv

first, second = ARGV

  

Function def .... : 

def print\_two(\*args): #here args is a list

    arg1, arg2 = args

    print(f"arg1: {arg1}, arg2: {arg2}")

  

def calcul\_IMC(poids = 60, taille = 1.70):

    imc = poids / taille\*\*2

    return(imc)

  

calcul\_IMC(poids = float(input("Quel poids (en kg) ? ")) ,

           taille = float(input("Quelle taille (en metres) ? ")))

Can return multiple values with ; 

    return a, b, c

  

Conditions if ... : ... elif ... : ... else: ...

if len(nom) > 0:

    print("Hello", nom, "!")

**elif** len(nom) <10:

    xxxxx

else:

    print("Hello World !")

True, False

any empty list, string, or tuple is [falsy](https://docs.python.org/3/library/stdtypes.html#truth-value-testing)

No switch statement, [use dict instead](https://www.evernote.com/l/ADkVrRw5rpFGf5C_wuRk_yk7qeaRyAEfIHw/). 

  

Loops

for element in myStringOrMyList:

for i in range(0,nb): #borne droite ouverte !!

for element in range(start,step,endPlusOne):  #shortcut range(1000) = range (0,1,999)

[https://docs.python.org/3/reference/compound\_stmts.html?highlight=while#for](https://docs.python.org/3/reference/compound_stmts.html?highlight=while#for)

while condition :

Interrupt loop with continue (next loop) or break loop

  

modules = classes+variables+functionsDS | librairies Python pour Data Science-2 - Initiez-vous à la librairie Numpyark

module = 1 file with classes+variables+fnctions

library or package  = set of files including an \_\_init\_\_.py

pip #Package Installer for Python

import myModule as myAlias # import the whole module, access via myAlias.myFunction()

import myPackage #access via myPackage.myModule.myFunction()

import myPackage.myModule as myAlias #import only 1 module in the package

from myModule import myVariableOrMyFunction

(ne pas tout importer avec from myModule import \* car risque de conflit)

Main modules/packages : 

*   random => random() ; uniform(a,b) bornes incluses ; randint(a,b) ⚠bornes incluses
    

*   gauss(avg, stdev) pour loi normale ;
    
*   choice(myList) échantillonage ; choices(myList, k=mySampleSize) échantillon avec remise ; sample(myList, mySampleSize) échantillon sans remise
    
*   !!! toujours seed() pour véritable aléatoire
    

*   numpy package => includes random ⚠ dans np, randint(a,b) borne sup exclue !!
    

  

  

file access

mystream = [open](https://docs.python.org/3/library/functions.html#open)(filename) opens a stream

mystream.read()  to look into

close – Closes the file

read – readline Reads the contents of the file (from the stream object returned by open) You can assign the result  to a variable – Reads just one line of a text file.  modifiers to the file modes can I use? The most important one  to

know for now is the + modifier, so you can do 'w+', 'r+', and 'a+'.

truncate – Empties the file. Watch out if you care about the file.

write('stuff') – Writes “stuff” to the file. target.write(line1) target.write("\\n")

[seek](https://docs.python.org/3/library/io.html?highlight=seek#io.IOBase.seek)(0) – Move the read/write location to the beginning of the file.

    print(line\_count, f.readline()) print can read a specific line number

  

Doc - manual

in python: 

pydoc name\_of\_function

help(name\_of\_function)

on Windows Powershell: 

python -m pydoc name\_of\_function

  </details>

* * *

  

revuiew Euler023 ; 

\_\_\_

  

![](Cheatsheet  Python  LP3THW - DataScience_files/Image.png)

  

![](Cheatsheet  Python  LP3THW - DataScience_files/Image [1].png)

  

![](Cheatsheet  Python  LP3THW - DataScience_files/Image [2].png)

  

  

  

  

* * *

  

from string import ascii\_lowercase

\>>> for c in ascii\_lowercase:

  

[![mementopython3.odg](Cheatsheet  Python  LP3THW - DataScience_files/6797389174bc11abce43c516411097a4.png)](Cheatsheet  Python  LP3THW - DataScience_files/mementopython3.odg)

  

* * *

  

  

2.Python for Data Science

  

Link to courses : [DS | Initiez-vous à Python pour l'analyse de données - OpenClassrooms](evernote:///view/6367254/s57/9671058b-017b-483c-b085-27aa9676a0d9/9671058b-017b-483c-b085-27aa9676a0d9/)

  

Dev environment : 

*   [Anaconda](https://www.anaconda.com/distribution/) : includes Jupyter Notebook. Launch from terminal with jupyter notebook  or (from working dir) jupyter notebook my\_notebook.ipynb
    
*   [Google Colaboratory](https://colab.research.google.com/?utm_source=scs-index) : full online
    

  

Les cellules

Notebook = includes executable code "cellules FR"

4 types of cells : code, markdown, row nbconvert et heading.

*    Code \= basic cell // heading = obsolete // Raw = to control document formating when converting
    
*   Markdown \[m\] \= basic formating info. See more here : [Reminder Markdown](https://fr.acervolima.com/cellule-markdown-dans-le-bloc-notes-jupyter/)
    

*   \* or \_ italic  \*\* or \_\_ bold ;
    
*   \# heading1 ## heading2 (and so on)
    
*   \> >> to indent text (same as <blockquote> )
    
*   \-  (double space) for bullet point, tab to indent
    
*   \`for monospace font\` \`\`\`code as illustration\`\`\`
    
*   $ for in-line math, $$ for separate line 
    
*   ![](Cheatsheet  Python  LP3THW - DataScience_files/Image [3].png)
    

  

  

Module Random

uses Mersenne Twister to generate random numbers

  

Main DS Packages : 

\* numpy et scipy pour les calculs

\* Matplotlib et Seaborn pour la visualisation

\* Scikit-learn pour les algorithmes

\* Pandas pour les gérer les données (les charger, appliquer des opérations d'algèbre relationnelle, etc.)

\* Tensorflow et PyTorch pour le deep learning

  

  

  

  

Jupyter

Executer code = Ctrl + Ent - open new code cell = Shit+Ent.

  

Créer un script pour partager des variables ou des fonctions sur plusieurs notebook: 

fichier .py

%timeit myFunction(myArgs) #renvoit le temps de travail d'une fonction

  

  

  

Matplotlib et Seaborn

  

%matplotlib inline # afficher les graphiques dans la continuité du code, pas dans fenêtre à part

import matplotlib.pyplot as plt

  

plt.style.use('seaborn-whitegrid')

  

  

#toutes ces fonctions plot renvoient un objet (conteneur)  avec tout l'objet

plot = plt.plot(myListOfValues) #lignes

plt.scatter(myListX, myListY) #scatter ne relie pas les points. C'est pourquoi on a souvent : 

plt.scatter(range(100),myListOf100Values)

plt.bar(myListOfLabels, myListOfValues)

plt.hist(myArray) #qu'on peut regrouper en "bin=100" paquests de 100

  

  

#si on crée initialement une figure: 

myFigure = plt.figure() 

\# conteneur avec tous les objets ensuite tous les plot, scatter etc s'appliquent à cette figure

myAxes = plt.axes()

x = np.linspace(0, 10, 1000)

myAxes.plot(x, np.sin(x));

  

Un exemple compliqué : 

\# Chanegr la taille de police par défaut

plt.rcParams.update({'font.size': 15})

fig = plt.figure()

ax = plt.axes()

plt.plot(x, np.sin(x - 0), color='blue', linestyle='solid', label='bleu')

plt.plot(x, np.sin(x - 1), color='g', linestyle='dashed', label='vert')

\# Valeur de gris entre 0 et 1, des traits et des points

plt.plot(x, np.sin(x - 2), color='0.75', linestyle='dashdot', label='gris')

plt.plot(x, np.sin(x - 3), color='#FF0000', linestyle='dotted', label='rouge')

\# Les limites des axes, essayez aussi les arguments 'tight' et 'equal'

plt.axis(\[-1, 11, -1.5, 1.5\]);

plt.title("Un exemple de graphe")

\# La légende est générée à partir de l'argument label de la fonction plot. 

\# L'argument loc spécifie le placement de la légende

plt.legend(loc='lower left');

\# Titres des axes

ax = ax.set(xlabel='x', ylabel='sin(x)')

  

plt.errorbar(x, y, yerr=dy, fmt='.k'); #afficher barres d'erreur (incertitude) , voir la doc

plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0);

  

Quelques exemples : 

print(plt.style.available\[:6\])

\# Notez la taille de la figure (en quelle unité ? nb de caracteres?)

fig = plt.figure(figsize=(12,8))

for i in range(6):

\# On peut ajouter des sous graphes ainsi

fig.add\_subplot(3,2,i+1)

plt.style.use(plt.style.available\[i\])

plt.plot(x, y)

\# Pour ajouter du texte

plt.text(s=plt.style.available\[i\], x=5, y=2, color='red')

  

\# On peut aussi tout personnaliser à la main

x = np.random.randn(1000)

plt.style.use('classic')

fig=plt.figure(figsize=(5,3))

ax = plt.axes(facecolor='#E6E6E6')

\# Afficher les ticks en dessous de l'axe

ax.set\_axisbelow(True)

\# Cadre en blanc

plt.grid(color='w', linestyle='solid')

\# Cacher le cadre

\# ax.spines contient les lignes qui entourent la zone où les

\# données sont affichées.

for spine in ax.spines.values():

spine.set\_visible(False)

\# Cacher les marqueurs en haut et à droite

ax.xaxis.tick\_bottom()

ax.yaxis.tick\_left()

\# Nous pouvons personnaliser les étiquettes des marqueurs

\# et leur appliquer une rotation

marqueurs = \[-3, -2, -1, 0, 1, 2, 3\]

xtick\_labels = \['A', 'B', 'C', 'D', 'E', 'F'\]

plt.xticks(marqueurs, xtick\_labels, rotation=30)

\# Changer les couleur des marqueurs

ax.tick\_params(colors='gray', direction='out')

for tick in ax.get\_xticklabels():

tick.set\_color('gray')

for tick in ax.get\_yticklabels():

tick.set\_color('gray')

\# Changer les couleur des barres

ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');

  

  

Cf cheatsheet Matplotlib Jupyter recommandée par le cours : 

[https://nbviewer.org/urls/gist.githubusercontent.com/Jwink3101/e6b57eba3beca4b05ec146d9e38fc839/raw/f486ca3dcad44c33fc4e7ddedc1f83b82c02b492/Matplotlib\_Cheatsheet](https://nbviewer.org/urls/gist.githubusercontent.com/Jwink3101/e6b57eba3beca4b05ec146d9e38fc839/raw/f486ca3dcad44c33fc4e7ddedc1f83b82c02b492/Matplotlib_Cheatsheet) 

  

Seaborn = complément esthétique et statistique de Matplotlib (qui fonctionne avec des DF Pandas)

  

import seaborn as sns

sns.set()

x = np.linspace(0, 10, 500)

y = np.random.randn(500)

plt.plot(x,y)

  

#graphiques avec fonctions statistiques incluses: 

sns.distplot(y, kde=True);

  

Numpy and its arrays (=matrices)

À chaque fois que vous vous trouvez en train d'utiliser une boucle pour effectuer une opération en Python, demandez-vous si cette opération ne peut pas s'accomplir grâce à Numpy sans boucle.

arrayarrays have same type (unlike lists)

import numpy as np _#alternative : import array as arr_

array\_of\_int = arr.array("i", \[3, 6, 9, 12\]) #array module requires same type (here "i" means integer)

array\_heterog = np.array(\["numbers", 3, 6, 9, 12\])

array = must be declared (created), are more efficient than lists for data storage, can handle math operations

    index 0 for first, -1 for last, negative indices cycle back, \[:2\] from 0(first) to 2(excluded). 

⚠ vecteurs numpy sont (200,) au lieu de (200,1). Cela permet de les voir soit comme ligne, soit comme colonne. Mais parfois il faut tout de même les \`\`\`reshape\`\`\` . 

Création d'un array Numpy: 

  

np.array(myList) # force la conversion le cas échéant, crée des array 1xn ou nxp (avec liste de listes)

np.array(myList, dtype = "float32")

  

Les tableaux remplis : 

np.zeros(10, dtype = int) #tableau 1x10

np.ones((3,50), dtype = float)

np.full((4,4), 3.14)

np.eye(3) #la matrice identité

np.linspace(start= 0 , stop = 1, num = 11)  #un sequence espacée linéairement avec num poteau (num-1 intervalles)

np.arange(start= 0 , stop = 1, step = 0.1) #une autre séquence, presque comme linspace (mais n'inclut par la borne sup)

np.random.rand(3,5) #une matrice aléatoire 3x5 - aussi avec random.random et random.normal

  

myArray.shape #renvoie une liste de 2 élements \[lignes,colonnes\]

numpy.shape -> tuple with size of table .reshape -> change size

  

  

Connaitre le tableau : 

myArray.ndim #nb de dimensions

myArray.shape

myArray.size #nombre total éléments = n x p

myArray.dtype

  

myArray\[1,3:14:2\] #slicing = start:end:step => 3 to 14 by a step of 2

myArray\[::-1\] #reverse of array

myTable\[1,:\] #la première ligne

Manipulations de matrices : 

np.sum(myMatrix,axis = 0) # 0 sum des lignes, 1 sommes des colonnes, pas d'axis -> somme totale

np.mean(myArray)

np.var(myArray) #variance

np.argmin(myArray) #index du min

np.percentile(....) ??

  

np.concatenate(myTable1, myTable2) # use rather np.vstack(myListOfArrays) and np.hstack(myListOfArrays)

np.where(myArrays > 3) #renvoit un extrait - quel type et taille ????

no.newaxis ?!?!?

b = np.arange(3)\[:, np.newaxis\] #cf cours python pour DS3-2.1

  

LA PUISSANCE DE NUMPY, c'est de parcourir des tableaux SANS BOUCLE for: 

\+ - \* / // #division entiere arrondie

np.abs(myArray)

np.exp(myArray) # log, etc.

myArrayOfBool = myArrayOfInt > 3

  

np.sum, np.std, np.argmin np.argmax np.percentile /// pour appliquer sur colonnes / lignes

avec BROADCASTING, on peut faire des opérations, y compris sur des tableaux de taille différente 

  

  

* * *

Pandas 🐼

un DF = des lignes (chacune nommée par un index) et des colonnes. 

pandas.Series => objet colonne (Pandas). Pour obtenir colonne numpy, methode .values

Un DF est un ensemble de Series (dès que c'est 2 séries ou plus, c'est un DF) 

  

  

Pour regarder dans un DF: 

  

df = pd.read\_csv("../Dataset/Titanic.csv")

pd.DataFrame(famille\_panda\_numpy, index = \['maman', 'bebe', 'papa'\], columns = \['pattes', 'poil', 'queue', 'ventre'\])

  

population = pd.Series(population\_dict) #une Série (colonne) se construit a partir d'un dict (avec key -> index), ou d'une Liste de valeur avec une liste index (par défaut, index = 0,1,2...)

  

  

df.head(2)

df.shape #returns tuple

df.info() # aussi dtypes (list Names and their types) and describe

len(myDF)

titanic.describe(include="all")  # (from np) donne des stats (count, mean, std, ... not median)

  

  

df.columns  # df.columns.difference(\[''\]) remet les colomnes par ordre alphabe'tique (pourquoi ?!?)

  

myDF\["Ventre"\] myDF.Ventre  # 3 methodes pour extraire unes colonne (objet pandas.Series )

myDF.iloc\[:,0\] # afficher la première colonne

myDF\[\['myColumn1', 'myColumn2'\]\] # attention liste de noms de colonnes

famille\_panda\_df.iloc\[2\] # pour les lignes / index 

famille\_panda\_df.loc\["papa"\] #loc avec les noms, iloc avec les no d'indices

df.loc\[:, \['PClass','Name'\]\]

\---- a bit smarter

mydf.columns.difference(\['Age','SexCode'\]) # afficher toutes les colonnes sauf ... 

\# la méthode columns.difference remet les colonnes par ordre alphabetique (?!?!) 

  

mydf.drop\_duplicates() # voir aussi unique pour la liste

  

#To print, change display options with pandas.set\_options() and pandas.get\_options()

pandas.set\_option('display.max\_rows', None)

On supprime la première colonne inutile

  

df.drop(\['Unnamed: 0'\], axis=1, inplace=True)

  

myDF.append(myOtherDF)

  

On manipule un DF: 

for ind\_ligne, contenu\_ligne in famille\_panda\_df.iterrows(): #envoie (à chaque itération de la boucle for) un tuple dont le premier élément est l'index de la ligne, et le second le contenu de la ligne en question 

  

masque = famille\_panda\_df\["ventre"\] == 80 #renvoi une série de booléens, qui peut servir de masque

df\[df.PClass != "1st"\]  # autre exemple de masque

~ df.PClass.isin(\['1st', '2nd'\]) # autre exemple de masque - le ~ est une negation

myDF\[myDF.PClass.isna()\] # also .notna()

  

famille\_panda\_df\[~masque\] #inversion du masque

\# ======= SORTING ==========

df.Age.sort\_values(ascending=False) # default ascending=True

df.sort\_values(by = \['PClass','Age'\], ascending=\[True,False\]) 

  

  

\# ======== UNIQUE VALUES =======

df.PClass.unique()  # renvoie la liste 

df.PClass.nunique()

  

  

titanic.fillna(value={"age": 0}).age.head(10)

  

titanic.dropna().head(10)

titanic.dropna(axis="columns").head()

  

titanic.rename(columns={"sex":"sexe"})

Bon nombre de fonctions Pandas, telles que  dropna  ,  fillna  ,  drop  , etc acceptent un argument  inplace  .

  

titanic.pivot\_table('survived', index='sex', columns='class', aggfunc="sum")

  

Projection = sélection de colonnes

Restriction = sélection de lignes

Union = vstack des lignes 

Jointure = pd.merge

pd.concat(myDF1, myDF2) #par défaut vstack, se change avec arg axis =1

!!! la concatenation concserve les index => utiliser des index hierarchiques

  

df3 = pd.merge(df1, df2, left\_on= "employee", right\_on= "emp\_name")

  

  

Stats on DF : 

  

import numpy as np

  

myDF.Age.mean()  # also median() max() min() std() var() 

df.Age.quantile(\[.1, .5\]) #calculates the quantiles in the list

df.Age.quantile(np.linspace(start = 0, stop = 1, num= 11)) # for 10 deciles

  

titanic.describe(include="all")  # donne des stats (count, mean, std, quartiles ... not median)

#in cludes all would include any type, not just nbers. To exclude numbers : 

df.describe(exclude=\[np.number\])  # df.describe(percentiles=np.linspace(start = 0, stop = 1, num= 11))

  

  

* * *
