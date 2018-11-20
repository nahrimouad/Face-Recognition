# Face-Recognition
face recognition using trainer
La première étape dans la reconnaissance faciale s’agit d’abord de détecter le visage de l’utilisateur, pour cela on utilise la bibliothèque de vision OPEN CV (Open Source Computer Vision Library) est une bibliothèque de logiciels de vision artificielle et d'apprentissage automatique. Open CV a été construit pour fournir une infrastructure commune pour les applications de vision par ordinateur et pour accélérer l'utilisation de la perception de la machine dans les produits commerciaux. En tant que produit sous licence BSD, Open CV facilite l'utilisation et la modification du code par les entreprises.
Open cv est la bibliothèque de vision par ordinateur la plus populaire, et maintenant nous allons apprendre comment configurer open cv, comment accéder à notre webcam et comment facilement nous pouvons écrire un programme de détection de visage.
Pour installer open cv dans notre environnement python, nous avons besoin de :


•	Python 2.x
•	Open CV 2.x
•	Bibliothèque Numpy (on peut la télécharger en utilisant pip)

Première chose après télécharger en premier python et l'installer dans son emplacement par défaut (c : / Python27)
Après l'avoir installé, et extraire l’open cv, nous allons dans le dossier "opencv / Build / python / 2.7 / x64" et copiez le fichier "cv2.pyd" dans le dossier "c: / python27 / Lib / site-packages /".
Et maintenant, nous sommes prêts à utiliser open cv en python. juste un seul problème est là, Open cv utilise une bibliothèque numpy pour ses images donc nous devons aussi installer une bibliothèque numpy (est une bibliothèque numérique apportant le support efficace de larges tableaux multidimensionnels, et de routines mathématiques de haut niveau (fonctions spéciales, algèbre linéaire, statistiques, etc...)
Maintenant que tout est configuré et en cours d'exécution allons écrire un code pour détecter les visages de la webcam.
C'est une sorte de programme de bonjour pour open cv. La méthode que nous allons utiliser dans ce programme est un classificateur en cascade, qui peut être chargé avec un fichier xml pré-entraîné, ces fichiers xml sont difficiles à former mais heureusement, nous ne devons pas nous inquiéter.
Pour utiliser le classificateur de détection de visage, nous devons copier le fichier xml classifier à partir du "[dossier extrait opencv]/ sources / data / haarcascades /", puis copier le fichier haarcascade_frontalface_default.xml dans notre dossier de projet (même emplacement où vous économiserez le programme python).
Maintenant c'est fait, nous pouvons continuer, nous pouvons charger le classificateur maintenant.
Pour faire un programme de reconnaissance faciale, nous devons d'abord former le système de reconnaissance avec l'ensemble de données des visages précédemment capturés avec son ID, par exemple nous avons deux personnes alors la première personne aura ID 1 et la deuxième personne aura ID 2, de sorte que tous les images de la première personne de l'ensemble de données auront l'ID 1 et toutes les images de la deuxième personne de l'ensemble de données auront l'ID 2, puis nous utiliserons ces images pour former le système de reconnaissance pour prédire chaque visage nouvellement présenté dans le cadre vidéo.
Laissons donc le programme en 3 parties principales:


 	Créateur de dataSet
 	Entraîneur
 	Détecteur


1)	Générateur de dataSet :

Créons le script du générateur de dataSet, ouvrons notre fichier Python IDLE et créons un nouveau fichier et enregistrons-le dans notre dossier de projet. Faut s’assurer également que le fichier haarcascade_frontalface_default.xml se trouve dans le même dossier precedent:
Bibliothèque cv2 (bibliothèque opencv)
Créer un objet de capture vidéo
Objet cascadeClassifier()
Notre générateur de dataSet va capturer quelques exemples de visages d'une personne de la trame vidéo en direct et lui assigner un ID et il sauvegardera ces échantillons dans un dossier que nous créerons et nous le nommerons dataSet.
Utilisateur. [ID]. [SampleNumber] .jpg Par exemple si l'ID utilisateur est 2 et son 10ème échantillon de la liste d'échantillons, le nom du fichier sera : ‘’User.2.10.jpg’’
Pourquoi ce format ?? Eh bien, nous pouvons facilement obtenir quel visage de l'utilisateur il est à partir de son nom de fichier lors du chargement de l'image pour la formation du système de reconnaissance.
Maintenant nous devons obtenir l'ID utilisateur à partir du Shell en entrée, et d'initialiser une variable de compteur pour stocker le numéro d'échantillon.
Laissons maintenant la boucle principale, nous prendrons 20 échantillons du flux vidéo et l'enregistrerons dans le dossier dataSet que nous avons créé précédemment, c’est le rôle de la variable entière sampleNum que nous allons incrémenter.
Nous avons donc ajouté ces deux lignes pour obtenir le numéro d'échantillon et enregistrer le visage au format jpg avec notre convention de nommage.
Et pour capturer le visage, c'est cette partie "grise [y: y + h, x: x + w]" où x, y est la coordonnée supérieure gauche du rectangle de visage et h, w est le la hauteur et le poids du visage en termes de pixels.
Mais ce code va prendre des échantillons varient rapidement comme 20 échantillons dans une seconde,  mais nous ne voulons pas cela, nous voulons capturer des visages sous des angles différents et pour cela il doit être lent. Pour cela, nous devons augmenter le délai entre les cadres et nous avons besoin de casser la boucle après avoir pris 20 échantillons.
Là nous allons, maintenant il faudra attendre 100 entre les images qui nous donnerons le temps de bouger votre visage pour obtenir un angle différent et il se fermera après avoir pris 20 échantillons.
Donc, notre boucle principale est faite maintenant nous avons juste à libérer la caméra et fermer les fenêtres.
Si nous exécutons ce code maintenant, nous verrons qu'il capturera les visages de la vidéo en direct et l'enregistrera dans le dossier dataSet.
Semble bon ... Maintenant nous avons notre ensemble de données, nous pouvons maintenant former le système de reconnaissance pour apprendre les visages de cet ensemble de données.

2)	Entraineur-Face Recognizer :	

Nous avons donc créé un jeu de données étiqueté pour notre système de reconnaissance faciale, maintenant il est temps d'utiliser cet ensemble de données pour former un reconnaisseur de visage.
Avant de commencer le codage, nous avons besoin d'une nouvelle bibliothèque appelée pillow (bibliothèque de traitement d'images, Elle permet d'ouvrir, de manipuler, et de sauver différents formats de fichiers graphiques).
Ouvrons donc la cmd (exécutez en tant qu'administrateur) et tapons la commande suivante pour naviguer vers le répertoire python pip: "cd c: / python27 / scripts /" "Pip installer pillow".
Importons la bibliothèque opencv / cv2, nous aurons besoin de l'os pour accéder à la liste des fichiers dans le dossier dataset, nous devons également importer la bibliothèque numpy, et nous devons importer la bibliothèque pillow / PIL que nous avons installée auparavant.
Maintenant, nous devons initialiser le détecteur et le détecteur de visage
Charger les données d'entraîneur :
Ok, maintenant nous allons créer une fonction qui récupérera les images d'entraînement du dossier du dataSet, et obtiendra également les identifiants correspondants à partir de son nom de fichier, (nous avons formaté le nom du fichier comme User.id.samplenumber dans notre script précédent).

Donc je vais nommer cette fonction "getImagesWithID" nous avons besoin du chemin du dossier dataset donc nous fournirons le chemin du dossier comme argument. Donc, la fonction sera comme ça :
def  getImagesWithID(path)
Donc maintenant, à l'intérieur de cette fonction, nous allons faire ce qui suit :

-	Charger les images d'entraîneur à partir du dossier du dataSet.
-	Capturer les visages et les identifiants des images d'entraîneur.
-	Placez-les dans une liste d'ID et d'échantillons de visage et la retourner.
Pour charger l'image, nous devons créer les chemins de l'image.
Cela obtiendra le chemin de chaque image dans le dossier.
Maintenant nous devons créer deux listes pour les visages et les identifiants pour les stocker.
Maintenant, nous allons boucler les images en utilisant le chemin de l'image et chargerons ces images et IDs, nous allons ajouter cela dans les listes.
Dans le code ci-dessus nous avons utilisé "Image.open (imagePath) .convert ('L')" qui charge l'image et la convertit en échelle de gris, mais maintenant c'est une image PIL que nous devons convertir en tableau numpy.
Pour cela nous le convertissons en un tableau numpy "imageNP = np.array (faceImg, 'uint8')".
Pour obtenir l'Id, nous avons divisé le chemin de l'image et pris le premier de la dernière partie (qui est "-1" en python) et c'est le nom du fichier image. Maintenant, voici l'astuce, nous avons enregistré le nom de fichier dans notre programme précédent comme ce "User.Id.SampleNumber" si nous divisons cela en utilisant "." le nous obtiendrons 3 jetons dans une liste "User", "Id", "Numéro d'échantillon".
Donc pour obtenir l'Id, nous allons choisir 1er index (index commence à partir de 0).
Maintenant, nous utilisons le détecteur pour extraire les visages et les ajouter dans la liste faces avec l'Id. Nous avons donc fini maintenant nous devons juste retourner cette valeur.
Maintenant, si nous exécutons ce code, il créera un fichier "training.yml" dans le dossier du traineur.
Nous utiliserons ce fichier dans notre prochain post pour reconnaître les visages que nous avons formés pour reconnaître le visage.
3)	Detecteur :

Nous avons déjà le reconnaisseur formé dans un dossier nommé "Recognizer" et "trainingData.yml" à l'intérieur. Maintenant, nous allons utiliser ces données d'entraînement pour reconnaître certains visages que nous avons déjà formés.
Commençons par importer les bibliothèques 
Ensuite, nous créons un objet de reconnaissance à l'aide de la bibliothèque open cv et chargeons les données d'entraînement (avant cela, il suffit de placer notre script au même endroit que notre dossier "Recognizer").
Maintenant, nous allons créer un classificateur en cascade en utilisant haarcascade pour la détection des visages, en supposant que le fichier cascade est au même endroit. Après avoir créer l'objet de capture vidéo
Ensuite, nous avons besoin d'une "police" parce que nous allons écrire le nom de cette personne dans l'image, donc nous avons besoin d'une police pour le texte à afficher
Le premier paramètre donc est le nom de la police, 2ème et 3ème est l'échelle horizontale et verticale, 4ème est cisaillement (comme italique), 5ème est l'épaisseur de la ligne, 6ème est le type de ligne.
Commençons alors la boucle principale et faisons les étapes de base suivantes :

	Démarrer la capture d'images à partir de l'objet de la caméra
	Convertir en échelle de gris
	Détecter et extraire les visages des images
	Utilisez le module de reconnaissance pour reconnaître l'identifiant de l'utilisateur
	Placez l'identifiant / nom et le rectangle prédits sur le visage détecté
Pour clarifier, dans les deux lignes ci-dessus, le système de reconnaissance prédit l'identité de l'utilisateur et la confiance de la prédiction respectivement, dans la ligne suivante, nous écrivons l'ID utilisateur dans l'écran sous la face, qui est la coordonnée (x, y + h)
Maintenant, avec cela, nous sommes à peu près fait, nous pouvons ajouter un peu plus de finition comme son identifiant d'utilisateur à la place du nom (choix optionnel)
