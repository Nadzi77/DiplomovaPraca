# Diplomová Práca

**Názov:** &ensp;&ensp;&ensp; **Bezpečnosť účastníkov cestnej premávky**  
**Autor:**  &ensp; &ensp; &ensp;Bc. Matúš Nadžady  
**Školiteľ:**  &ensp; &ensp;RNDr. Zuzana Černeková, PhD.  
**Anotácia:**   &ensp;  Naštudovať problematiku sledovania cestnej premávky. Zamerať sa na detekciu a segmentáciu účastníkov cestnej premávky, najmä chodcov. Analyzovať existujúce riešenia publikované v dostupnej odbornej literatúre. Navrhnúť a implementovať metódu na predikciu pohybu chodcov a možnej kolízie s nimi. Preskúmať prínosy pri využití RGBD dát. Vyhodnotiť dosiahnuté výsledky.  

**Link na prácu:**
* https://www.overleaf.com/read/tytpcxpkjdvz

**Literatúra:**
* A Review of Intelligent Driving Pedestrian Detection Based on Deep Learning (https://www.hindawi.com/journals/cin/2021/5410049/)
* Pedestrian and Ego-vehicle Trajectory Prediction from Monocular Camera (https://openaccess.thecvf.com/content/CVPR2021/papers/Neumann_Pedestrian_and_Ego-Vehicle_Trajectory_Prediction_From_Monocular_Camera_CVPR_2021_paper.pdf)
* Future Person Localization in First-Person Videos (https://arxiv.org/pdf/1711.11217.pdf)
* SGAN (https://arxiv.org/pdf/1803.10892.pdf)
* SGCN (https://arxiv.org/pdf/2104.01528.pdf)


**Priebeh výskumu:**
* Špecifikácia zadania (marec 2022)
* Študovanie prvej literatúry, hľadanie adekvátnych článkov (marec-apríl 2022)
* Výber vhodného detektora pre rozpoznávanie a detekciu objektov v reálnom čase (apríl 2022)
* Testovanie detektora YOLOv5 (apríl 2022)
* Príprava kostry diplomovej práce v LaTex (máj 2022)
* Hľadanie ďalších článkov, bližšie skúmajúcich predikciu pohybu (jún 2022)
* Študovanie oblasti neurónových sietí (august-december 2022)
* Prvé pokusy o trénovanie neurónových sietí (október 2022)
* Hľadanie vhodných datasetov pre túto prácu (október-november 2022)
* Vybavovanie datasetu z firmy Asseco Central Europe, a.s. (november 2022)
* Testovanie detektora YOLOv7 (október 2022)
* Pridanie sortovacieho algoritmu do detektora YOLOv7 (november 2022)
* Pridanie trackingu (november 2022)
* Písanie prvej časti diplomovej práce (Teória) v LaTex (november 2022)
* Pokračovanie v písaní v LaTex (Predošlé práce a Výskum), návrh zvyšných kapitol (december 2022)


<br>

## **Ukážky z výskumu:**

<br>

**YOLOv7 + SORT**

Porovnanie výsledkov detekcie dvoch rôznych detektorov: YOLOv5 a YOLOv7. Test na rovnakom obraze:

<div style="display: flex; justify-content: center;">
  <img src="data\YOLOv5vsYOLOv7.png" alt="drawing" style="width:60%;"/>
</div>
<br>


Okrem toho, že verzia 7 bola presnejšia, je aj násobne rýchlejšia. Preto v ďalšom výskume budeme používať tento detektor. Na nasledujúcom gif-e je YOLOv7 aplikované na videu z nami získaného datasetu.



**YOLOv7 + SORT**

Objekty sú detekované nezávisle na snímku, nezachovávajú informáciu medzi nimi. Aby sme dokázali objekty identifikovať, využívame algoritmus SORT. Vďaka svojej jednoduchosti je jedným z najrýchlejších algoritmov, ktoré riešia tento problém identifikácie.


Okrem identifikácie bola zapracovaná funkcionalita sledovania bývalých pozícii daného objektu. Tie budeme využívať pri predikcii ďalšieho pohybu.
