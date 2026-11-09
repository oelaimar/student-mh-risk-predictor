## Dictionnaire de données

Source : [Kaggle – Mental Health Risk Prediction Dataset (Students)](https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students)
Jeu de données **synthétique** (étudiants fictifs) : 2 000 lignes, 17 colonnes (16 variables + 1 cible).

| # | Nom | Type | Description | Valeurs possibles | Unité | Manquants | Traitement prévu |
|---|-----|------|-------------|-------------------|-------|-----------|------------------|
| 1 | `Student_ID` | Identifiant (texte) | Identifiant unique de l'étudiant | STU0001, STU0002… | — | 0 % | Supprimée (aucune information prédictive) |
| 2 | `Age` | Numérique discrète | Âge de l'étudiant | 18 – 30 | années | 0 % | Standardisation |
| 3 | `Gender` | Catégorielle nominale | Genre déclaré | Male, Female, Non-binary | — | 0 % | One-hot encoding |
| 4 | `Year_of_Study` | Numérique discrète (ordinale) | Année d'études en cours | 1 – 5 | année | 0 % | Standardisation |
| 5 | `Sleep_Hours` | Numérique continue | Durée moyenne de sommeil par nuit | 3.5 – 9.5 | heures / nuit | ~3,0 % | Imputation médiane + capping IQR + standardisation |
| 6 | `Study_Hours_Per_Week` | Numérique continue | Heures d'étude déclarées par semaine | 5 – 55 | heures / semaine | 0 % | Capping IQR + standardisation |
| 7 | `GPA` | Numérique continue | Moyenne cumulée | 1.5 – 4.0 | échelle /4.0 | ~1,9 % | Imputation médiane + standardisation |
| 8 | `Social_Media_Hours` | Numérique continue | Temps quotidien moyen sur les réseaux sociaux | 0.5 – 10.0 | heures / jour | ~3,0 % | Imputation médiane + standardisation |
| 9 | `Screen_Time_Hours` | Numérique continue | Temps d'écran total quotidien | 1.0 – 14.0 | heures / jour | 0 % | Capping IQR + standardisation |
| 10 | `Physical_Activity` | Catégorielle ordinale | Fréquence d'activité physique hebdomadaire | Low < Moderate < High | — | 0 % | Encodage ordinal |
| 11 | `Financial_Stress` | Numérique continue (score) | Stress financier auto-évalué | 1 – 10 | score /10 | ~4,5 % | Imputation médiane + standardisation |
| 12 | `Family_Support` | Numérique continue (score) | Soutien familial perçu | 1 – 10 | score /10 | ~2,7 % | Imputation médiane + standardisation |
| 13 | `Attendance_Rate` | Numérique continue | Taux de présence en cours | 30 – 100 | % | ~2,4 % | Imputation médiane + standardisation |
| 14 | `Part_Time_Job` | Binaire | L'étudiant a un emploi à temps partiel | Yes, No | — | 0 % | Encodage binaire (0/1) |
| 15 | `Diet_Quality` | Catégorielle ordinale | Qualité de l'alimentation déclarée | Poor < Average < Good | — | ~3,5 % | Imputation mode + encodage ordinal |
| 16 | `Previous_Mental_Health_History` | Binaire | Antécédent diagnostiqué de trouble de santé mentale | Yes, No | — | 0 % | Encodage binaire (0/1) |
| 17 | `Mental_Health_Risk` | Catégorielle ordinale | **Variable cible** : niveau de risque de santé mentale | Low, Medium, High | — | 0 % | Cible (classification multiclasse) |

**Remarques**
- Les pourcentages de valeurs manquantes proviennent de la documentation Kaggle et sont à vérifier pendant l'EDA.
- Les scores (`Financial_Stress`, `Family_Support`) et l'alimentation sont **auto-déclarés** (biais de déclaration possible).
- Le jeu de données est **synthétique** : la cible est dérivée d'un score de risque pondéré avec du bruit. Les résultats ne se généralisent pas à une population réelle.