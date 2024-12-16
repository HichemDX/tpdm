import pandas as pd
import numpy as np
from itertools import chain, combinations


# Charger les données et effectuer le prétraitement
print("Chargement des données...")
data = pd.read_csv('./NewDatasetExos.csv', sep=';')

# Supprimer les 7 premières colonnes inutiles
df_cleaned = data.iloc[:, 7:]
print(f"Dimensions des données après nettoyage : {df_cleaned.shape}")

# Grouper les transactions par 'ID' et 'Set'
transactions = df_cleaned.groupby(['ID', 'Set'], group_keys=False).apply(
    lambda group: list(zip(group['Label'], group['Category'])),
    include_groups=False
).reset_index(name="Items")
print(f"Nombre de transactions générées : {len(transactions)}")

# Extraire les items uniques (combinaisons uniques de Label et Category)
unique_items = df_cleaned[['Label', 'Category']].drop_duplicates().reset_index(drop=True)
print(f"Nombre d'items uniques : {len(unique_items)}")


### Fonction pour générer les candidats C1
def generate_C1(unique_items):
    """
    Génère les 1-itemsets candidats (C1) à partir des items uniques.
    """
    return {frozenset([(row['Label'], row['Category'])]) for _, row in unique_items.iterrows()}


### Fonction pour générer les candidats Ck (k-itemsets)
def generate_candidates(Lk_minus_1, k):
    """
    Génère les k-itemsets candidats (Ck) à partir des itemsets fréquents L(k-1).
    """
    candidats = set()
    Lk_minus_1 = list(Lk_minus_1)
    for i in range(len(Lk_minus_1)):
        for j in range(i + 1, len(Lk_minus_1)):
            union_set = Lk_minus_1[i].union(Lk_minus_1[j])
            if len(union_set) == k:
                candidats.add(frozenset(union_set))
    return candidats


### Fonction pour calculer le support des itemsets
def calculate_support(Ck, transactions):
    """
    Calcule le support pour chaque itemset candidat dans Ck.
    """
    support_count = {candidat: 0 for candidat in Ck}
    for transaction in transactions:
        transaction_set = frozenset(transaction)
        for candidat in Ck:
            print('candidat', candidat)
            if candidat.issubset(transaction_set):
                print("Transaction : ", transaction)
                support_count[candidat] += 1
    total_transactions = len(transactions)
    print("Items : ", support_count)
    support = {itemset: count for itemset, count in support_count.items()}
    return support


### Fonction pour filtrer les itemsets fréquents (Lk)
def generate_frequent_itemsets(Ck, support, supp_min):
    """
    Filtre les itemsets fréquents (Lk) à partir des candidats Ck.
    """
    return {itemset for itemset, supp in support.items() if supp >= supp_min}


### Fonction pour générer toutes les règles d'association
def generate_association_rules(Lk):
    """
    Génère toutes les règles d'association possibles à partir des itemsets fréquents Lk.
    """
    rules = []
    for itemset in Lk:
        if len(itemset) > 1:
            subsets = list(map(frozenset, [x for x in powerset(itemset) if 0 < len(x) < len(itemset)]))
            for A in subsets:
                B = itemset - A
                rules.append((A, B))
    return rules


### Fonction pour générer toutes les sous-parties d'un ensemble
def powerset(iterable):
    """
    Génère toutes les sous-parties possibles d'un ensemble.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


### Fonction pour calculer la confiance des règles d'association
def calculate_confidence(rules, support):
    """
    Calcule la confiance pour chaque règle d'association.
    """
    confidence = {}
    for A, B in rules:
        if A in support and (A | B) in support:
            confidence[(A, B)] = support[A | B] / support[A]
    return confidence


### Partie principale : Algorithme Apriori avec génération des règles
supp_min = 0.2  # Seuil minimal de support
conf_min = 0.5  # Seuil minimal de confiance

# Générer les 1-itemsets candidats et leurs supports
print("\nGénération de C1 (1-itemsets candidats)...")
C1 = generate_C1(unique_items)
print(f"C1 : {C1}")

print("\nCalcul du support pour C1...")
support_C1 = calculate_support(C1, transactions['Items'])
print(f"Support pour C1 : {support_C1}")

print("\nGénération de L1 (itemsets fréquents)...")
L1 = generate_frequent_itemsets(C1, support_C1, supp_min)
print(f"L1 (itemsets fréquents) : {L1}")

# Initialiser la boucle pour générer Lk et les règles
k = 2
Lk = L1
all_rules = []

while Lk:
    print(f"\n--- Génération de L{k} (itemsets fréquents de taille {k}) ---")

    # Générer les candidats Ck
    print("\nGénération des candidats Ck...")
    Ck = generate_candidates(Lk, k)
    print(f"C{k} : {Ck}")

    # Calculer le support de Ck
    print("\nCalcul du support pour Ck...")
    support_Ck = calculate_support(Ck, transactions['Items'])
    print(f"Support pour C{k} : {support_Ck}")

    # Filtrer pour obtenir Lk
    print("\nFiltrage pour obtenir Lk...")
    Lk = generate_frequent_itemsets(Ck, support_Ck, supp_min)
    print(f"L{k} (itemsets fréquents) : {Lk}")

    if Lk:
        # Générer les règles d'association
        print("\nGénération des règles d'association...")
        rules = generate_association_rules(Lk)
        print(f"Règles générées pour L{k} :")
        for A, B in rules:
            print(f"  {set(A)} ⇒ {set(B)}")

        # Calculer la confiance des règles
        print("\nCalcul de la confiance pour les règles...")
        confidence = calculate_confidence(rules, {**support_C1, **support_Ck})
        print(f"Confiance des règles : {confidence}")

        # Filtrer les règles selon conf_min
        print("\nFiltrage des règles selon la confiance minimale...")
        filtered_rules = {rule: conf for rule, conf in confidence.items() if conf >= conf_min}
        all_rules.extend(filtered_rules)
        print(f"Règles fréquentes (confiance ≥ {conf_min}) :")
        for rule, conf in filtered_rules.items():
            print(f"  {set(rule[0])} ⇒ {set(rule[1])} (confiance : {conf:.2f})")

    k += 1
