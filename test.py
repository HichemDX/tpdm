def generate_candidates(Lk_minus_1, k):


    candidats = set()  # Ensemble des candidats
    Lk_minus_1 = list(Lk_minus_1)  # Convertir en liste pour accéder aux indices

    for i in range(len(Lk_minus_1)):
        for j in range(i + 1, len(Lk_minus_1)):  # Comparer chaque paire
            # Union des deux ensembles
            union_set = Lk_minus_1[i].union(Lk_minus_1[j])

            # Vérifier que l'union contient exactement k éléments
            if len(union_set) == k:
                candidats.add(frozenset(union_set))

    return candidats


# Données : L1
L1 = {
    frozenset({'Apple'}),
    frozenset({'Banana'}),
    frozenset({'Carrot'}),
    frozenset({'Dates'})
}

# Générer C2 (2-itemsets candidats)
C2 = generate_candidates(L1, 2)

# Affichage des candidats
print("C2 (2-itemsets candidats) :")
for itemset in C2:
    print(set(itemset))
