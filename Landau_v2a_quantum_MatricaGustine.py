#!/usr/bin/env python3

"""Landau Quantum v2: conditional density matrix for complete 7/39 layouts."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"

N, K, D, START, WINDOW = 39, 7, 16, 32, 768


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not value.strip() for value in row):
                continue
            try:
                numbers = [int(value.strip()) for value in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= number <= N for number in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def shape(rows):
    rows = np.atleast_2d(rows)
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N + 1),
    ))
    empty_gaps = np.diff(edges, axis=1) - 1

    return np.column_stack((
        rows / (N + 1),
        empty_gaps / (N - K + 1),
    ))


def states(rows):
    # Jedna referentna amplituda, sedam polozaja i osam praznina.
    angles = 2 * np.pi * shape(rows)
    return np.column_stack((
        np.ones(len(angles)),
        np.exp(1j * angles),
    )) / np.sqrt(D)


def training_data(rows):
    layouts = shape(rows)
    features = np.asarray([
        np.r_[
            layouts[t-1],
            layouts[t-2],
            layouts[t-8:t].mean(axis=0),
            layouts[t-32:t].mean(axis=0),
        ]
        for t in range(START, len(rows) + 1)
    ])

    vectors = states(rows)
    density = vectors[:, :, None] * vectors[:, None, :].conj()
    return features, density.reshape(len(rows), D * D)


def project_density(matrix):
    # Hermitska, pozitivno semidefinitna matrica sa tragom jedan.
    hermitian = (matrix + matrix.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    eigenvalues = np.maximum(eigenvalues, 0)
    total = float(eigenvalues.sum())

    if not np.isfinite(total) or total <= 0:
        raise ArithmeticError("Neispravna matrica gustine.")

    return (
        (eigenvectors * (eigenvalues / total))
        @ eigenvectors.conj().T
    )


def forecast(features, densities, t, alpha):
    # Ciljni red t i svi kasniji redovi iskljuceni su iz podesavanja.
    begin = max(START, t - WINDOW)
    x = features[begin-START:t-START]
    y = densities[begin:t]

    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-8)
    a = np.column_stack((
        np.ones(len(x)),
        (x - center) / scale,
    ))
    penalty = np.diag(np.r_[
        0., np.full(x.shape[1], alpha)
    ])

    coefficients = np.linalg.solve(
        a.T @ a + penalty,
        a.T @ y,
    )
    query = np.r_[
        1., (features[t-START] - center) / scale
    ]
    return project_density(
        (query @ coefficients).reshape(D, D)
    )


def evaluate(rows, density):
    vectors = states(rows)
    return np.einsum(
        "bi,ij,bj->b",
        vectors.conj(),
        density,
        vectors,
        optimize=True,
    ).real


def seeds(history):
    # Istorijski i fiksni geometrijski poceci; bez uzorkovanja.
    candidates = [
        tuple(map(int, row)) for row in history[-32:]
    ]
    for step in range(1, 7):
        for first in range(1, N - (K - 1) * step + 1):
            candidates.append(tuple(
                first + j * step for j in range(K)
            ))
    return np.asarray(sorted(set(candidates)), dtype=int)


def search(history, density):
    # Deterministicka pretraga iz vise pocetnih konfiguracija.
    # Dostize koordinatne optimume, bez potvrde globalnog optimuma.
    current = seeds(history)
    values = np.arange(1, N + 1)

    for iteration in range(1, 101):
        blocks = [current]
        owners = [np.arange(len(current))]

        for j in range(K):
            lower = (
                current[:, j-1] + 1
                if j else np.ones(len(current), dtype=int)
            )
            upper = (
                current[:, j+1] - 1
                if j < K-1 else np.full(len(current), N)
            )

            who, which = np.nonzero(
                (values[None, :] >= lower[:, None])
                & (values[None, :] <= upper[:, None])
            )
            neighbors = current[who].copy()
            neighbors[:, j] = values[which]
            blocks.append(neighbors)
            owners.append(who)

        candidates = np.vstack(blocks)
        owners = np.concatenate(owners)
        scores = evaluate(candidates, density)

        best_scores = np.full(len(current), -np.inf)
        np.maximum.at(best_scores, owners, scores)

        indices = np.arange(len(candidates))
        eligible = scores >= best_scores[owners] - 1e-12
        chosen = np.full(
            len(current), len(candidates), dtype=int
        )
        np.minimum.at(
            chosen,
            owners,
            np.where(eligible, indices, len(candidates)),
        )

        updated = candidates[chosen]
        if np.array_equal(updated, current):
            final_scores = evaluate(current, density)
            best = int(np.argmax(final_scores))
            return (
                tuple(map(int, current[best])),
                float(final_scores[best]),
                iteration,
            )
        current = updated

    raise RuntimeError(
        "Pretraga nije dostigla koordinatni optimum u 100 koraka."
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}"
    )


def self_test():
    rows = np.asarray([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
    ])
    vectors = states(rows)

    assert np.allclose(
        np.sum(np.abs(vectors)**2, axis=1), 1
    )
    for vector, row in zip(vectors, rows):
        rho = project_density(
            np.outer(vector, vector.conj())
        )
        assert np.isclose(np.trace(rho).real, 1)
        assert np.linalg.eigvalsh(rho).min() >= -1e-12
        assert np.isclose(evaluate([row], rho)[0], 1)

    history = np.tile(rows, (70, 1))
    x, y = training_data(history)
    t = 170
    original = forecast(x, y, t, 10.)

    changed = history.copy()
    changed[t:] = [2, 7, 12, 18, 24, 30, 39]
    changed_x, changed_y = training_data(changed)
    assert np.array_equal(
        original,
        forecast(changed_x, changed_y, t, 10.),
    )

    found, score, _ = search(history[:t], original)
    again = search(history[:t], original)
    assert (found, score) == again[:2]
    assert len(set(found)) == K
    assert list(found) == sorted(found)
    assert 1 <= found[0] < found[-1] <= N

    for j in range(K):
        lo = found[j-1] + 1 if j else 1
        hi = found[j+1] - 1 if j < K-1 else N
        neighbors = np.tile(found, (hi-lo+1, 1))
        neighbors[:, j] = np.arange(lo, hi+1)
        assert (
            evaluate(neighbors, original).max()
            <= score + 1e-12
        )

    print(
        "Self-test OK: gustina, skor, hronologija, "
        "ponovljivost i lokalni optimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows) - args.test
    validation_start = split - args.validation
    if validation_start < START + 128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    features, densities = training_data(rows)
    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )
    print(
        "Validacija: podesavanje matrice gustine.",
        flush=True,
    )

    alphas = (1., 10., 100.)
    validation_scores = []

    for alpha in alphas:
        scores = []
        for t in range(validation_start, split):
            rho = forecast(features, densities, t, alpha)
            combo, _, _ = search(rows[:t], rho)
            scores.append(hits(combo, rows[t]))
        validation_scores.append(float(np.mean(scores)))

    alpha = alphas[int(np.argmax(validation_scores))]
    print(
        f"Validacija: redovi {validation_start+1}..{split}; "
        f"alpha={alpha:g}; "
        f"prosek={max(validation_scores):.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    structural_scores = []

    for t in range(split, len(rows)):
        rho = forecast(features, densities, t, alpha)
        combo, _, _ = search(rows[:t], rho)
        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        # Isti prikaz i pretraga, bez vremenskog uslovljavanja.
        unconditional = densities[
            max(0, t-WINDOW):t
        ].mean(axis=0).reshape(D, D)

        reference, _, _ = search(
            rows[:t],
            project_density(unconditional),
        )
        structural_scores.append(hits(reference, rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("Landau Quantum v2", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize(
        "Gustina bez vremenskog uslovljavanja",
        structural_scores,
    )

    rho = forecast(
        features, densities, len(rows), alpha
    )
    combo, score, iterations = search(rows, rho)

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"S={score:.10f}; iteracija={iterations}; "
        f"Tr(rho^2)={np.trace(rho @ rho).real:.6f}"
    )
    print(
        "S je skor rangiranja. Pretraga je deterministicka; "
        "globalni maksimum nije potvrden."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: podesavanje matrice gustine.
Validacija: redovi 4587..4634; alpha=10; prosek=1.1250
Test: redovi 4635..4698
Landau Quantum v2: prosek=1.0625; pogodaka 0..7=[16, 32, 12, 4, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Gustina bez vremenskog uslovljavanja: prosek=1.1719; pogodaka 0..7=[13, 30, 18, 3, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 6,11,15,21,25,31,35
S=0.6367982401; iteracija=38; Tr(rho^2)=0.437876
S je skor rangiranja. Pretraga je deterministicka; globalni maksimum nije potvrden.





Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: podesavanje matrice gustine.
Validacija: redovi 2860..2907; alpha=100; prosek=1.1042
Test: redovi 2908..2971
Landau Quantum v2: prosek=1.3438; pogodaka 0..7=[14, 24, 18, 6, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Gustina bez vremenskog uslovljavanja: prosek=1.1875; pogodaka 0..7=[11, 34, 15, 4, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 6,10,15,20,24,30,35
S=0.6762174535; iteracija=40; Tr(rho^2)=0.487903
S je skor rangiranja. Pretraga je deterministicka; globalni maksimum nije potvrden.





Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: podesavanje matrice gustine.
Validacija: redovi 1616..1663; alpha=1; prosek=1.3333
Test: redovi 1664..1727
Landau Quantum v2: prosek=1.0938; pogodaka 0..7=[12, 38, 10, 4, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Gustina bez vremenskog uslovljavanja: prosek=1.2344; pogodaka 0..7=[13, 28, 18, 5, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 5,11,15,20,25,30,35
S=0.6449312415; iteracija=36; Tr(rho^2)=0.447870
S je skor rangiranja. Pretraga je deterministicka; globalni maksimum nije potvrden.
"""







"""
Ruski teoretski fizicar Lev Davidovich Landau. 

Quantum Theory & Solid State Physics

Matrica gustine potiče iz kvantne teorije, konkretno iz Landauovog rada iz 1927.

V2 — LANDAU QUANTUM pokriva samo ideju matrice gustine.

Landauova ideja matrice gustine: opisati više mogućih struktura i njihove međusobne veze, umesto svesti predikciju na jedan centralni raspored.
Landau je taj pristup razvio u radu The Damping Problem in Wave Mechanics iz 1927. Kod povezanih kvantnih sistema pokazuje zašto opis pomoću jedne talasne funkcije podsistema nije dovoljan i uvodi matrični opis njegovog stanja. Originalni rad, prvi odeljak
Korisna ideja za moj loto model bila bi da sačuva različite oblike distribucije. Na primer, istorija može sugerisati dva odvojena rasporeda: jedan sa grupisanjem u donjem delu raspona, drugi sa grupisanjem u gornjem. Njihov prosek daje srednji raspored koji možda slabo predstavlja oba. V1 upravo procenjuje jedan ciljni raspored; v2 bi mogao da zadrži više strukturnih mogućnosti.

- Svaku celu kombinaciju predstaviti vektorom koji opisuje položaje, razmake i grupisanje.
- Iz prethodnih kola učiti matricu, koja opisuje trenutno stanje tih struktura i njihove veze.
- Kandidata ocenjivati izrazom  

Ovde je \(v(C)\) matematički opis kombinacije, a \(S_t(C)\) njen skor za rangiranje. Matrica bi bila ograničena tako da skor ostane nenegativan. Članovi van dijagonale omogućavaju da zajedničko pojavljivanje dva strukturna svojstva utiče na ocenu — recimo, određeno grupisanje zajedno sa velikom prazninom.
To je konkretan pravac za Landau Quantum v2: predviđati stanje struktura, pa iz njega rangirati cele kombinacije. Kvantni formalizam ovde služi kao matematička inspiracija. Više mogućih oblika i korisne veze između njih morali bi stvarno naučiti iz podataka; sama matrica ih ne obezbeđuje.



Matrica gustine prošla je provere normalizacije i nenegativnosti, a promena budućih redova nije promenila raniju predikciju. Hronološka validacija je završena; sada se proveravaju poslednja 64 kola. Pretraga je deterministička i koristi više početnih konfiguracija, ali ne potvrđuje globalni maksimum.


Samostalan model matrice gustine, izračunat NumPy-em.

Predlog za red 4699:
6, 11, 15, 21, 25, 31, 35


Na poslednja 64 kola:

Metod	                                        Prosečan broj pogodaka
Quantum v2	                                    1,0625
Prethodna kombinacija	                        1,2188
Matrica gustine bez vremenskog uslovljavanja	1,1719


V2 na ovom testu nije pokazala prednost. 
Prošli su provera matrice gustine, izolacija budućih podataka, ponovljivost i provera pretrage. 
Pretraga nalazi najviši skor među deterministički pronađenim kandidatima; globalni maksimum nije potvrđen.
"""
