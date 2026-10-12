#!/usr/bin/env python3

"""LANDAU V2a — MATRICA GUSTINE: whole-combination output states."""

import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)
N, K, SIZE, WINDOW, START = 39, 7, 8, 128, 2


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error
            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)
    return np.asarray(rows, dtype=int).reshape(-1, K)


def incoming_state(rows):
    rows = np.atleast_2d(rows)
    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), N+1),
    ))
    distance = np.diff(edges, axis=1)
    midpoint = .5*(edges[:, :-1]+edges[:, 1:])
    return (
        np.sqrt(distance/(N+1))
        * np.exp(2j*np.pi*midpoint/(N+1))
    )


def replacements(row):
    empty = np.setdiff1d(np.arange(1, N+1), row)
    count = K*len(empty)
    candidates = np.repeat(
        np.asarray(row)[None, :], count, axis=0
    )
    candidates[
        np.arange(count),
        np.repeat(np.arange(K), len(empty)),
    ] = np.tile(empty, K)
    return np.sort(candidates, axis=1)


def candidate_operator(seeds):
    neighbors = np.vstack([
        replacements(row) for row in seeds
    ])
    owners = np.repeat(
        np.arange(len(seeds)), K*(N-K)
    )
    unique, inverse = np.unique(
        neighbors, axis=0, return_inverse=True
    )
    known = {tuple(row) for row in seeds}
    keep = np.asarray([
        tuple(row) not in known for row in unique
    ])
    remap = np.full(len(unique), -1, dtype=int)
    remap[keep] = np.arange(int(keep.sum()))
    target = remap[inverse]
    valid = target >= 0

    seed_state = incoming_state(seeds)
    neighbor_state = incoming_state(neighbors)
    overlap = np.abs(np.einsum(
        "bi,bi->b",
        neighbor_state.conj(),
        seed_state[owners],
    ))**2

    norm = np.sqrt(np.bincount(
        owners[valid],
        weights=overlap[valid]**2,
        minlength=len(seeds),
    ))
    if np.any(norm <= 0):
        raise ArithmeticError(
            "Neispravan operator kandidata."
        )

    b = csr_matrix(
        (
            overlap[valid]/norm[owners[valid]],
            (target[valid], owners[valid]),
        ),
        shape=(int(keep.sum()), len(seeds)),
    )
    gram = (b.T @ b).toarray()
    values, vectors = np.linalg.eigh(
        (gram+gram.T)/2
    )
    if values.min() < -1e-9:
        raise ArithmeticError(
            "Operator nije pozitivno semidefinitan."
        )

    omega = np.sqrt(np.maximum(values, 0))
    candidates = np.vstack((seeds, unique[keep]))
    return candidates, b, omega, vectors


def prepare(history):
    begin = max(START, len(history)-WINDOW)
    states = incoming_state(history)

    x = np.einsum(
        "bi,bj->bij",
        states[begin-1:-1],
        states[begin-2:-2],
    ).reshape(-1, SIZE*SIZE)

    query = np.outer(
        states[-1], states[-2]
    ).ravel()

    kernel = np.abs(x.conj() @ x.T)**2
    query_kernel = np.abs(query.conj() @ x.T)**2

    seeds, labels = np.unique(
        history[begin:],
        axis=0,
        return_inverse=True,
    )

    # Svaka ciljna kolona predstavlja celu konkretnu kombinaciju.
    targets = np.eye(len(seeds))[labels]
    candidates, b, omega, vectors = candidate_operator(seeds)

    return {
        "kernel": kernel,
        "query": query_kernel,
        "targets": targets,
        "seeds": seeds,
        "candidates": candidates,
        "b": b,
        "omega": omega,
        "vectors": vectors,
    }


def initial_amplitudes(prepared, regularization):
    k = prepared["kernel"]
    coefficient = np.linalg.solve(
        k+regularization*np.eye(len(k)),
        prepared["targets"],
    )
    amplitudes = prepared["query"] @ coefficient
    norm = np.linalg.norm(amplitudes)

    if not np.isfinite(norm) or norm <= 1e-14:
        raise ArithmeticError(
            "Neispravno pocetno kvantno stanje."
        )
    return amplitudes.astype(complex)/norm


def evolve(prepared, amplitudes, tau):
    vectors = prepared["vectors"]
    omega = prepared["omega"]
    modes = vectors.conj().T @ amplitudes

    retained = vectors @ (
        np.cos(tau*omega)*modes
    )
    transferred = -1j*(
        prepared["b"] @ (
            vectors @ (
                tau*np.sinc(tau*omega/np.pi)*modes
            )
        )
    )

    state = np.r_[retained, transferred]
    norm = float(np.sum(np.abs(state)**2))

    if (
        not np.isfinite(norm)
        or not np.isclose(
            norm, 1., rtol=1e-9, atol=1e-9
        )
    ):
        raise ArithmeticError(
            "Evolucija ne cuva trag matrice gustine."
        )
    return state/np.sqrt(norm)


def density_diagonal(state):
    # rho = |state><state|; faktor cuva matricu bez velikog n*n niza.
    return np.abs(state)**2


def select(prepared, state):
    score = density_diagonal(state)
    best = int(np.argmax(score))
    return (
        tuple(map(int, prepared["candidates"][best])),
        float(score[best]),
    )


def predict(prepared, regularization, tau):
    amplitudes = initial_amplitudes(
        prepared, regularization
    )
    state = evolve(prepared, amplitudes, tau)
    combo, score = select(prepared, state)
    return combo, score, state


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True,
    )


def self_test():
    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [33, 34, 35, 36, 37, 38, 39],
        [4, 12, 18, 28, 35, 36, 37],
        [2, 7, 12, 18, 24, 30, 39],
    ]), (50, 1))

    incoming = incoming_state(rows)
    assert np.allclose(
        np.sum(np.abs(incoming)**2, axis=1), 1
    )

    prepared = prepare(rows[:170])
    a = initial_amplitudes(prepared, 1.)

    for tau in (0., 1.25, 1.55, 4.):
        state = evolve(prepared, a, tau)
        p = density_diagonal(state)
        assert np.all(p >= 0)
        assert np.isclose(p.sum(), 1)

        sub = np.outer(
            state[:8], state[:8].conj()
        )
        assert np.allclose(sub, sub.conj().T)
        assert np.linalg.eigvalsh(sub).min() >= -1e-12

    state = evolve(prepared, a, 0.)
    assert np.allclose(
        density_diagonal(state)[len(a):], 0
    )

    midpoint = (17, 18, 19, 20, 21, 22, 23)
    assert midpoint not in {
        tuple(row) for row in prepared["candidates"]
    }

    result = predict(prepared, 1., 1.55)
    changed = rows.copy()
    changed[170:] = [3, 8, 13, 19, 25, 31, 38]
    altered = predict(
        prepare(changed[:170]), 1., 1.55
    )

    assert result[:2] == altered[:2]
    assert np.array_equal(result[2], altered[2])
    assert result[:2] == predict(
        prepared, 1., 1.55
    )[:2]

    candidates = prepared["candidates"]
    assert np.all(np.diff(candidates, axis=1) > 0)
    assert candidates.min() >= 1
    assert candidates.max() <= N

    print(
        "Self-test OK: cela ortogonalna stanja, gustina, "
        "unitarna evolucija i hronologija."
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)

    if validation_count < 16 or test_count < 16:
        raise ValueError(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-test_count
    begin = split-validation_count
    if begin < START+128:
        raise ValueError(
            f"{path}: premalo istorije za zadatu validaciju i test."
        )

    settings = tuple(
        (reg, tau)
        for reg in (.1, 1., 10.)
        for tau in (0., 1.25, 1.55)
    )
    validation = [[] for _ in settings]

    print(
        f"Ulaz: {Path(path).resolve()}; redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows[:t])
        amplitudes = {
            reg: initial_amplitudes(prepared, reg)
            for reg in (.1, 1., 10.)
        }

        for j, (reg, tau) in enumerate(settings):
            state = evolve(
                prepared, amplitudes[reg], tau
            )
            combo, _ = select(prepared, state)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: {t-begin+1}/{validation_count}",
                flush=True,
            )

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    regularization, tau = settings[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"regularizacija={regularization:g}; tau={tau:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    without_evolution = []

    for t in range(split, len(rows)):
        prepared = prepare(rows[:t])
        amplitudes = initial_amplitudes(
            prepared, regularization
        )

        combo, _ = select(
            prepared,
            evolve(prepared, amplitudes, tau),
        )
        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        reference, _ = select(
            prepared,
            evolve(prepared, amplitudes, 0.),
        )
        without_evolution.append(hits(reference, rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True,
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V2a MATRICA GUSTINE", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize("Ista gustina bez evolucije", without_evolution)

    prepared = prepare(rows)
    combo, score, state = predict(
        prepared, regularization, tau
    )

    print(
        f"Poslednji red: {','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Born_skor={score:.10f}; "
        f"Tr(rho)={density_diagonal(state).sum():.10f}; "
        f"kandidata={len(prepared['candidates'])}"
    )
    print(
        "Skor se odnosi na kandidatni skup: "
        "istorijske kombinacije i zamene jednog broja."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        help="Bez putanja obradjuje sva tri DEFAULTS fajla.",
    )
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; regularizacija=1; tau=0; prosek=1.4375
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V2a MATRICA GUSTINE: prosek=1.3906; pogodaka 0..7=[14, 25, 14, 9, 1, 1, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Ista gustina bez evolucije: prosek=1.3906; pogodaka 0..7=[14, 25, 14, 9, 1, 1, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 8,x,10,y,13,z,34
Born_skor=0.1763516905; Tr(rho)=1.0000000000; kandidata=28772
Skor se odnosi na kandidatni skup: istorijske kombinacije i zamene jednog broja.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; regularizacija=0.1; tau=0; prosek=1.4792
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V2a MATRICA GUSTINE: prosek=1.1562; pogodaka 0..7=[12, 35, 12, 5, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Ista gustina bez evolucije: prosek=1.1562; pogodaka 0..7=[12, 35, 12, 5, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 2,x,15,y,27,z,29
Born_skor=0.2293035520; Tr(rho)=1.0000000000; kandidata=28772
Skor se odnosi na kandidatni skup: istorijske kombinacije i zamene jednog broja.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; regularizacija=0.1; tau=1.55; prosek=1.1042
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V2a MATRICA GUSTINE: prosek=1.2031; pogodaka 0..7=[9, 35, 18, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Ista gustina bez evolucije: prosek=1.0781; pogodaka 0..7=[13, 36, 12, 3, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,5,y,19,z,30
Born_skor=0.0007436729; Tr(rho)=1.0000000000; kandidata=28772
Skor se odnosi na kandidatni skup: istorijske kombinacije i zamene jednog broja.
"""



"""
Rangirati konkretne cele kombinacije kao odvojena stanja, umesto da iz interpolirane matrice izvodi srednji raspored brojeva.
Izlazna stanja biće tačne sedmorke brojeva. 
Matrica gustine će određivati njihove skorove, bez računanja prosečnih položaja. 
Kandidatni skup činiće stvarne istorijske kombinacije i njihove zamene jednog broja.


Proseci su za poslednja 64 test-kola:

Fajl / sledeći red      Predlog	            V2a	    Prethodna kombinacija
Objedinjeni / 4699	    8,x,10,y,13,z,34    1,3906	1,2188
Loto / 2972	            2,x,15,y,27,z,29	1,1562	1,2969
Loto Plus / 1728	    1,x,5,y,19,z,30	    1,2031	1,2969

Kandidati su istorijske kombinacije i njihove zamene jednog broja.
"""
