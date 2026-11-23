"""Deterministic, SymPy-backed Indonesian STEM problem generators for IndoSTEM-OCR."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from pindai.dataset.schema import (
    AugmentationProfile,
    FormatKind,
    LayoutStyle,
    Split,
    StemSample,
    Subject,
)


@dataclass(frozen=True)
class GeneratorSpec:
    template_id: str
    subject: Subject
    topic: str
    format_kind: FormatKind
    fn: Callable[[random.Random, int], str]


def _format_options(rng: random.Random, correct: str, distractors: list[str]) -> str:
    unique: list[str] = [correct]
    for item in distractors:
        if item not in unique:
            unique.append(item)
        if len(unique) == 5:
            break
    while len(unique) < 5:
        unique.append(f"${rng.randint(-15, 25)}$")
    rng.shuffle(unique)
    labels = ["A", "B", "C", "D", "E"]
    return "\n".join(f"{label}. {opt}" for label, opt in zip(labels, unique, strict=True))


def _gen_math_quadratic(rng: random.Random, number: int) -> str:
    x = sp.Symbol("x")
    r1 = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
    r2 = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
    a = rng.choice([1, 1, 2, 3])
    expr = sp.expand(a * (x - r1) * (x - r2))
    val = r1 + r2
    correct = f"${val}$"
    distractors = [f"${-val}$", f"${val + 2}$", f"${r1 * r2}$", f"${val - 1}$", f"${val + 3}$"]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Diketahui persamaan kuadrat ${sp.latex(expr)} = 0$ memiliki akar-akar $x_1$ dan $x_2$. "
        f"Nilai dari $x_1 + x_2$ adalah ...\n\n"
        f"{options}"
    )


def _gen_math_integral(rng: random.Random, number: int) -> str:
    x = sp.Symbol("x")
    a = rng.choice([3, 6, 9])
    b = rng.choice([-4, -2, 2, 4, 6])
    c = rng.randint(-5, 7)
    lo = rng.choice([0, 1])
    hi = lo + rng.choice([1, 2, 3])
    integrand = a * x**2 + b * x + c
    result = sp.integrate(integrand, (x, lo, hi))
    correct = f"${sp.latex(result)}$"
    distractors = [
        f"${sp.latex(result + 2)}$",
        f"${sp.latex(result - 2)}$",
        f"${sp.latex(result + a)}$",
        f"${sp.latex(-result)}$",
    ]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Hitung nilai integral tentu berikut:\n\n"
        f"$$\\int_{{{lo}}}^{{{hi}}} \\left({sp.latex(integrand)}\\right) \\, dx$$\n\n"
        f"{options}"
    )


def _gen_math_limit(rng: random.Random, number: int) -> str:
    x = sp.Symbol("x")
    k = rng.choice([-4, -3, -2, 2, 3, 4, 5])
    m = rng.choice([-5, -3, -1, 1, 3, 5])
    num = sp.expand((x - k) * (x + m))
    den = x - k
    ans = k + m
    correct = f"${ans}$"
    distractors = [f"${ans + 1}$", f"${ans - 1}$", f"${k - m}$", f"${-ans}$", f"${ans + 2}$"]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Tentukan nilai limit fungsi aljabar berikut:\n\n"
        f"$$\\lim_{{x \\to {k}}} \\frac{{{sp.latex(num)}}}{{{sp.latex(den)}}}$$\n\n"
        f"{options}"
    )


def _gen_math_matrix(rng: random.Random, number: int) -> str:
    a, b, c, d = [rng.randint(-5, 7) for _ in range(4)]
    mat = sp.Matrix([[a, b], [c, d]])
    det = int(mat.det())
    correct = f"${det}$"
    distractors = [
        f"${-det}$",
        f"${a * d + b * c}$",
        f"${det + 3}$",
        f"${det - 2}$",
        f"${det + 1}$",
    ]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Diketahui matriks $A$ sebagai berikut:\n\n"
        f"$$A = {sp.latex(mat)}$$\n\n"
        f"Determinan dari matriks $A$, yaitu $\\det(A)$, bernilai ...\n\n"
        f"{options}"
    )


def _gen_math_derivative_essay(rng: random.Random, number: int) -> str:
    x = sp.Symbol("x")
    a = rng.choice([1, 2, 3])
    b = rng.choice([-4, -2, 2, 4])
    c = rng.randint(-6, 6)
    x0 = rng.choice([-2, -1, 1, 2, 3])
    f = a * x**3 + b * x**2 + c * x
    df = sp.diff(f, x)
    slope = df.subs(x, x0)
    return (
        f"### Soal {number} (Uraian)\n\n"
        f"Diberikan fungsi $f(x) = {sp.latex(f)}$.\n\n"
        f"1. Tentukan turunan pertama $f'(x)$.\n"
        f"2. Hitung gradien garis singgung kurva di titik dengan absis $x = {x0}$.\n\n"
        f"**Ringkasan Kunci:** $f'(x) = {sp.latex(df)}$ sehingga $m = f'({x0}) = {sp.latex(slope)}$."
    )


def _gen_math_stats_table(rng: random.Random, number: int) -> str:
    scores = [60, 70, 80, 90, 100]
    freqs = [rng.randint(2, 8) for _ in scores]
    total_f = sum(freqs)
    total_fx = sum(s * f for s, f in zip(scores, freqs, strict=True))
    mean = sp.Rational(total_fx, total_f)
    rows = "\n".join(f"| {s} | {f} |" for s, f in zip(scores, freqs, strict=True))
    return (
        f"### Soal {number}\n\n"
        f"Perhatikan tabel distribusi nilai ujian Matematika kelas XII berikut:\n\n"
        f"| Nilai ($x_i$) | Frekuensi ($f_i$) |\n"
        f"| --- | --- |\n"
        f"{rows}\n\n"
        f"Hitung nilai rata-rata ($\\bar{{x}}$) data tersebut menggunakan rumus:\n\n"
        f"$$\\bar{{x}} = \\frac{{\\sum f_i x_i}}{{\\sum f_i}} = \\frac{{{total_fx}}}{{{total_f}}} \\approx {float(mean):.2f}$$"
    )


def _gen_physics_glbb(rng: random.Random, number: int) -> str:
    v0 = rng.choice([4, 6, 8, 10, 12])
    a = rng.choice([2, 3, 4, 5])
    t = rng.choice([3, 4, 5, 6])
    s = v0 * t + (a * t * t) // 2
    if (a * t * t) % 2 != 0:
        t = 4
        s = v0 * t + (a * t * t) // 2
    correct = f"${s}\\text{{ m}}$"
    distractors = [
        f"${v0 * t}\\text{{ m}}$",
        f"${s + 10}\\text{{ m}}$",
        f"${s - 8}\\text{{ m}}$",
        f"${v0 * t + a * t * t}\\text{{ m}}$",
    ]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Sebuah mobil bergerak lurus dengan kecepatan awal $v_0 = {v0}\\text{{ m/s}}$ "
        f"dan percepatan tetap $a = {a}\\text{{ m/s}}^2$. "
        f"Jarak tempuh mobil setelah $t = {t}\\text{{ s}}$ berdasarkan persamaan "
        f"$s = v_0 t + \\frac{{1}}{{2}} a t^2$ adalah ...\n\n"
        f"{options}"
    )


def _gen_physics_ohm_table(rng: random.Random, number: int) -> str:
    r_val = rng.choice([4, 5, 8, 10, 12])
    voltages = [r_val * i for i in (1, 2, 3, 4)]
    currents = [1, 2, 3, 4]
    rows = "\n".join(
        f"| {idx} | {v} | {i} |"
        for idx, (v, i) in enumerate(zip(voltages, currents, strict=True), 1)
    )
    return (
        f"### Soal {number}\n\n"
        f"Tabel berikut menunjukkan hasil pengukuran tegangan ($V$) dan kuat arus ($I$) "
        f"pada sebuah resistor ohmik:\n\n"
        f"| Percobaan | Tegangan $V$ (volt) | Arus $I$ (ampere) |\n"
        f"| --- | --- | --- |\n"
        f"{rows}\n\n"
        f"Berdasarkan Hukum Ohm $V = I \\cdot R$, besar hambatan $R$ adalah ${r_val}\\,\\Omega$."
    )


def _gen_physics_energy_essay(rng: random.Random, number: int) -> str:
    m = rng.choice([2, 4, 5, 6])
    h = rng.choice([5, 10, 15, 20])
    g = 10
    ep = m * g * h
    return (
        f"### Soal {number} (Uraian)\n\n"
        f"Sebuah benda bermassa $m = {m}\\text{{ kg}}$ dijatuhkan bebas dari ketinggian "
        f"$h = {h}\\text{{ m}}$ di atas tanah ($g = {g}\\text{{ m/s}}^2$).\n\n"
        f"Hitung energi potensial awal benda menggunakan $E_p = m g h$ dan tentukan energi kinetiknya "
        f"tepat sebelum menyentuh tanah.\n\n"
        f"**Catatan:** $E_p = {m} \\times {g} \\times {h} = {ep}\\text{{ J}}$."
    )


def _gen_chemistry_stoichiometry(rng: random.Random, number: int) -> str:
    mol_h2 = rng.choice([2, 4, 6, 8])
    mol_h2o = mol_h2
    mr_h2o = 18
    mass = mol_h2o * mr_h2o
    correct = f"${mass}\\text{{ gram}}$"
    distractors = [
        f"${mass // 2}\\text{{ gram}}$",
        f"${mass + 18}\\text{{ gram}}$",
        f"${mass * 2}\\text{{ gram}}$",
        f"${mass - 9}\\text{{ gram}}$",
    ]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Perhatikan reaksi pembentukan air berikut:\n\n"
        f"$$2\\text{{H}}_2(g) + \\text{{O}}_2(g) \\rightarrow 2\\text{{H}}_2\\text{{O}}(l)$$\n\n"
        f"Jika direaksikan ${mol_h2}\\text{{ mol}}$ gas $\\text{{H}}_2$ dengan oksigen berlebih "
        f"($M_r\\,\\text{{H}}_2\\text{{O}} = {mr_h2o}$), massa air yang terbentuk adalah ...\n\n"
        f"{options}"
    )


def _gen_chemistry_ph(rng: random.Random, number: int) -> str:
    exp = rng.choice([1, 2, 3, 4])
    coeff = rng.choice([1, 2, 5])
    if coeff == 1:
        conc = f"10^{{-{exp}}}"
        correct = f"${exp}$"
        distractors = [f"${exp - 1}$", f"${exp + 1}$", f"${14 - exp}$", f"${exp + 2}$"]
    else:
        conc = f"{coeff} \\times 10^{{-{exp}}}"
        correct = f"${exp} - \\log {coeff}$"
        distractors = [
            f"${exp} + \\log {coeff}$",
            f"${14 - exp} + \\log {coeff}$",
            f"${exp - 1} - \\log {coeff}$",
            f"${exp}$",
        ]
    options = _format_options(rng, correct, distractors)
    return (
        f"### Soal {number}\n\n"
        f"Suatu larutan asam kuat $\\text{{HCl}}$ memiliki konsentrasi "
        f"$[\\text{{H}}^+] = {conc}\\text{{ M}}$. "
        f"Nilai $\\text{{pH}} = -\\log[\\text{{H}}^+]$ larutan tersebut adalah ...\n\n"
        f"{options}"
    )


GENERATOR_REGISTRY: tuple[GeneratorSpec, ...] = (
    GeneratorSpec(
        "math_quadratic",
        Subject.MATHEMATICS,
        "persamaan_kuadrat",
        FormatKind.MULTIPLE_CHOICE,
        _gen_math_quadratic,
    ),
    GeneratorSpec(
        "math_integral",
        Subject.MATHEMATICS,
        "integral_tentu",
        FormatKind.MULTIPLE_CHOICE,
        _gen_math_integral,
    ),
    GeneratorSpec(
        "math_limit",
        Subject.MATHEMATICS,
        "limit_fungsi",
        FormatKind.MULTIPLE_CHOICE,
        _gen_math_limit,
    ),
    GeneratorSpec(
        "math_matrix", Subject.MATHEMATICS, "matriks", FormatKind.MULTIPLE_CHOICE, _gen_math_matrix
    ),
    GeneratorSpec(
        "math_derivative",
        Subject.MATHEMATICS,
        "turunan",
        FormatKind.ESSAY,
        _gen_math_derivative_essay,
    ),
    GeneratorSpec(
        "math_stats_table",
        Subject.MATHEMATICS,
        "statistika",
        FormatKind.TABLE_PROBLEM,
        _gen_math_stats_table,
    ),
    GeneratorSpec(
        "physics_glbb",
        Subject.PHYSICS,
        "kinematika_glbb",
        FormatKind.MULTIPLE_CHOICE,
        _gen_physics_glbb,
    ),
    GeneratorSpec(
        "physics_ohm_table",
        Subject.PHYSICS,
        "listrik_dinamis",
        FormatKind.TABLE_PROBLEM,
        _gen_physics_ohm_table,
    ),
    GeneratorSpec(
        "physics_energy",
        Subject.PHYSICS,
        "usaha_energi",
        FormatKind.ESSAY,
        _gen_physics_energy_essay,
    ),
    GeneratorSpec(
        "chemistry_stoich",
        Subject.CHEMISTRY,
        "stoikiometri",
        FormatKind.MULTIPLE_CHOICE,
        _gen_chemistry_stoichiometry,
    ),
    GeneratorSpec(
        "chemistry_ph",
        Subject.CHEMISTRY,
        "larutan_asam_basa",
        FormatKind.MULTIPLE_CHOICE,
        _gen_chemistry_ph,
    ),
)


def _assign_split(index: int, total: int, train_ratio: float, val_ratio: float) -> Split:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return Split.TRAIN
    if index < val_end:
        return Split.VAL
    return Split.TEST


def generate_dataset(
    count: int = 300,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> list[StemSample]:
    """Generate a deterministic, deduplicated Indonesian STEM OCR dataset."""
    if count <= 0:
        raise ValueError("count must be positive")
    if train_ratio + val_ratio >= 1.0 or train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio and val_ratio must leave room for test split")

    master_rng = random.Random(seed)
    layouts = list(LayoutStyle)
    augmentations = list(AugmentationProfile)

    samples: list[StemSample] = []
    seen_hashes: set[str] = set()
    attempt = 0

    while len(samples) < count:
        spec = GENERATOR_REGISTRY[len(samples) % len(GENERATOR_REGISTRY)]
        sample_seed = master_rng.randint(0, 2**31 - 1) + attempt
        sample_rng = random.Random(sample_seed)
        question_number = (len(samples) % 40) + 1
        markdown = spec.fn(sample_rng, question_number)
        digest = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        attempt += 1

        # Deduplicate content across splits; allow fallback after many retries on small templates.
        if digest in seen_hashes and attempt < count * 8:
            continue
        seen_hashes.add(digest)

        idx = len(samples)
        split = _assign_split(idx, count, train_ratio, val_ratio)
        layout = layouts[idx % len(layouts)]
        augmentation = augmentations[idx % len(augmentations)]
        sample_id = f"indostem_{idx + 1:05d}"

        samples.append(
            StemSample(
                id=sample_id,
                split=split,
                subject=spec.subject,
                topic=spec.topic,
                template_id=spec.template_id,
                format_kind=spec.format_kind,
                layout=layout,
                augmentation=augmentation,
                markdown=markdown,
                seed=sample_seed,
            )
        )

    return samples
