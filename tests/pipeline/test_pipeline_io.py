"""I/O format contracts for universal pipeline stages."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.get_mpra import read_wide_row
from src.pipeline.common import (
    ID_CSV_COLUMNS,
    INTERSECT_COLUMNS,
    SPLIT_CSV_COLUMNS,
    parse_marked_header,
    read_csv,
)
from tests.pipeline.conftest import resolve_pipeline_prok
from src.pipeline import (
    id_gen,
    id_rule,
    parse_target,
    adapt,
    parse_data,
    generate_stratification,
    split_predict,
    split_materialize,
    train,
    train_viz,
    adversarial,
)

LEGNET_LEN = 230
DNA = set("ACGTN")


def _pipe_delim_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return next(csv.reader(fh, delimiter="|"))


def test_id_gen_ready_v2_like_structure(mini_raw, ready_v2_mock, tmp_path):
    """Fixture GTF → ID.csv schema."""
    out = id_gen.run_id_gen(
        mini_raw / "gtf", gtf_column="transcript", outdir=tmp_path / "id_out"
    )
    assert out.name == "ID.csv"
    header = _pipe_delim_header(out)
    assert header == ID_CSV_COLUMNS
    rows = read_csv(out)
    assert len(rows) == 3
    ids = [int(r["ID"]) for r in rows]
    assert ids == list(range(1, 4))
    assert len(set(ids)) == len(ids)
    # ready_v2 mock genes appear
    genes = {r["gene_nameORnon_coding_ID"] for r in rows}
    ready_genes = {r["GeneOrID"] for r in read_csv(ready_v2_mock / "ready.csv")}
    assert genes == ready_genes
    for r in rows:
        assert r["genome"].startswith("GCF_")
        assert int(r["pos1"]) <= int(r["pos2"])


def test_id_gen_cds_aggregates_spans_and_prefers_gene_id(mini_raw, tmp_path):
    out = id_gen.run_id_gen(
        mini_raw / "gtf", gtf_column="CDS", outdir=tmp_path / "id_cds"
    )
    assert _pipe_delim_header(out) == ID_CSV_COLUMNS
    rows = read_csv(out)
    assert [(r["raw_target_ID"], int(r["pos1"]), int(r["pos2"])) for r in rows] == [
        ("g1", 60, 140),
        ("g2", 130, 210),
        ("g3", 20, 70),
    ]
    assert [int(r["ID"]) for r in rows] == [1, 2, 3]


def test_id_gen_real_prokaryote_gene_ids(tmp_path):
    """Real RefSeq GTF: E. coli gene IDs are TPM-joinable locus tags."""
    gtf = (
        Path(__file__).resolve().parents[2]
        / "prokaryotes"
        / "gtf"
        / "GCF_000005845.2_ASM584v2_genomic.gtf"
    )
    assert gtf.is_file() and gtf.stat().st_size > 0
    out = id_gen.run_id_gen(gtf, gtf_column="gene", outdir=tmp_path / "id_real")
    assert _pipe_delim_header(out) == ID_CSV_COLUMNS
    rows = read_csv(out)
    assert len(rows) > 4_000
    assert [int(r["ID"]) for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0]["genome"] == "GCF_000005845.2"
    assert rows[0]["gene_nameORnon_coding_ID"] == "thrL"
    assert rows[0]["raw_target_ID"] == "b0001"
    assert all(r["raw_target_ID"] for r in rows)


def test_id_rule_maps_columns(id_csv):
    rows = read_csv(id_csv)
    gene_list = [r["gene_nameORnon_coding_ID"] for r in rows[:2]]
    mapped = id_rule.run_id_rule(
        gene_list, id_csv, id_col_1="gene_nameORnon_coding_ID", id_col_2="ID"
    )
    assert mapped == [rows[0]["ID"], rows[1]["ID"]]
    # reverse
    back = id_rule.run_id_rule(
        mapped, id_csv, id_col_1="ID", id_col_2="gene_nameORnon_coding_ID"
    )
    assert back == gene_list


def _real_prok_id_csv() -> Path:
    """Return the real prokaryote ID table, regenerating it only when absent."""
    root = Path(__file__).resolve().parents[2]
    panel = resolve_pipeline_prok()
    id_csv = (panel / "id_gen" / "ID.csv") if panel else root / "output" / "pipeline_prok" / "id_gen" / "ID.csv"
    if not id_csv.is_file():
        id_gen.run_id_gen(
            root / "prokaryotes" / "gtf",
            gtf_column="gene",
            outdir=id_csv.parent,
        )
    return id_csv


def test_id_rule_real_prokaryote_mappings_and_cli(capsys):
    """Map real 37,791-row prokaryote IDs without fabricated fixtures."""
    id_csv = _real_prok_id_csv()
    rows = read_csv(id_csv)
    assert len(rows) == 37_791

    raw_ids = [row["raw_target_ID"] for row in rows[:5]]
    expected_ids = [row["ID"] for row in rows[:5]]
    assert id_rule.run_id_rule(
        raw_ids + ["not-a-real-target"] + raw_ids[:1],
        id_csv,
        id_col_1="raw_target_ID",
        id_col_2="ID",
    ) == expected_ids + expected_ids[:1]
    assert id_rule.run_id_rule(
        expected_ids,
        id_csv,
        id_col_1="ID",
        id_col_2="raw_target_ID",
    ) == raw_ids

    assert id_rule.main(
        [
            "--ids",
            ",".join(raw_ids),
            "--id-csv",
            str(id_csv),
            "--id-col-1",
            "raw_target_ID",
            "--id-col-2",
            "ID",
        ]
    ) == 0
    assert capsys.readouterr().out.splitlines() == expected_ids

    with pytest.raises(ValueError, match="missing columns"):
        id_rule.run_id_rule(raw_ids, id_csv, id_col_1="missing", id_col_2="also_missing")


def test_generate_stratification_stub_warns_not_implemented(tmp_path):
    """MJ-003: stub warns 'Not implemented' and does not invent stratification.csv."""
    id_csv = tmp_path / "ID.csv"
    prepare = tmp_path / "prepare_strat.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n",
        encoding="utf-8",
    )
    prepare.write_text("identificator|column|strat\n", encoding="utf-8")
    outdir = tmp_path / "out"

    with pytest.warns(UserWarning, match=r"^Not implemented$"):
        with pytest.raises(NotImplementedError, match=r"^Not implemented$"):
            generate_stratification.run_generate_stratification(
                id_csv, prepare, outdir=outdir
            )

    assert not (outdir / "stratification.csv").exists()
    with pytest.warns(UserWarning, match=r"^Not implemented$"):
        assert generate_stratification.main(
            [
                "--id-csv",
                str(id_csv),
                "--prepare-strat",
                str(prepare),
                "--outdir",
                str(outdir),
            ]
        ) == 2
    assert not (outdir / "stratification.csv").exists()


@pytest.mark.parametrize("to_type", ["caduceus", "legnet"])
def test_parse_target_predict_schema(mini_raw, id_csv, tmp_path, to_type):
    outdir = tmp_path / f"pt_{to_type}"
    paths = parse_target.run_parse_target(
        mini_raw / "tpm",
        outdir=outdir,
        id_csv=id_csv,
        input_type="folder",
        to_type=to_type,
    )
    pred_csv = paths["predict_csv"]
    assert pred_csv == outdir / "PREDICT" / "predict.csv"
    header = _pipe_delim_header(pred_csv)
    assert header[0] == "id"
    assert header[1].startswith("predict_var")
    rows = read_csv(pred_csv)
    assert len(rows) == 3
    by_id = {r["id"]: float(r["predict_var1"]) for r in rows}
    # GENEA=1.5 etc from fixture; missing → 0 rule not triggered here
    assert by_id[rows[0]["id"]] in {1.5, 2.5, 3.5}
    for r in rows:
        ext = outdir / "PREDICT" / f"{r['id']}.ext"
        assert ext.is_file()
        val = float(ext.read_text(encoding="utf-8").strip())
        assert val == float(r["predict_var1"])
    # Caduceus labels.tsv-like: id + TPM column semantics
    if to_type == "caduceus":
        assert all("predict_var1" in r for r in rows)
    # LegNet mean_value semantics
    if to_type == "legnet":
        assert all(float(r["predict_var1"]) >= 0 for r in rows)


@pytest.mark.parametrize("to_type", ["caduceus", "legnet"])
def test_parse_target_real_mpra_targets_use_raw_gene_ids(tmp_path, to_type):
    """Real MPRA wide CSV joins RefSeq locus tags, not display gene names."""
    root = Path(__file__).resolve().parents[2]
    id_csv = _real_prok_id_csv()
    target = root / "prokaryotes" / "mpra"
    mpra = target / "GCF_000005845.2_ASM584v2.csv"
    assert mpra.is_file() and mpra.stat().st_size > 0

    paths = parse_target.run_parse_target(
        target, outdir=tmp_path / to_type, id_csv=id_csv, to_type=to_type
    )
    rows = read_csv(paths["predict_csv"])
    assert len(rows) == 37_791
    assert _pipe_delim_header(paths["predict_csv"]) == ["id", "predict_var1"]

    genes, values = read_wide_row(mpra)
    expected = dict(zip(genes, values))
    first = rows[0]
    assert first["id"] == "1"
    assert float(first["predict_var1"]) == expected["b0001"]
    assert float((paths["predict_dir"] / "1.ext").read_text().strip()) == expected["b0001"]


def test_parse_target_mapping_uses_sample_subtrees_and_mpra_basenames(tmp_path):
    """One real mapping row yields sample-scoped MPRA predictions."""
    root = Path(__file__).resolve().parents[2]
    source_mappings = root / "prokaryotes" / "expr_file_mappings.csv"
    mapping_row = next(csv.DictReader(source_mappings.open(encoding="utf-8")))
    mappings = tmp_path / "one_mapping.csv"
    with mappings.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "tpm", "genome"])
        writer.writeheader()
        writer.writerow({key: mapping_row[key] for key in writer.fieldnames})

    paths = parse_target.run_parse_target(
        root / "prokaryotes" / "mpra",
        outdir=tmp_path / "mapped",
        id_csv=_real_prok_id_csv(),
        to_type="caduceus",
        mappings=mappings,
    )
    rows = read_csv(paths["predict_csv"])
    assert _pipe_delim_header(paths["predict_csv"]) == ["id", "sample_id", "predict_var1"]
    assert rows and {row["sample_id"] for row in rows} == {mapping_row["id"]}
    assert len(rows) > 4_000
    first = rows[0]
    ext = paths["predict_dir"] / mapping_row["id"] / f"{first['id']}.ext"
    assert ext.is_file()
    assert float(ext.read_text().strip()) == float(first["predict_var1"])


def test_parse_target_calls_id_rule_for_header_remaps(mini_raw, id_csv, tmp_path, monkeypatch):
    """parse_target resolves TARGET headers through run_id_rule (MJ-001)."""
    calls: list[tuple[list[str], str, str]] = []
    real = parse_target.run_id_rule

    def spy(id_list, path, *, id_col_1, id_col_2):
        calls.append((list(id_list), id_col_1, id_col_2))
        return real(id_list, path, id_col_1=id_col_1, id_col_2=id_col_2)

    monkeypatch.setattr(parse_target, "run_id_rule", spy)
    paths = parse_target.run_parse_target(
        mini_raw / "tpm", outdir=tmp_path / "pt_id_rule", id_csv=id_csv, to_type="caduceus"
    )
    assert calls
    assert all(c[2] == "ID" for c in calls)
    cols = {c[1] for c in calls}
    assert "raw_target_ID" in cols
    # Mini TPM headers are display names (GENEA…), so gene_name fallback is exercised.
    assert "gene_nameORnon_coding_ID" in cols
    by_id = {r["id"]: float(r["predict_var1"]) for r in read_csv(paths["predict_csv"])}
    assert by_id == {"1": 1.5, "2": 2.5, "3": 3.5}


def test_parse_target_gene_name_remap_via_id_rule_absent_zero(tmp_path):
    """gene_name TARGET headers map via id_rule; unmatched panel IDs stay 0."""
    id_csv = tmp_path / "ID.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n"
        "GCF_X.1|chr1|1|10|thrL|b0001|1\n"
        "GCF_X.1|chr1|20|30|thrA|b0002|2\n"
        "GCF_X.1|chr1|40|50|orphan|b0099|3\n",
        encoding="utf-8",
    )
    target_dir = tmp_path / "tpm"
    target_dir.mkdir()
    # Headers use gene_name (not locus tags); orphan gene absent → prediction 0.
    (target_dir / "GCF_X.1.csv").write_text("thrL,thrA\n1.25,2.5\n", encoding="utf-8")

    paths = parse_target.run_parse_target(
        target_dir, outdir=tmp_path / "out", id_csv=id_csv, to_type="caduceus"
    )
    by_id = {r["id"]: float(r["predict_var1"]) for r in read_csv(paths["predict_csv"])}
    assert by_id == {"1": 1.25, "2": 2.5, "3": 0.0}
    assert float((paths["predict_dir"] / "3.ext").read_text().strip()) == 0.0


def test_parse_target_prefers_raw_target_id_remap_via_id_rule(tmp_path):
    """When both raw and gene keys exist, raw_target_ID remap wins (same as prior join)."""
    id_csv = tmp_path / "ID.csv"
    id_csv.write_text(
        "genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID\n"
        "GCF_X.1|chr1|1|10|thrL|b0001|1\n",
        encoding="utf-8",
    )
    target_dir = tmp_path / "tpm"
    target_dir.mkdir()
    (target_dir / "GCF_X.1.csv").write_text("b0001,thrL\n9.0,1.0\n", encoding="utf-8")

    paths = parse_target.run_parse_target(
        target_dir, outdir=tmp_path / "out", id_csv=id_csv, to_type="caduceus"
    )
    rows = read_csv(paths["predict_csv"])
    assert len(rows) == 1
    assert float(rows[0]["predict_var1"]) == 9.0


def test_adapt_marked_and_intersect(mini_raw, id_csv, tmp_path):
    outdir = tmp_path / "adapt_gene"
    paths = adapt.run_adapt(
        mini_raw / "gtf",
        mini_raw / "fna",
        outdir=outdir,
        id_csv=id_csv,
        environment="gene",
        window={"pos1": -10, "pos2": 10},
        max_window=None,
    )
    assert paths["intersect_csv"].name == "intersect.csv"
    ih = _pipe_delim_header(paths["intersect_csv"])
    assert ih == INTERSECT_COLUMNS
    marked = list((outdir / "MARKED").glob("*.fa"))
    assert marked
    for fa in marked:
        lines = fa.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith(">|")
        meta = parse_marked_header(lines[0])
        for k in (
            "genome",
            "chr",
            "pos1",
            "pos2",
            "gene_nameORnon_coding_ID",
            "raw_target_ID",
            "ID",
        ):
            assert meta[k]
        seq = "".join(lines[1:])
        assert seq
        assert set(seq) <= DNA


@pytest.mark.parametrize("to_type", ["caduceus", "legnet"])
def test_adapt_real_prokaryote_pm100_window(tmp_path, to_type):
    """Real E. coli GTF/FNA + IDs produce 200-base anchor-excluding windows."""
    root = Path(__file__).resolve().parents[2]
    outdir = tmp_path / f"adapt_{to_type}"
    paths = adapt.run_adapt(
        root / "prokaryotes" / "gtf",
        root / "prokaryotes" / "fna",
        outdir=outdir,
        id_csv=_real_prok_id_csv(),
        environment="gene",
        window={"pos1": -100, "pos2": 100},
        max_window=None,
        genomes=["GCF_000005845.2"],
    )
    assert _pipe_delim_header(paths["intersect_csv"]) == INTERSECT_COLUMNS
    marked = sorted((outdir / "MARKED").glob("*.fa"))
    assert len(marked) > 4_000
    lines = marked[1].read_text(encoding="utf-8").splitlines()
    meta = parse_marked_header(lines[0])
    assert all(meta[field] for field in ("genome", "chr", "pos1", "pos2", "gene_nameORnon_coding_ID", "raw_target_ID", "ID"))
    assert len("".join(lines[1:])) == 200


@pytest.mark.parametrize("to_type", ["caduceus", "legnet"])
def test_parse_data_model_ready_formats(tmp_path, to_type):
    """Caduceus stays raw; LegNet uses the canonical 200 bp adapter stitch."""
    marked = tmp_path / "MARKED"
    marked.mkdir()
    crs = ("ACGTN" * 40)
    (marked / "7.fa").write_text(
        ">||chr1|100|300|gene7|raw7|7\n" + crs + "\n",
        encoding="utf-8",
    )

    parsed = parse_data.run_parse_data(marked, outdir=tmp_path / to_type, to_type=to_type)
    body = (parsed / "7.ext").read_text(encoding="utf-8").strip()
    if to_type == "caduceus":
        assert body == crs
        assert set(body) <= DNA
        assert len(body) == 200
    else:
        assert len(body) == LEGNET_LEN
        assert body.startswith("AGGACCGGATCAACT")
        assert body.endswith("CATTGCGTGAACCGA")
        assert body[15:-15] == crs


def test_parse_data_real_marked_subset(tmp_path):
    """parse_data on actual 200 bp MARKED files (caduceus + legnet)."""
    root = Path(__file__).resolve().parents[2]
    panel = resolve_pipeline_prok() or (root / "output" / "pipeline_prok")
    marked = panel / "adapt_gene_pm100" / "MARKED"
    files = sorted(marked.glob("*.fa"))[:3]
    assert len(files) == 3

    parsed = parse_data.run_parse_data(files[0], outdir=tmp_path / "leg", to_type="legnet")
    assert len(list(parsed.glob("*.ext"))) == 1
    assert len(next(parsed.glob("*.ext")).read_text().strip()) == LEGNET_LEN

    parsed = parse_data.run_parse_data(marked / files[1].name, outdir=tmp_path / "real", to_type="caduceus")
    meta, sequence = parse_data.read_marked_fasta(files[1])
    assert (parsed / f"{meta['ID']}.ext").read_text().strip() == sequence
    assert len(sequence) == 200



def test_parse_data_rejects_unexpected_legnet_length(tmp_path):
    from src.pipeline import parse_data
    import pytest

    marked = tmp_path / "short.fa"
    marked.write_text(">|g|c|1|10|g|r|9\nACGTACGTAC\n", encoding="utf-8")
    # default: skip incomplete
    out = parse_data.run_parse_data(marked, outdir=tmp_path / "ok", to_type="legnet")
    assert list(out.glob("*.ext")) == []
    # strict: raise
    with pytest.raises(ValueError, match="200 bp"):
        parse_data.run_parse_data(
            marked, outdir=tmp_path / "strict", to_type="legnet", skip_incomplete_legnet=False
        )


def test_split_predict_and_split_materialize(mini_raw, id_csv, tmp_path):
    pt = tmp_path / "panel"
    parse_target.run_parse_target(mini_raw / "tpm", outdir=pt, id_csv=id_csv, to_type="caduceus")
    adapt.run_adapt(
        mini_raw / "gtf", mini_raw / "fna", outdir=pt, id_csv=id_csv,
        environment="gene", window={"pos1": -10, "pos2": 10}, max_window=None,
    )
    parse_data.run_parse_data(pt / "MARKED", outdir=pt, to_type="caduceus")

    # optional fold/strat tables
    rows = read_csv(id_csv)
    fold_path = tmp_path / "fold.csv"
    with fold_path.open("w", encoding="utf-8") as fh:
        fh.write("genome|chr|pos1|pos2|gene_nameORnon_coding_ID|raw_target_ID|ID|fold\n")
        for i, r in enumerate(rows):
            fh.write(
                f"{r['genome']}|{r['chr']}|{r['pos1']}|{r['pos2']}|{r['gene_nameORnon_coding_ID']}|{r['raw_target_ID']}|{r['ID']}|{i%2}\n"
            )
    strat_path = tmp_path / "strat.csv"
    with strat_path.open("w", encoding="utf-8") as fh:
        fh.write("ID|strat1\n")
        for r in rows:
            fh.write(f"{r['ID']}|A\n")

    split_csv = split_predict.run_split_predict(
        outdir=pt,
        type="random",
        seed=42,
        id_csv=id_csv,
        fold_csv=fold_path,
        stratification_csv=strat_path,
    )
    assert _pipe_delim_header(split_csv) == SPLIT_CSV_COLUMNS
    srows = read_csv(split_csv)
    assert {r["train_test"] for r in srows} <= {"train", "test", "val"}
    assert all(r["ID"] and r["fold"] != "" for r in srows)

    split_root = split_materialize.run_split(
        split_csv, pt / "PREDICT", pt / "PARSED", outdir=pt, strategy="traintestval"
    )
    assert (split_root / "PREDICT").is_dir()
    assert (split_root / "FASTA").is_dir()
    buckets = [p.name for p in (split_root / "PREDICT").iterdir() if p.is_dir()]
    assert buckets
    for b in buckets:
        assert (split_root / "PREDICT" / b / "predict.csv").is_file()
        assert list((split_root / "PREDICT" / b).glob("*.ext"))
        assert list((split_root / "FASTA" / b).glob("*.ext"))


def test_train_caduceus_legnet_smoke(mini_raw, id_csv, tmp_path):
    pt = tmp_path / "panel"
    parse_target.run_parse_target(mini_raw / "tpm", outdir=pt, id_csv=id_csv, to_type="caduceus")
    adapt.run_adapt(
        mini_raw / "gtf", mini_raw / "fna", outdir=pt, id_csv=id_csv,
        environment="gene", window={"pos1": -10, "pos2": 10}, max_window=None,
    )
    parse_data.run_parse_data(pt / "MARKED", outdir=pt, to_type="caduceus")
    split_predict.run_split_predict(outdir=pt, type="random", seed=42, id_csv=id_csv)
    split_materialize.run_split(
        pt / "split.csv", pt / "PREDICT", pt / "PARSED", outdir=pt, strategy="traintestval"
    )

    cad_out = train.run_train(
        model="caduceus", type="regression", folders=pt / "SPLIT", outdir=tmp_path / "runs_cad", smoke=True
    )
    assert (cad_out / "logs" / "train_metrics.jsonl").is_file()

    # LegNet smoke: build a minimal all.tsv from PARSED + PREDICT
    tsv = tmp_path / "all.tsv"
    pred = {r["id"]: r["predict_var1"] for r in read_csv(pt / "PREDICT" / "predict.csv")}
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("seq_id\tseq\tmean_value\tfold\trev\n")
        for i, ext in enumerate(sorted((pt / "PARSED").glob("*.ext")), start=1):
            # pad/crop via parse_data legnet path
            pass
    # regenerate legnet parsed
    parse_data.run_parse_data(pt / "MARKED", outdir=pt / "leg", to_type="legnet")
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("seq_id\tseq\tmean_value\tfold\trev\n")
        for ext in sorted((pt / "leg" / "PARSED").glob("*.ext")):
            seq = ext.read_text(encoding="utf-8").strip()
            assert len(seq) == LEGNET_LEN
            fh.write(f"{ext.stem}\t{seq}\t{pred.get(ext.stem, '0')}\t{(hash(ext.stem)%10)+1}\t0\n")
    leg_out = train.run_train(
        model="legnet", type="regression", folders=tsv, outdir=tmp_path / "runs_leg", smoke=True
    )
    assert (leg_out / "logs" / "train_metrics.jsonl").is_file()


def test_train_tiny_split_links_real_prokaryote_artifacts(tmp_path):
    """Tiny Caduceus input remains a subset of real SPLIT artifacts."""
    import os

    root = Path(__file__).resolve().parents[2]
    panel = resolve_pipeline_prok() or (root / "output" / "pipeline_prok")
    source = panel / "split_caduceus" / "SPLIT"
    tiny = train.materialize_tiny_split(
        source, outdir=tmp_path / "tiny", counts={"train": 2, "val": 1, "test": 1}
    )
    assert tiny == tmp_path / "tiny" / "SPLIT"
    for source_fold, expected in (("TRAIN", 2), ("VAL", 1), ("TEST", 1)):
        files = sorted((tiny / "FASTA" / source_fold).glob("*.ext"))
        assert len(files) == expected
        assert os.path.samefile(files[0], source / "FASTA" / source_fold / files[0].name)
        assert len(read_csv(tiny / "PREDICT" / source_fold / "predict.csv")) == expected


def test_train_viz_and_adversarial(mini_raw, id_csv, tmp_path):
    pt = tmp_path / "panel"
    parse_target.run_parse_target(mini_raw / "tpm", outdir=pt, id_csv=id_csv, to_type="caduceus")
    adapt.run_adapt(
        mini_raw / "gtf", mini_raw / "fna", outdir=pt, id_csv=id_csv,
        environment="gene", window={"pos1": -10, "pos2": 10}, max_window=None,
    )
    parse_data.run_parse_data(pt / "MARKED", outdir=pt, to_type="caduceus")
    split_predict.run_split_predict(outdir=pt, type="random", seed=42, id_csv=id_csv)

    split_materialize.run_split(
        pt / "split.csv", pt / "PREDICT", pt / "PARSED", outdir=pt, strategy="traintestval"
    )
    run_dir = train.run_train(
        model="caduceus", type="regression", folders=pt / "SPLIT", outdir=tmp_path / "run2", smoke=True
    )
    viz = train_viz.run_train_viz(run_dir, outdir=tmp_path / "viz")
    assert (viz / "loss_curve.svg").is_file()
    assert (viz / "training_summary.json").is_file()

    adv = adversarial.run_adversarial(outdir=pt, outdir_new=tmp_path / "adv")
    assert (adv / "PREDICT" / "predict.csv").is_file()
    assert (adv / "PARSED").is_dir()
    assert (adv / "split.csv").is_file()
