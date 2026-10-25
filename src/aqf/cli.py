"""Command-line entry point: ``aqf <stage>`` (or ``python -m aqf.cli <stage>``)."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import ROOT, load_config


def cmd_discover(cfg, args):
    from .data.openaq import discover_stations

    st = discover_stations(cfg["ncr_bbox"])
    st.to_csv(ROOT / "configs" / "stations.csv", index=False)
    print(f"{len(st)} stations written to configs/stations.csv")


def cmd_stations(cfg, args):
    from .data.openaq import download_station

    st = pd.read_csv(ROOT / "configs" / "stations.csv")
    for loc in st["location_id"]:
        download_station(int(loc), cfg["period"]["start"], cfg["period"]["end"], cfg.path("raw", "openaq", "x").parent)


def cmd_gfs(cfg, args):
    import concurrent.futures as cf

    from .data.gfs import download_day

    g = cfg["gfs"]
    # Start 4 days early so the first forecast origins have history.
    days = pd.date_range(pd.Timestamp(cfg["period"]["start"]) - pd.Timedelta(days=4), cfg["period"]["end"], freq="D")
    out = cfg.path("raw", "gfs", "x").parent
    with cf.ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda d: download_day(d, out, g["cycle"], g["max_lead"], g["grid"]), days))


def cmd_fires(cfg, args):
    from .data.fires import download_firms, download_viirs_noaa

    start = (pd.Timestamp(cfg["period"]["start"]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    out = cfg.path("raw", "fires", "x").parent
    if args.source == "firms":
        download_firms(start, cfg["period"]["end"], cfg["fire_bbox"], out)
    else:
        download_viirs_noaa(start, cfg["period"]["end"], cfg["fire_satellites"], cfg["fire_bbox"], out, workers=args.workers)


def cmd_build(cfg, args):
    from .dataset import build_dataset

    build_dataset(cfg)


def cmd_train(cfg, args):
    from .experiment import run

    run(cfg, models=args.models.split(","), fold=args.fold, epochs=args.epochs)


def cmd_train_final(cfg, args):
    from .serve import train_final

    train_final(cfg, models=tuple(args.models.split(",")))


def cmd_export_app(cfg, args):
    from .serve import add_sensor_ids, export_backtest

    print("backtest data:", export_backtest(cfg))
    print("sensor ids for", len(add_sensor_ids()), "stations")


def cmd_report(cfg, args):
    from .report import make_report

    make_report(cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog="aqf", description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("discover-stations", help="scan the OpenAQ archive for NCR stations (slow)").set_defaults(fn=cmd_discover)
    sub.add_parser("download-stations", help="hourly PM2.5 + co-pollutants per station").set_defaults(fn=cmd_stations)
    s = sub.add_parser("download-gfs", help="GFS 1° forecasts via byte-range requests")
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(fn=cmd_gfs)
    s = sub.add_parser("download-fires", help="VIIRS 375 m active fires")
    s.add_argument("--source", choices=["noaa", "firms"], default="noaa")
    s.add_argument("--workers", type=int, default=8)
    s.set_defaults(fn=cmd_fires)
    sub.add_parser("build", help="align everything into model-ready arrays").set_defaults(fn=cmd_build)
    s = sub.add_parser("train", help="train + evaluate models on a fold")
    s.add_argument(
        "--models",
        default=(
            "persistence,diurnal_persistence,climatology,lightgbm,lightgbm_nofire,lightgbm_nofcwx,"
            "lightgbm_resid,lightgbm_resid_nofire,tft,stgnn,ensemble,tft_anchored,stgnn_anchored,ensemble_anchored"
        ),
    )
    s.add_argument("--fold", default="main", help="fold name from configs/default.yaml split.folds")
    s.add_argument("--epochs", type=int, default=None)
    s.set_defaults(fn=cmd_train)
    sub.add_parser("report", help="figures + markdown summary of all results").set_defaults(fn=cmd_report)
    s = sub.add_parser("train-final", help="fit the serving models on all data into models/final")
    s.add_argument("--models", default="lightgbm_resid,tft_anchored,stgnn_anchored")
    s.set_defaults(fn=cmd_train_final)
    sub.add_parser("export-app", help="write app/backtest.npz and OpenAQ sensor ids into the bundle").set_defaults(fn=cmd_export_app)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.fn(load_config(args.config), args)


if __name__ == "__main__":
    main()
