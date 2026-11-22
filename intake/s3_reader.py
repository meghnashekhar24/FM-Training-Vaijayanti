"""Read the batch3 10k dataset from S3 into pandas.

    from s3_reader import S3Reader
    reader = S3Reader()
    df = reader.read_file(reader.list_files()[0])
    all_df = reader.read_all()

    python s3_reader.py            # prints and saves s3_reader_output.txt

Needs: boto3, pandas (pyarrow for parquet, openpyxl for xlsx).
"""
import argparse
import io
import os
from urllib.parse import urlparse

import boto3
import pandas as pd

DEFAULT_S3_URI = (
    "s3://ld-paypalb1-data-448049797912/Learning and Development/"
    "Batches/batch3_data_10k/10k_data/"
)

READERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda f, **kw: pd.read_csv(f, sep="\t", **kw),
    ".json": pd.read_json,
    ".jsonl": lambda f, **kw: pd.read_json(f, lines=True, **kw),
    ".parquet": pd.read_parquet,
    ".xlsx": pd.read_excel,
}


class S3Reader:
    def __init__(self, uri=DEFAULT_S3_URI, profile=None, region=None):
        url = urlparse(uri)
        if url.scheme != "s3":
            raise ValueError(f"Not an S3 URI: {uri}")
        self.bucket = url.netloc
        self.prefix = url.path.lstrip("/")
        self.s3 = boto3.Session(profile_name=profile, region_name=region).client("s3")

    def list_files(self):
        """Object keys under the prefix, folders skipped."""
        pages = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.prefix)
        return [o["Key"] for p in pages for o in p.get("Contents", [])
                if not o["Key"].endswith("/")]

    def read_file(self, key, **kwargs):
        """One object -> DataFrame, picked by file extension."""
        ext = os.path.splitext(key.lower())[1]
        if ext not in READERS:
            raise ValueError(f"Unsupported file type: {key}")
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return READERS[ext](io.BytesIO(body), **kwargs)

    def read_all(self, **kwargs):
        """Every supported file, concatenated, with a source_file column."""
        frames = []
        for key in self.list_files():
            try:
                frames.append(self.read_file(key, **kwargs).assign(
                    source_file=os.path.basename(key)))
            except ValueError as e:
                print(f"Skipping {key}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(reader):
    """One block per file: short name, rows, columns, nulls, memory."""
    lines = []
    for key in reader.list_files():
        name = "/".join(key.split("/")[-2:])  # folder/file.parquet
        try:
            df = reader.read_file(key)
        except ValueError as e:
            lines.append(f"{name}: skipped ({e})\n")
            continue
        lines.append(
            f"{name}\n"
            f"  rows: {len(df)}\n"
            f"  columns: {df.shape[1]}\n"
            f"  column names: {', '.join(df.columns)}\n"
            f"  null cells: {int(df.isna().sum().sum())}\n"
            f"  memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize files under an S3 prefix")
    ap.add_argument("--uri", default=DEFAULT_S3_URI)
    ap.add_argument("--profile")
    ap.add_argument("--region")
    ap.add_argument("--out", default="s3_reader_output.txt")
    args = ap.parse_args()

    text = summarize(S3Reader(args.uri, args.profile, args.region))
    print(text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
