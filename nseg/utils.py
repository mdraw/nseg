import json

from typing import Any, Optional

import numpy as np
import zstandard
import pymongo


def np_savezstd(fp: str, level=5, *args, **kwargs):
    """
    Save numpy arrays to a file using zstd compression.
    Like np.savez_compressed(), but with zstd instead of gzip.
    """

    with open(fp, 'wb') as f:
        cctx = zstandard.ZstdCompressor(level=level)
        with cctx.stream_writer(f) as comp:
            np.savez(comp, *args, **kwargs)


def np_loadzstd(fp: str):
    """
    Load numpy arrays from a file using zstd decompression.
    Like np.load(), but with zstd instead of gzip.
    """

    with open(fp, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        return np.load(dctx.decompress(f.read()))
        # with dctx.stream_reader(f) as decomp:
        #     return numpy.load(decomp)


def upsert_document(
        doc: dict[str, Any],
        filt: Optional[dict[str, Any]] = None,
        collection_name: str = '_sandbox',
        db_name: str = '_sandbox',
        db_host: str = 'cajalg001',
        replace: bool = True,
):
    filt = filt or {}  # Default to no filter (i.e. unconditional insert)
    # Convert all possibly non-string keys of possibly nested dict to strings for db compat
    doc = json.loads(json.dumps(doc))

    with pymongo.MongoClient(db_host) as client:
        coll = client[db_name][collection_name]
        coll.replace_one(
            filter=filt,
            replacement=doc,
            upsert=True,
        )


def read_document(
        filt: Optional[dict[str, Any]] = None,
        collection_name: str = '_sandbox',
        db_name: str = '_sandbox',
        db_host: str = 'cajalg001',
):
    filt = filt or {}
    with pymongo.MongoClient(db_host) as client:
        coll = client[db_name][collection_name]
        return coll.find_one(filter=filt)


def read_documents(
        filt: Optional[dict[str, Any]] = None,
        collection_name: str = '_sandbox',
        db_name: str = '_sandbox',
        db_host: str = 'cajalg001',
):
    filt = filt or {}
    with pymongo.MongoClient(db_host) as client:
        coll = client[db_name][collection_name]
        return list(coll.find(filter=filt))
