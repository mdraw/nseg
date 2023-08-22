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


def store_document(
        doc: dict,
        collection_name: str = '_sbxc',
        db_name: str = '_sbxd',
        db_host: str = 'cajalg001',
        replace: bool = True,
):
    with pymongo.MongoClient(db_host) as client:
        coll = client[db_name][collection_name]
        if replace:
            coll.replace_one(
                {},
                doc,
                upsert=True,
            )
        else:
            coll.insert_one(doc)
